# encoding=utf-8

import argparse
import time

from others.logging import init_logger
from others.utils import str2bool, _get_word_ngrams, mkdir
import gc
import glob
import hashlib
import json
import os
import re
import subprocess
from os.path import join as pjoin
import torch
from multiprocess import Pool
from others.logging import logger
from others.tokenization import BertTokenizer


def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size=3):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    # 把for s in range(len(summary_size)) 改成 for s in range(len(abstract_sent_list)) 消除hard code
    for s in range(len(abstract_sent_list)):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, real_sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if (not is_test) and len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        _real_sent_labels = [0] * len(src)
        for l in real_sent_labels:
            _real_sent_labels[l] = 1

        # 增加一个real_labels变量 用于保存对应于src_str的label
        real_labels = [_real_sent_labels[i] for i in idxs]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if (not is_test) and len(src) < self.args.min_src_nsents:
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if (not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens:
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, real_labels


def format_to_bert_mp(args):
    if args.dataset != '':
        datasets = [args.dataset]
    else:
        datasets = ['train', 'val', 'test']

    for corpus_type in datasets:
        a_lst = []
        for jsonl_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.jsonl')):
            real_name = jsonl_f.split('/')[-1]
            a_lst.append((jsonl_f, pjoin(args.save_path, real_name.replace('jsonl', 'bert.jsonl'))))
        if len(a_lst) != 1:
            raise RuntimeError(f"文件夹里面包含多个命中为 *{corpus_type}*.jsonl 的文件或无此文件")

        jsonl_file, save_file = a_lst[0]
        logger.info('Processing %s' % jsonl_file)
        jsonl_insts = load_jsonl(jsonl_file)
        is_test = [(corpus_type == 'test')] * len(jsonl_insts)

        with Pool(args.n_cpu) as p:
            formatted_insts = p.map(format_to_bert, zip(jsonl_insts, is_test))
        logger.info('Processed instances %d' % len(formatted_insts))
        logger.info('Saving to %s' % save_file)

        with open(save_file, "w") as f:
            for inst in formatted_insts:
                if inst is not None:
                    print(json.dumps(inst), file=f)
        gc.collect()


def format_to_bert(args):
    logger.info("Process#: {}".format(os.getppid()))
    inst, is_test = args

    source = [sent.split() for sent in inst['text']]
    tgt = [sent.split() for sent in inst['summary']]

    sent_labels = greedy_selection(source[:parsed_args.max_src_nsents], tgt)
    real_sent_labels = greedy_selection(source, tgt)
    if parsed_args.lower:
        source = [' '.join(s).lower().split() for s in source]
        tgt = [' '.join(s).lower().split() for s in tgt]

    b_data = bert.preprocess(source, tgt, sent_labels, real_sent_labels,
                             use_bert_basic_tokenizer=parsed_args.use_bert_basic_tokenizer,
                             is_test=is_test)

    if b_data is None:
        return None

    src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, real_labels = b_data
    b_data_dict = {"text_ids": src_subtoken_idxs, "summary_ids": tgt_subtoken_idxs,
                   "label": sent_labels, "segment_ids": segments_ids, 'cls_ids': cls_ids,
                   'text': src_txt, "summary": tgt_txt, 'real_label': real_labels}
    return b_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='../jsonl_data')
    parser.add_argument('--save_path', default='../data/')
    # parser.add_argument('--has_label', default=True)
    parser.add_argument('--min_src_nsents', default=3, type=int)
    parser.add_argument('--max_src_nsents', default=50, type=int)
    parser.add_argument('--min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('--max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('--min_tgt_ntokens', default=5, type=int)
    parser.add_argument('--max_tgt_ntokens', default=400, type=int)

    parser.add_argument("--lower", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_bert_basic_tokenizer", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--log_file', default='../logs/default.log')

    parser.add_argument('--dataset', default='')

    parser.add_argument('--n_cpu', default=32, type=int)

    parsed_args = parser.parse_args()

    logger = init_logger(parsed_args.log_file)
    bert = BertData(parsed_args)

    mkdir(parsed_args.save_path)
    logger.info(time.clock())
    format_to_bert_mp(parsed_args)
    logger.info(time.clock())
