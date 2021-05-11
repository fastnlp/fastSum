import os
import re
import shutil
import time

# from others import pyrouge
# import pyrouge
import argparse

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

ROOT = "root/path"
_ROUGE_PATH = '/path/to/RELEASE-1.5.5'

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )


import os
from os.path import exists


def get_data_path(mode, label_type):
    '''
    :param mode: train, validate or test
    :param label_type: dataset dir name
    :return:
    '''
    paths = {}
    if mode == 'train':
        paths['train'] = 'data/' + label_type + '/bert.train.jsonl'
        paths['val'] = 'data/' + label_type + '/bert.val.jsonl'
    else:
        paths['test'] = 'data/' + label_type + '/bert.test.jsonl'
    return paths


def get_rouge_path(label_type):
    # if label_type == 'others':
    #     data_path = 'data/' + label_type + '/bert.test.jsonl'
    # else:
    #     data_path = 'data/' + label_type + '/test.jsonl'
    data_path = 'data/' + label_type + '/bert.test.jsonl'
    dec_path = 'dec'
    ref_path = 'ref'
    if not exists(ref_path):
        os.makedirs(ref_path)
    if not exists(dec_path):
        os.makedirs(dec_path)
    return data_path, dec_path, ref_path


def configure_training(args):
    devices = [int(gpu) for gpu in args.visible_gpus.split(',')]
    params = {}
    params['task'] = args.task
    params['encoder'] = args.encoder
    params['mode'] = args.mode
    params['max_pos'] = args.max_pos
    params['sep_optim'] = args.sep_optim
    params['beam_size'] = args.beam_size
    params['batch_size'] = args.batch_size
    params['accum_count'] = args.accum_count
    params['lr_bert'] = args.lr_bert
    params['lr_dec'] = args.lr_dec
    params['lr'] = args.lr
    params['warmup_steps'] = args.warmup_steps
    params['warmup_steps_bert'] = args.warmup_steps_bert
    params['warmup_steps_dec'] = args.warmup_steps_dec
    params['n_epochs'] = args.n_epochs
    # params['valid_steps'] = args.valid_steps
    return devices, params


import re
import os
import shutil
import copy
import datetime
import numpy as np
from rouge import Rouge
import random
import tempfile
import os
import pyrouge
import logging
import tensorflow as tf
import datetime

import sys

sys.setrecursionlimit(10000)

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def ranstr(num):
    # 猜猜变量名为啥叫 H,生成随机字符串
    H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

    salt = ''
    for i in range(num):
        salt += random.choice(H)

    return salt


def clean(x):
    return re.sub(r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", lambda m: REMAP.get(m.group()), x)


def pyrouge_score_all(hyps_list, refer_list, config, tmp_path, remap=True):
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    tempfile.tempdir = tmp_path
    assert not os.path.exists(tmp_path)
    os.mkdir(tmp_path)

    PYROUGE_ROOT = os.path.join(ROOT, 'tmp', nowTime + "_" + ranstr(6))
    while os.path.exists(PYROUGE_ROOT):
        PYROUGE_ROOT = os.path.join(ROOT, 'tmp', nowTime + "_" + ranstr(6))
    assert not os.path.exists(PYROUGE_ROOT), "pyrouge root already exist!"

    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'model')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')

    os.makedirs(SYSTEM_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    gold_path = os.path.join(config.decode_path, "gold.txt")
    pred_path = os.path.join(config.decode_path, "pred.txt")

    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Gold.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

        with open(gold_path, 'a') as f:
            f.write(refer.replace("\n", " "))
            f.write("\n")
        with open(pred_path, 'a') as f:
            f.write(hyps.replace("\n", " "))
            f.write("\n")

    r = pyrouge.Rouge155(_ROUGE_PATH)

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Gold.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate(rouge_args="-e {}/data -a -m -n 2 -d".format(_ROUGE_PATH))
    output_dict = r.output_to_dict(output)

    shutil.rmtree(PYROUGE_ROOT)
    shutil.rmtree(tmp_path)

    scores = {}
    scores['rouge_1_precision'], scores['rouge_1_recall'], scores['rouge_1_f_score'] = output_dict['rouge_1_precision'], \
                                                                                       output_dict['rouge_1_recall'], \
                                                                                       output_dict[
                                                                                           'rouge_1_f_score']
    scores['rouge_2_precision'], scores['rouge_2_recall'], scores['rouge_2_f_score'] = output_dict['rouge_2_precision'], \
                                                                                       output_dict['rouge_2_recall'], \
                                                                                       output_dict[
                                                                                           'rouge_2_f_score']
    scores['rouge_l_precision'], scores['rouge_l_recall'], scores['rouge_l_f_score'] = output_dict['rouge_l_precision'], \
                                                                                       output_dict['rouge_l_recall'], \
                                                                                       output_dict[
                                                                                           'rouge_l_f_score']
    return scores


def pyrouge_score_all_multi(hyps_list, refer_list, config, tmp_path, remap=True):
    # 暂时不能用，使用时需要检查过
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    tempfile.tempdir = tmp_path
    assert not os.path.exists(tmp_path)
    os.mkdir(tmp_path)

    PYROUGE_ROOT = os.path.join(ROOT, 'tmp', nowTime + "_" + ranstr(6))
    while os.path.exists(PYROUGE_ROOT):
        PYROUGE_ROOT = os.path.join(ROOT, 'tmp', nowTime + "_" + ranstr(6))
    assert not os.path.exists(PYROUGE_ROOT), "pyrouge root already exist!"

    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'model')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')

    os.makedirs(SYSTEM_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)

    pred_path = os.path.join(config.decode_path, "pred.txt")

    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        # model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))

        referType = ["A", "B", "C", "D", "E", "F", "G"]

        for j in range(len(refer_list[i])):
            model_file = os.path.join(MODEL_PATH, "Gold.%s.%d.txt" % (referType[j], i))
            refer = clean(refer_list[i][j]) if remap else refer_list[i][j]
            with open(model_file, 'wb') as f:
                f.write(refer.encode('utf-8'))

            gold_path = os.path.join(config.decode_path, "gold.%s.txt" % (referType[j]))
            with open(gold_path, 'a') as f:
                f.write(refer.replace("\n", " "))
                f.write("\n")

        with open(pred_path, 'a') as f:
            f.write(hyps.replace("\n", " "))
            f.write("\n")

    r = pyrouge.Rouge155()

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Gold.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)

    shutil.rmtree(PYROUGE_ROOT)
    shutil.rmtree(tmp_path)

    scores = {}
    scores["rouge_1_precision"], scores['rouge_1_recall'], scores['rouge_1_f_score'] = output_dict['rouge_1_precision'], \
                                                                                       output_dict['rouge_1_recall'], \
                                                                                       output_dict[
                                                                                           'rouge_1_f_score']
    scores['rouge_2_precision'], scores['rouge_2_recall'], scores['rouge_2_f_score'] = output_dict['rouge_2_precision'], \
                                                                                       output_dict['rouge_2_recall'], \
                                                                                       output_dict[
                                                                                           'rouge_2_f_score']
    scores['rouge_l_precision'], scores['rouge_l_recall'], scores['rouge_l_f_score'] = output_dict['rouge_l_precision'], \
                                                                                       output_dict['rouge_l_recall'], \
                                                                                       output_dict[
                                                                                           'rouge_l_f_score']
    return scores
