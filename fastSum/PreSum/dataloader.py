import bisect
from time import time
from datetime import timedelta

from fastNLP.io.loader import JsonLoader
from fastNLP.modules.tokenizer import BertTokenizer
from fastNLP.io.data_bundle import DataBundle
from fastNLP.core.const import Const


class BertData(JsonLoader):

    def __init__(self, max_nsents=60, max_ntokens=100, max_len=512):

        fields = {'article': 'article',
                  'label': 'label'}
        super(BertData, self).__init__(fields=fields)

        self.max_nsents = max_nsents
        self.max_ntokens = max_ntokens
        self.max_len = max_len

        self.tokenizer = BertTokenizer.from_pretrained('/path/to/uncased_L-12_H-768_A-12')
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']

    def _load(self, paths):
        dataset = super(BertData, self)._load(paths)
        return dataset

    def process(self, paths):

        def truncate_articles(instance, max_nsents=self.max_nsents, max_ntokens=self.max_ntokens):
            article = [' '.join(sent.lower().split()[:max_ntokens]) for sent in instance['article']]
            return article[:max_nsents]

        def truncate_labels(instance):
            label = list(filter(lambda x: x < len(instance['article']), instance['label']))
            return label

        def bert_tokenize(instance, tokenizer, max_len, pad_value):
            article = instance['article']
            article = ' [SEP] [CLS] '.join(article)
            word_pieces = tokenizer.tokenize(article)[:(max_len - 2)]
            word_pieces = ['[CLS]'] + word_pieces + ['[SEP]']
            token_ids = tokenizer.convert_tokens_to_ids(word_pieces)
            while len(token_ids) < max_len:
                token_ids.append(pad_value)
            assert len(token_ids) == max_len
            return token_ids

        def get_seg_id(instance, max_len, sep_id):
            _segs = [-1] + [i for i, idx in enumerate(instance['article']) if idx == sep_id]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            segment_id = []
            for i, length in enumerate(segs):
                if i % 2 == 0:
                    segment_id += length * [0]
                else:
                    segment_id += length * [1]
            while len(segment_id) < max_len:
                segment_id.append(0)
            return segment_id

        def get_cls_id(instance, cls_id):
            classification_id = [i for i, idx in enumerate(instance['article']) if idx == cls_id]
            return classification_id

        def get_labels(instance):
            labels = [0] * len(instance['cls_id'])
            label_idx = list(filter(lambda x: x < len(instance['cls_id']), instance['label']))
            for idx in label_idx:
                labels[idx] = 1
            return labels

        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])

            # remove empty samples
            datasets[name].drop(lambda ins: len(ins['article']) == 0 or len(ins['label']) == 0)

            # truncate articles
            datasets[name].apply(lambda ins: truncate_articles(ins, self.max_nsents, self.max_ntokens),
                                 new_field_name='article')

            # truncate labels
            datasets[name].apply(truncate_labels, new_field_name='label')

            # tokenize and convert tokens to id
            datasets[name].apply(lambda ins: bert_tokenize(ins, self.tokenizer, self.max_len, self.pad_id),
                                 new_field_name='article')

            # get segment id
            datasets[name].apply(lambda ins: get_seg_id(ins, self.max_len, self.sep_id), new_field_name='segment_id')

            # get classification id
            datasets[name].apply(lambda ins: get_cls_id(ins, self.cls_id), new_field_name='cls_id')

            # get label
            datasets[name].apply(get_labels, new_field_name='label')

            # rename filed
            datasets[name].rename_field('article', Const.INPUTS(0))
            datasets[name].rename_field('segment_id', Const.INPUTS(1))
            datasets[name].rename_field('cls_id', Const.INPUTS(2))
            datasets[name].rename_field('lbael', Const.TARGET)

            # set input and target
            datasets[name].set_input(Const.INPUTS(0), Const.INPUTS(1), Const.INPUTS(2))
            datasets[name].set_target(Const.TARGET)

            # set paddding value
            datasets[name].set_pad_val('article', 0)

        return DataBundle(datasets=datasets)


class PreSummABSLoader(JsonLoader):
    """

    """

    def __init__(self, args):
        fields = {
            'text_ids': 'text_ids',
            'summary_ids': Const.TARGET,
            'segment_ids': 'segment_ids',
            'cls_ids': 'cls_ids',
            'label': 'label',
            'summary': 'summary'
        }
        self.max_pos = args.max_pos
        self.max_summary_len = args.max_summary_len

        super(PreSummABSLoader, self).__init__(fields=fields)

    def _load(self, paths):
        dataset = super(PreSummABSLoader, self)._load(paths)
        return dataset

    def process(self, paths):
        def truncate_input(instance):
            text_ids = instance['text_ids']
            summary_ids = instance[Const.TARGET]
            cls_ids = instance['cls_ids']
            segment_ids = instance['segment_ids']
            label = instance['label']
            tgt_txt = instance['summary']

            end_id = [text_ids[-1]]
            text_ids = text_ids[:-1][:self.max_pos - 1] + end_id
            summary_ids = summary_ids[:self.max_summary_len][:-1] + [2]
            segment_ids = segment_ids[:self.max_pos]
            max_sent_id = bisect.bisect_left(cls_ids, self.max_pos)
            label = label[:max_sent_id]
            cls_ids = cls_ids[:max_sent_id]

            return {'text_ids': text_ids, Const.TARGET: summary_ids, "summary_ids": summary_ids, 'cls_ids': cls_ids,
                    'segment_ids': segment_ids, 'label': label, 'tgt_txt': tgt_txt}

        print('Start loading datasets !!!')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])
            print(name)
            print(datasets[name][0])

            datasets[name].apply_more(lambda ins: truncate_input(ins))

            # set input and target
            datasets[name].set_input('text_ids', 'segment_ids', 'cls_ids', 'tgt_txt', 'summary_ids')
            datasets[name].set_target(Const.TARGET, 'tgt_txt')

            # set padding value
            # 如果使用其他的预训练模型要替换这里的pad value
            datasets[name].set_pad_val('text_ids', 0)
            datasets[name].set_pad_val('segment_ids', 0)
            datasets[name].set_pad_val('summary_ids', 0)
            datasets[name].set_pad_val('cls_ids', -1)
            datasets[name].set_pad_val(Const.TARGET, 0)

        print('Finished in {}'.format(timedelta(seconds=time() - start)))

        return DataBundle(datasets=datasets)


class PreSummEXTLoader(JsonLoader):

    def __init__(self, args):
        fields = {'text_ids': 'text_ids',
                  'segment_ids': 'segment_ids',
                  'cls_ids': 'cls_ids',
                  'label': Const.TARGET,
                  'summary': 'summary',
                  'text': 'text',
                  }

        self.max_pos = args.max_pos
        self.max_summary_len = args.max_summary_len

        super(PreSummEXTLoader, self).__init__(fields=fields)

    def _load(self, paths):
        dataset = super(PreSummEXTLoader, self)._load(paths)
        return dataset

    def process(self, paths):
        def get_seq_len(instance):
            return len(instance['text_ids'])

        def truncate_input(instance):
            text_ids = instance['text_ids']
            cls_ids = instance['cls_ids']
            segment_ids = instance['segment_ids']
            label = instance[Const.TARGET]
            tgt_txt = instance['summary']
            src_txt = instance['text']

            end_id = [text_ids[-1]]
            text_ids = text_ids[:-1][:self.max_pos - 1] + end_id
            segment_ids = segment_ids[:self.max_pos]
            max_sent_id = bisect.bisect_left(cls_ids, self.max_pos)
            label = label[:max_sent_id]
            cls_ids = cls_ids[:max_sent_id]
            assert len(label) == len(
                cls_ids), "label and cls_ids size not match! Label size: {} while cls_ids size: {}".format(len(label),
                                                                                                           len(cls_ids))

            return {'text_ids': text_ids, 'cls_ids': cls_ids, 'src_txt': src_txt,
                    'segment_ids': segment_ids, Const.TARGET: label, 'tgt_txt': tgt_txt}

        print('Start loading datasets !!!')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])
            print(name)
            print(datasets[name][0])

            datasets[name].apply_more(lambda ins: truncate_input(ins))
            datasets[name].apply(get_seq_len, new_field_name='seq_len')

            # set input and target
            datasets[name].set_input('text_ids', 'segment_ids', 'cls_ids')
            datasets[name].set_target(Const.TARGET, 'tgt_txt', 'src_txt')

            # set padding value
            datasets[name].set_pad_val('text_ids', 0)
            datasets[name].set_pad_val('segment_ids', 0)
            datasets[name].set_pad_val('cls_ids', -1)
            datasets[name].set_pad_val(Const.TARGET, 0)

        print('Finished in {}'.format(timedelta(seconds=time() - start)))

        return DataBundle(datasets=datasets)
