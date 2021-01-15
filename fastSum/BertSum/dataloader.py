from time import time
from datetime import timedelta
from typing import Dict, List

from fastNLP.io import JsonLoader
from fastNLP.modules.tokenizer import BertTokenizer
from fastNLP.io.data_bundle import DataBundle
from fastNLP.core.const import Const
from fastNLP.core.instance import Instance


class BertData(JsonLoader):
    
    def __init__(self, fields: Dict[str, str], max_nsents: int = 60, max_ntokens: int = 100, max_len: int = 512):
        """

        :param fields:
        :param max_nsents: 每个 article 的最大句子数量限制
        :param max_ntokens: 每个句子的最大单词数限制
        :param max_len: 每个 article 的最大单词数
        """
        
        # fields = {
        #     'text': 'text',
        #     'label': 'label',
        #     'summary': 'summary'  # train 里面没有 summary 字样
        #     }
        super(BertData, self).__init__(fields=fields, dropna=True)

        self.max_nsents = max_nsents
        self.max_ntokens = max_ntokens
        self.max_len = max_len

        self.tokenizer = BertTokenizer.from_pretrained('data/uncased_L-12_H-768_A-12')
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        assert self.pad_id == 0

    # _load 实际上只用了 JsonLoader 的实现，未作任何修改
    # def _load(self, path):
    #     dataset = super(BertData, self)._load(path)
    #     return dataset

    def process(self, paths: Dict[str, str]) -> DataBundle:
        """

        :param paths: Dict[name, real_path]；real_path 为真正的路径；name 索引或者简略名
        :return:
        """
        
        def truncate_articles(instance: Instance, max_nsents: int = self.max_nsents, max_ntokens: int = self.max_ntokens) -> List[str]:
            """

            :param instance: 某条数据
            :param max_nsents: 详见 __init__ 中 max_nsents
            :param max_ntokens: 详见 __init__ 中 max_ntokens
            :return: 返回截断后的 article；格式为 List[sentence]，sentence 是一个句子（str 格式）
            """
            article = [' '.join(sent.lower().split()[:max_ntokens]) for sent in instance['article']]
            return article[:max_nsents]

        def truncate_labels(instance: Instance):
            """
            超出 max_nsents 的摘要的指示 label 都删掉
            :param instance:
            :return:
            """
            label = list(filter(lambda x: x < len(instance['article']), instance['label']))  # label 是 indices_chose，告诉用户选了那些句子作为 summary
            return label
        
        def bert_tokenize(instance: Instance, tokenizer: BertTokenizer, max_len: int, pad_value: int) -> List[int]:
            """
            执行 WordPiece 操作
            :param instance:
            :param tokenizer:
            :param max_len: 详见 __init__ 中 max_len
            :param pad_value: [PAD]
            :return:
            """
            article = instance['article']
            article = ' [SEP] [CLS] '.join(article)
            # tokenizer 会分割输入句子，类似于 pytorch-pretrained-bert 那里的 BertTokenizer 那样执行 WordPiece 操作，把一个 article 转为 List[str]
            # 利用 max_len 截断
            word_pieces = tokenizer.tokenize(article)[:(max_len - 2)]
            word_pieces = ['[CLS]'] + word_pieces + ['[SEP]']
            token_ids = tokenizer.convert_tokens_to_ids(word_pieces)
            # 不足长度的填充
            while len(token_ids) < max_len:
                token_ids.append(pad_value)
            # assert len(token_ids) == max_len
            return token_ids

        def get_seg_id(instance: Instance, max_len: int, sep_id: int, pad_value: int) -> List[int]:
            """
            Interval Segment Embeddings 生成

            :param instance:
            :param max_len: 详见 __init__ 中 max_len
            :param sep_id: [SEP]
            :param pad_value:
            :return:
            """
            _segs = [-1] + [i for i, idx in enumerate(instance['article']) if idx == sep_id]  # [CLS, 第 0 句, SEP, CLS, 第 1 句, SEP, ..., 第 n-1 句, SEP]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]  # 两两相减，求间隔长度；List[间隔长度]
            segment_id = []
            # [(CLS, 第 0 句, SEP), (CLS, 第 1 句, SEP), ..., 第 n-1 句, SEP]
            # [(0, 0.......0, 0), (1, 1....1, 1), ..., ...]
            for i, length in enumerate(segs):
                if i % 2 == 0:
                    segment_id += length * [0]
                else:
                    segment_id += length * [1]
            # 不足长度的填充
            while len(segment_id) < max_len:
                segment_id.append(pad_value)
            return segment_id
        
        def get_cls_id(instance: Instance, cls_id: int) -> List[int]:
            """
            找到所有 [CLS] 的在执行了 WordPiece 后的 article 位置
            :param instance:
            :param cls_id: [CLS]
            :return:
            """
            classification_id = [i for i, idx in enumerate(instance['article']) if idx == cls_id]
            return classification_id
        
        def get_labels(instance: Instance) -> List[int]:
            """
            根据 label （即 indices_chose）对每个句子生成 0 1 标签
            :param instance:
            :return:
            """
            labels = [0] * len(instance['cls_id'])
            label_idx = list(filter(lambda x: x < len(instance['cls_id']), instance['label']))  # 我觉得这里 filter 和 truncate_labels 操作重复了
            for idx in label_idx:
                labels[idx] = 1
            return labels

        datasets = {}
        for name in paths:
            # _load 中会调用 _read_json
            # _read_json 返回的格式如下："yield line_idx, _res" 详见 io/file_reader.py；line_idx 表示这是第几行数据的 load 结果；_res 格式为 Dict[fields' key, 某条数据中的对应该 key 的 value]
            # _load 处理数据后，首先针对每条数据把 fields' key 处理为 fields' value 得到了 Dict[fields' value, 某条数据中的对应该 key 的 value]，然后把该条数据包裹为一个 Instance（core/instance.py），
            # 所有的 Instance 使用 append 方法，丢到一个统一的数据容器里面 DataSet（core/dataset.py），并最终返回这个 DataSet
            datasets[name] = self._load(paths[name])

            datasets[name].copy_field('text', 'article')
            
            # remove empty samples（丢弃空数据）
            datasets[name].drop(lambda ins: len(ins['article']) == 0 or len(ins['label']) == 0)
            
            # truncate articles（截断文章）
            # new_field_name：将 func 返回的内容放入到 `new_field_name` 这个 field 中，如果名称与已有的 field 相同，则覆盖之前的 field。
            datasets[name].apply(lambda ins: truncate_articles(ins, self.max_nsents, self.max_ntokens), new_field_name='article')
            
            # truncate labels（与上面类似，对照截断 label）
            # new_field_name 见上面
            datasets[name].apply(truncate_labels, new_field_name='label')
            
            # tokenize and convert tokens to id（执行 WordPiece 操作）
            datasets[name].apply(lambda ins: bert_tokenize(ins, self.tokenizer, self.max_len, pad_value=self.pad_id), new_field_name='article')
            
            # get segment id（Interval Segment Embeddings 生成）
            datasets[name].apply(lambda ins: get_seg_id(ins, self.max_len, self.sep_id, pad_value=0), new_field_name='segment_id')
            
            # get classification id（提取 [CLS] 的位置）
            datasets[name].apply(lambda ins: get_cls_id(ins, self.cls_id), new_field_name='cls_id')

            # get label（生成 0 1 标签）
            datasets[name].apply(get_labels, new_field_name='label')

            # set input and target
            datasets[name].set_input('article', 'segment_id', 'cls_id')
            datasets[name].set_target('label')
            
            # set padding value
            datasets[name].set_pad_val('article', self.pad_id)

        return DataBundle(datasets=datasets)


class BertSumLoader(JsonLoader):
    
    def __init__(self):
        fields = {'article': 'article',
                  'segment_id': 'segment_id',
                  'cls_id': 'cls_id',
                  'label': Const.TARGET
                  }
        super(BertSumLoader, self).__init__(fields=fields)

    # _load 实际上只用了 JsonLoader 的实现，未作任何修改
    # def _load(self, path):
    #     dataset = super(BertSumLoader, self)._load(path)
    #     return dataset

    def process(self, paths):
        """

        :param paths: Dict[name, real_path]；real_path 为真正的路径；name 索引或者简略名
        :return:
        """
        
        def get_seq_len(instance: Instance):
            return len(instance['article'])

        print('Start loading datasets !!!')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            # _load 中会调用 _read_json
            # _read_json 返回的格式如下："yield line_idx, _res" 详见 io/file_reader.py；line_idx 表示这是第几行数据的 load 结果；_res 格式为 Dict[fields' key, 某条数据中的对应该 key 的 value]
            # _load 处理数据后，首先针对每条数据把 fields' key 处理为 fields' value 得到了 Dict[fields' value, 某条数据中的对应该 key 的 value]，然后把该条数据包裹为一个 Instance（core/instance.py），
            # 所有的 Instance 使用 append 方法，丢到一个统一的数据容器里面 DataSet（core/dataset.py），并最终返回这个 DataSet
            datasets[name] = self._load(paths[name])
            
            datasets[name].apply(get_seq_len, new_field_name='seq_len')

            # set input and target
            datasets[name].set_input('article', 'segment_id', 'cls_id')
            datasets[name].set_target(Const.TARGET)
        
            # set padding value
            datasets[name].set_pad_val('article', 0)  # 这里与 BertData 的 padding 保持一致
            datasets[name].set_pad_val('segment_id', 0)  # 这里与 BertData 的 padding 保持一致
            datasets[name].set_pad_val('cls_id', -1)
            datasets[name].set_pad_val(Const.TARGET, 0)

        print('Finished in {}'.format(timedelta(seconds=time()-start)))

        return DataBundle(datasets=datasets)
