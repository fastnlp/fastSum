"""
tokenize 花费时间是很长的，预处理保存好
"""

import argparse
from os.path import exists
import json
import os

from fastNLP.core import DataSet

from dataloader import BertData


def save(path: str, dataset: DataSet, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        for instance in dataset:
            dict_instance = {key: value for key, value in instance.items()}
            f.write(json.dumps(dict_instance))
            f.write('\n')


def preprocess_model0(args: argparse.Namespace):
    # check if the data_path exists
    paths = {
        'train': 'data/' + args.label_type + '/train.label.jsonl',
    }
    for name in paths:
        assert exists(paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    fields = {
        'text': 'text',
        'label': 'label',
        # 'summary': 'summary'  # train 里面没有 summary 字样
        }

    # load summarization datasets
    datasets = BertData(fields).process(paths)
    print('Information of dataset is:')
    print(datasets)

    train_set = datasets.datasets['train']

    save(f'{args.save_path}/bert.train.jsonl', train_set)


def preprocess_model1(args: argparse.Namespace):
    # check if the data_path exists
    paths = {
        'val': 'data/' + args.label_type + '/val.label.jsonl',
        'test': 'data/' + args.label_type + '/test.label.jsonl'
    }
    for name in paths:
        assert exists(paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    fields = {
        'text': 'text',
        'label': 'label',
        'summary': 'summary'  # train 里面没有 summary 字样
        }

    # load summarization datasets
    datasets = BertData(fields).process(paths)
    print('Information of dataset is:')
    print(datasets)

    valid_set = datasets.datasets['val']
    test_set = datasets.datasets['test']

    save(f'{args.save_path}/bert.val.jsonl', valid_set)
    save(f'{args.save_path}/bert.test.jsonl', test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of BertSum(liu et al. 2019)'
    )
    parser.add_argument('--mode', required=True,
                        help='preprocessing, training or testing of BertSum', type=str)

    parser.add_argument('--label_type', default='greedy',
                        help='greedy/limit', type=str)

    parser.add_argument('--save_path', required=True,
                        help='root of the model', type=str)

    args = parser.parse_args()
    assert args.mode == 'preprocess'

    preprocess_model0(args)
    preprocess_model1(args)
