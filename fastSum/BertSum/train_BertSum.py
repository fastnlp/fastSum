import argparse
import os
import json
from os.path import join, exists

import torch
from torch.optim import Adam
from fastNLP.core.trainer import Trainer
from fastNLP.core.tester import Tester

from utils import get_data_path, get_rouge_path
from dataloader import BertSumLoader
from model import BertSum
from metrics import MyBCELoss, LossMetric, RougeMetric
from callback import MyCallback, SaveModelCallback


def configure_training(args: argparse.Namespace):
    devices = [int(gpu) for gpu in range(torch.cuda.device_count())]
    params = {
        'label_type': args.label_type,
        'batch_size': args.batch_size,
        'accum_count': args.accum_count,
        'max_lr': args.max_lr,
        'warmup_steps': args.warmup_steps,
        'n_epochs': args.n_epochs,
        'valid_steps': args.valid_steps
    }
    return devices, params


def train_model(args: argparse.Namespace):
    # check if the data_path and save_path exists
    data_paths = get_data_path(args.mode, args.label_type)
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)
    
    # load summarization datasets
    datasets = BertSumLoader().process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    train_set = datasets.datasets['train']
    valid_set = datasets.datasets['val']
    
    # configure training
    devices, train_params = configure_training(args)
    with open(join(args.save_path, 'params.json'), 'w') as f:
        json.dump(train_params, f, indent=4)
    print('Devices is:')
    print(devices)

    # configure model
    model = BertSum()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0)
    callbacks = [MyCallback(args), SaveModelCallback(args.save_path)]
    criterion = MyBCELoss()
    val_metric = [LossMetric()]
    # sampler = BucketSampler(num_buckets=32, batch_size=args.batch_size)
    trainer = Trainer(train_data=train_set, model=model, optimizer=optimizer,
                      loss=criterion, batch_size=args.batch_size,  # sampler=sampler,
                      update_every=args.accum_count, n_epochs=args.n_epochs, 
                      print_every=100, dev_data=valid_set, metrics=val_metric, 
                      metric_key='-loss', validate_every=args.valid_steps, 
                      save_path=args.save_path, device=devices, callbacks=callbacks)
    
    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()


def test_model(args: argparse.Namespace):

    models = os.listdir(args.save_path)  # 请确保 path 下面有 *.pt 文件
    
    # load dataset
    data_paths = get_data_path(args.mode, args.label_type)
    datasets = BertSumLoader().process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    test_set = datasets.datasets['test']
    
    # only need 1 gpu for testing
    device = 0
    
    args.batch_size = 1

    for cur_model in models:
        
        print('Current model is {}'.format(cur_model))

        # load model
        model = torch.load(join(args.save_path, cur_model))
    
        # configure testing
        original_path, dec_path, ref_path = get_rouge_path(args.label_type)
        test_metric = RougeMetric(data_path=original_path, dec_path=dec_path, 
                                  ref_path=ref_path, n_total=len(test_set))
        tester = Tester(data=test_set, model=model, metrics=[test_metric], 
                        batch_size=args.batch_size, device=device)
        tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of BertSum(liu et al. 2019)'
    )
    parser.add_argument('--mode', required=True,
                        help='training or testing of BertSum', type=str)

    # CNN/Dailymail 原数据集仅有生成式摘要,所以需要用户生成自己的(人工)抽取式摘要。这个生成方法就是 label_type
    parser.add_argument('--label_type', default='greedy', 
                        help='greedy/limit', type=str)
    parser.add_argument('--save_path', required=True,
                        help='root of the model', type=str)

    # CUDA_VISIBLE_DEVICES=4,5
    # 来指定 cuda device
    
    parser.add_argument('--batch_size', default=18,
                        help='the training batch size', type=int)
    parser.add_argument('--accum_count', default=2,
                        help='number of updates steps to accumulate before performing a backward/update pass.', type=int)
    parser.add_argument('--max_lr', default=2e-5,
                        help='max learning rate for warm up', type=float)
    parser.add_argument('--warmup_steps', default=10000,
                        help='warm up steps for training', type=int)
    parser.add_argument('--n_epochs', default=10,
                        help='total number of training epochs', type=int)
    parser.add_argument('--valid_steps', default=1000,
                        help='number of update steps for checkpoint and validation', type=int)

    args = parser.parse_args()
    
    if args.mode == 'train':
        print('Training process of BertSum !!!')
        train_model(args)
    else:
        print('Testing process of BertSum !!!')
        test_model(args)
