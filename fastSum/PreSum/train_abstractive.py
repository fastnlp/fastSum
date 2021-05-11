#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
from pytorch_transformers import BertTokenizer

import distributed
# from models import data_loader, model_builder
# from models.data_loader import load_dataset
# from models.loss import abs_loss
from models.model_builder import AbsSummarizer
# from models.predictor import build_predictor
# from models.trainer import build_trainer
from others.logging import logger, init_logger
from others.utils import get_data_path, configure_training
import torch.distributed as dist

import json
from os.path import join, exists
from dataloader import PreSummABSLoader

from models.optimizers import build_optim
from metrics import MyNLLLoss, ABSLossMetric, FastRougeMetricABS, PyRougeMetricABS
from callback import MyCallback, SaveModelCallback, EarlyStopCallback, FitlogCallback
from fastNLP.core.trainer import Trainer
from trainer import MyTrainer
from fastNLP.core.tester import Tester
from fastNLP import DistTrainer, get_local_rank

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#
# def baseline(args, cal_lead=False, cal_oracle=False):
#     test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
#                                        args.batch_size, 'cpu',
#                                        shuffle=False, is_test=True)
#
#     trainer = build_trainer(args, '-1', None, None, None)
#     #
#     if (cal_lead):
#         trainer.test(test_iter, 0, cal_lead=True)
#     elif (cal_oracle):
#         trainer.test(test_iter, 0, cal_oracle=True)


def train_abs(args):
    init_logger(args.log_file)
    logger.info(str(args))

    # check if the data_path and save_path exists
    data_paths = get_data_path(args.mode, args.label_type)
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    # device = "cpu" if args.visible_gpus == '-1' else "cuda"
    # logger.info('Device ID %d' % device_id)
    # logger.info('Device %s' % device)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    #
    # if device_id >= 0:
    #     torch.cuda.set_device(device_id)
    #     torch.cuda.manual_seed(args.seed)
    if args.local_rank not in [-1, 0]:
        dist.barrier()

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    if (args.load_from_extractive != ''):
        logger.info('Loading bert from extractive model %s' % args.load_from_extractive)
        bert_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True

    # load summarization datasets
    datasets = PreSummABSLoader(args).process(data_paths)
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    model = AbsSummarizer(args, checkpoint, bert_from_extractive, symbols, tokenizer)

    logger.info(model)

    # if (args.sep_optim):
    #     optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
    #     optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
    #     optim = [optim_bert, optim_dec]
    # else:
    #     optim = [model_builder.build_optim(args, model, checkpoint)]

    optim = build_optim(args, model, checkpoint)

    if args.local_rank not in [-1, 0]:
        dist.barrier()

    # train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
    #                       label_smoothing=args.label_smoothing)
    train_loss = MyNLLLoss(model.generator, model.vocab_size, pred='pred', target='target',
                           label_smoothing=args.label_smoothing,
                           pad_id=symbols['PAD'])
    callbacks = [MyCallback(args, optims=optim),
                 SaveModelCallback(args.save_path, optims=optim, args=args, save_on_exception=True),
                 EarlyStopCallback(), FitlogCallback(data=valid_set)]
    val_metric = [ABSLossMetric(model.generator, model.vocab_size, pred='pred', target='target',
                                label_smoothing=args.label_smoothing,
                                pad_id=symbols['PAD'])]

    trainer = MyTrainer(train_data=train_set, model=model, optimizer=None,
                        loss=train_loss, batch_size=args.batch_size,  # sampler=sampler,
                        update_every=args.accum_count, n_epochs=args.n_epochs,
                        print_every=100, dev_data=valid_set, metrics=val_metric,
                        metric_key='-loss', validate_every=args.valid_steps * args.accum_count,
                        save_path=args.save_path, device=devices, callbacks=callbacks, config=args)

    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()

    # trainer = build_trainer(args, device_id, model, optim, train_loss)
    #
    # trainer.train(train_iter_fct, args.train_steps)


def test_abs(args, pt):
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    # load summarization datasets
    data_paths = get_data_path(args.mode, args.label_type)
    datasets = PreSummABSLoader(args).process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    test_set = datasets.datasets['test']

    # only need 1 gpu for testing
    device = int(args.visible_gpus)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    model = AbsSummarizer(args, checkpoint=checkpoint, symbols=symbols, tokenizer=tokenizer)
    model.eval()

    test_metric = PyRougeMetricABS(pred='predictions', tgt_txt='tgt_txt', config=args, vocab=tokenizer, logger=logger)
    tester = Tester(data=test_set, model=model, metrics=[test_metric],
                    batch_size=args.test_batch_size, device=device)
    tester.test()

    # predictor = build_predictor(args, tokenizer, symbols, model, logger)
    # predictor.translate(test_iter, step)
