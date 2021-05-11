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

import distributed
# from models import data_loader, model_builder
# from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
# from models.trainer_ext import build_trainer
from others.logging import logger, init_logger
from others.utils import get_data_path, configure_training

import json
from os.path import join, exists

from models.optimizers import build_optim
from metrics import MyBCELoss, EXTLossMetric, PyRougeMetricEXT, FastRougeMetricEXT
from callback import MyCallback, SaveModelCallback, EarlyStopCallback, FitlogCallback
from fastNLP.core.trainer import Trainer
from trainer import MyTrainer
from fastNLP.core.tester import Tester
from dataloader import PreSummEXTLoader

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


#
# def test_ext(args, device_id, pt, step):
#     device = "cpu" if args.visible_gpus == '-1' else "cuda"
#     if (pt != ''):
#         test_from = pt
#     else:
#         test_from = args.test_from
#     logger.info('Loading checkpoint from %s' % test_from)
#     checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
#     opt = vars(checkpoint['opt'])
#     for k in opt.keys():
#         if (k in model_flags):
#             setattr(args, k, opt[k])
#     print(args)
#
#     model = ExtSummarizer(args, device, checkpoint)
#     model.eval()
#
#     test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
#                                        args.test_batch_size, device,
#                                        shuffle=False, is_test=True)
#     trainer = build_trainer(args, device_id, model, None)
#     trainer.test(test_iter, step)


def train_ext(args):
    init_logger(args.log_file)
    logger.info(str(args))

    # check if the data_path and save_path exists
    data_paths = get_data_path(args.mode, args.label_type)
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

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

    # load summarization datasets
    datasets = PreSummEXTLoader(args).process(data_paths)
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

    model = ExtSummarizer(args, checkpoint)
    optim = build_optim(args, model, checkpoint)

    train_loss = MyBCELoss()
    callbacks = [MyCallback(args, optims=optim),
                 SaveModelCallback(args.save_path, optims=optim, args=args, save_on_exception=True),
                 EarlyStopCallback(), FitlogCallback(data=valid_set)]
    val_metric = [EXTLossMetric()]

    logger.info(model)

    trainer = MyTrainer(train_data=train_set, model=model, optimizer=None,
                        loss=train_loss, batch_size=args.batch_size,  # sampler=sampler,
                        update_every=args.accum_count, n_epochs=args.n_epochs,
                        print_every=100, dev_data=valid_set, metrics=val_metric,
                        metric_key='-loss', validate_every=args.valid_steps * args.accum_count,
                        save_path=args.save_path, device=devices, callbacks=callbacks, config=args)

    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()

    # trainer = build_trainer(args, device_id, model, optim)
    # trainer.train(train_iter_fct, args.train_steps)


def test_ext(args, pt):
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    init_logger(args.log_file)
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    # load summarization datasets
    data_paths = get_data_path(args.mode, args.label_type)
    datasets = PreSummEXTLoader(args).process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    test_set = datasets.datasets['test']

    # only need 1 gpu for testing
    device = int(args.visible_gpus)

    model = ExtSummarizer(args, checkpoint)
    model.eval()

    test_metric = PyRougeMetricEXT(n_ext=3, ngram_block=3, pred='pred', src_txt='src_txt', tgt_txt='tgt_txt',
                                   mask='mask', logger=logger, config=args)
    tester = Tester(data=test_set, model=model, metrics=[test_metric],
                    batch_size=args.test_batch_size, device=device)
    tester.test()

    # trainer = build_trainer(args, device_id, model, None)
    # trainer.test(test_iter, step)
