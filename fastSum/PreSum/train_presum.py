#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
# from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
# from train_extractive import train_ext, validate_ext, test_ext

from train_abstractive import train_abs, test_abs
from train_extractive import train_ext, test_ext
import torch
import torch.distributed as dist
import numpy as np
import random

import fitlog

fitlog.commit(__file__)
fitlog.set_log_dir("fitlogs/")

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'val', 'test'])
    # parser.add_argument("-bert_data_path", default='data/CNNDM_bert')
    # parser.add_argument("-model_path", default='models')
    # parser.add_argument("-result_path", default='results/cnndm')
    parser.add_argument("-temp_dir", default='temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    # parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=768, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    # parser.add_argument("-report_every", default=1, type=int)
    # parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    # parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    # parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    # parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    # parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-label_type", type=str, required=True)
    parser.add_argument("-save_path", type=str, default="checkpoints", help="path to saving the params")
    parser.add_argument("-valid_steps", type=int, help="how much steps to perform validation")
    parser.add_argument("-n_epochs", type=int)
    parser.add_argument("-decode_path", type=str, default="results",
                        help="the path to save gold summary and generated summary, this is only needed when mode is test")
    parser.add_argument("-max_summary_len", type=int, required=True, help="max summary length when loading dataset")
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--init_method', type=str, default='env://')

    args = parser.parse_args()
    # args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    # args.world_size = len(args.gpu_ranks)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    set_random_seeds(args.seed)

    # if args.local_rank is not None:
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group("nccl", init_method=args.init_method)
    #     args.dist = True
    # else:
    #     args.dist = False

    init_logger(args.log_file)
    # device = "cpu" if args.visible_gpus == '-1' else "cuda"
    # device_id = 0 if device == "cuda" else -1
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args)
        # elif (args.mode == 'val'):
        #     validate_abs(args, device_id)
        # elif (args.mode == 'lead'):
        #     baseline(args, cal_lead=True)
        # elif (args.mode == 'oracle'):
        #     baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            if not os.path.exists(args.decode_path):
                os.mkdir(args.decode_path)
            test_abs(args, cp)
        # elif (args.mode == 'test_text'):
        #     cp = args.test_from
        #     try:
        #         step = int(cp.split('.')[-2].split('_')[-1])
        #     except:
        #         step = 0
        #         test_text_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args)
        # elif (args.mode == 'val'):
        #     validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            if not os.path.exists(args.decode_path):
                os.mkdir(args.decode_path)
            test_ext(args, cp)
        # elif (args.mode == 'test_text'):
        #     cp = args.test_from
        #     try:
        #         step = int(cp.split('.')[-2].split('_')[-1])
        #     except:
        #         step = 0
        #         test_text_abs(args, device_id, cp, step)
