from data_util.config import Config
from data_util.data import prepare_dataInfo, PAD_TOKEN
from data_util.logging import logger
from model.loss import MyLoss
from model.model import Model
from fastNLP import SequentialSampler
from fastNLP import Trainer
from fastNLP import Tester
from torch.optim import Adagrad
from model.metric import PyRougeMetric, FastRougeMetric
import os
import time
from data_util.utils import print_config, write_eval_results
from training_ptr_gen.callback import TrainCallback
import torch
import sys
import tensorflow as tf
import argparse


def initial_dir(mode, config, model_file_path=None):
    if not os.path.exists(config.log_root):
        os.mkdir(config.log_root)

    if mode == 'train':
        _train_name = ""
        if config.pointer_gen:
            _train_name = _train_name + "_pointer_gen"
        if config.is_coverage:
            _train_name = _train_name + "_coverage"

        train_dir = os.path.join(config.log_root, 'train{}'.format(_train_name))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        return train_dir, model_dir
    else:
        if model_file_path is None:
            logger.error("error!, no model to load")
            raise Exception("empty model file path!", model_file_path)
        parent_path = os.path.dirname(model_file_path)
        train_path = os.path.dirname(parent_path)
        model_name = os.path.basename(model_file_path)
        decode_path = os.path.join(train_path, 'decode_{}_{}'.format(config.test_data_name, model_name))
        # decode_path = os.path.join(train_path, 'decode_%s' % (model_name))

        if not os.path.exists(decode_path):
            os.mkdir(decode_path)
        else:
            if os.path.exists(decode_path+"/"+"gold.txt"):
                os.remove(decode_path+"/"+"gold.txt")
            if os.path.exists(decode_path+"/"+"pred.txt"):
                os.remove(decode_path+"/"+"pred.txt")

        return decode_path


def set_up_data(mode, config):
    datainfo = prepare_dataInfo(mode=mode, test_data_path=config.decode_data_path,
                                train_data_path=config.train_data_path, vocab_size=config.vocab_size,
                                config=config)
    logger.info('-' * 10 + "set up data done!" + '-' * 10)
    return datainfo


def run_test(model_file_path, config):
    decode_path = initial_dir('test', config, model_file_path)
    config.decode_path = decode_path
    if os.path.exists(os.path.join(config.decode_path, 'result.jsonl')):
        os.remove(os.path.join(config.decode_path, 'result.jsonl'))

    datainfo = set_up_data('test', config)

    model = Model(vocab=datainfo.vocabs["train"], config=config)
    if model_file_path is not None:
        # state = torch.load(model_file_path, map_location=lambda storage, location: storage)

        state = torch.load(model_file_path, map_location=lambda storage, location: storage).state_dict()
        model.load_state_dict(state)

        '''
        model.encoder.load_state_dict(state['encoder_state_dict'])
        model.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        model.reduce_state.load_state_dict(state['reduce_state_dict'])
        '''

    tester = Tester(model=model, data=datainfo.datasets['test'], metrics=PyRougeMetric(pred='prediction',
                                                                                       art_oovs='article_oovs',
                                                                                       abstract_sentences='abstract_sentences',
                                                                                       config=config,
                                                                                       vocab=datainfo.vocabs["train"]),
                    batch_size=config.batch_size)
    eval_results = tester.test()
    write_eval_results(decode_path, eval_results)


def getting_k_model_path(path, top_k):
    _list = [name for name in os.listdir(path) if name.startswith("model_")]
    assert len(_list) >= top_k, "error: too little models to choose lowest k loss models!!"
    tmp = {}
    for f in _list:
        tmp_loss = float(f.split("_")[-1])
        tmp[f] = tmp_loss
    k_result = sorted(tmp.items(), key=lambda item: item[1])[:top_k]
    print("test models: ", k_result)
    return [os.path.join(path, _item[0]) for _item in k_result]


# python decode.py -decode_data_path CNNDM/finished_files_new1/CNNDM.test.json -train_data_path CNNDM/finished_files_new1/CNNDM.train.json -test_model ../log/CNNDM/train_1576560623/model/model_223000_1576669601 -log_root CNNDM -is_pointer_gen -is_coverage -test_data_name cnndm -visible_gpu 5
# python decode.py -decode_data_path CNNDM/finished_files_new1/CNNDM.test.json -train_data_path CNNDM/finished_files_new1/CNNDM.train.json -m ../log/CNNDM/train_pointer_gen_coverage/model/ -log_root CNNDM -is_pointer_gen -is_coverage -test_data_name cnndm -visible_gpu 5 -top_k 5
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-top_k", default=1, help="choose the k lowest loss model to test", type=int)
    parser.add_argument("-m", dest="model_file_path", required=False, default=None,
                        help="Model file for retraining (default: None).")

    parser.add_argument('-visible_gpu', default=-1, type=int, required=True)
    parser.add_argument('-dataset_path', default="/remote-home/yrchen/Datasets")
    parser.add_argument('-train_data_path',
                        default="CNNDM/finished_files_new1/CNNDM.train.json", required=True)
    # parser.add_argument('-eval_data_path',
    #                     default="CNNDM/finished_files_new1/CNNDM.val.json", required=True)
    parser.add_argument('-decode_data_path',
                        default="CNNDM/finished_files_new1/CNNDM.test.json", required=True)
    # parser.add_argument('-vocab_path', default='CNNDM/finished_files_new1/vocab.pkl')
    parser.add_argument('-root',
                        default='/remote-home/yrchen/tasks/fastnlp-relevant/summarization/my-pnt-sum/log')
    parser.add_argument('-log_root', default='CNNDM', required=True)

    parser.add_argument('-hidden_dim', default=256, type=int)
    parser.add_argument('-emb_dim', default=128, type=int)
    # parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-max_enc_steps', default=400, type=int)
    parser.add_argument('-max_dec_steps', default=100, type=int)
    parser.add_argument('-beam_size', default=4, type=int)
    parser.add_argument('-min_dec_steps', default=35, type=int)
    parser.add_argument('-vocab_size', default=50000, type=int)

    parser.add_argument('-lr', default=0.15, type=float)
    parser.add_argument('-adagrad_init_acc', default=0.1, type=float)
    parser.add_argument('-rand_unif_init_mag', default=0.02, type=float)
    parser.add_argument('-trunc_norm_init_std', default=1e-4, type=float)
    parser.add_argument('-max_grad_norm', default=2.0, type=float)

    parser.add_argument('-is_pointer_gen', dest='pointer_gen', nargs='?', const=True, default=False,
                        type=bool)
    parser.add_argument('-is_coverage', nargs='?', const=True, default=False, type=bool)
    parser.add_argument('-cov_loss_wt', default=1.0, type=float)

    parser.add_argument('-eps', default=1e-12, type=float)
    # parser.add_argument('-max_iterations', default=500000, required=True, type=int)
    parser.add_argument("-n_epochs", default=33, type=int)

    parser.add_argument('-lr_coverage', default=0.15, type=float)
    parser.add_argument('-test_data_name', required=True, type=str)
    parser.add_argument('-test_model', default='', type=str)
    args = parser.parse_args()

    args.train_data_path = os.path.join(args.dataset_path, args.train_data_path)
    # args.eval_data_path = os.path.join(args.dataset_path, args.eval_data_path)
    args.decode_data_path = os.path.join(args.dataset_path, args.decode_data_path)
    # args.vocab_path = os.path.join(args.dataset_path, args.vocab_path)

    args.log_root = os.path.join(args.root, args.log_root)

    if args.visible_gpu != -1:
        args.use_gpu = True
        torch.cuda.set_device(args.visible_gpu)
        print("using gpu: ", args.visible_gpu)
    else:
        args.use_gpu = False

    logger.info("------start mode test-------")
    if args.test_model == '':
        k_model_path_list = getting_k_model_path(args.model_file_path, args.top_k)
        for tmp_path in k_model_path_list:
            run_test(tmp_path, args)
    else:
        run_test(args.test_model, args)
