from data_util.config import Config
from data_util.data import prepare_dataInfo, PAD_TOKEN
from data_util.logging import logger
from model.loss import MyLoss
from model.model import Model
from fastNLP import BucketSampler
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
        decode_path = os.path.join(train_path, 'decode_%s' % (model_name))

        if not os.path.exists(decode_path):
            os.mkdir(decode_path)

        return decode_path


def set_up_data(mode, config):
    datainfo = prepare_dataInfo(mode=mode, train_data_path=config.train_data_path, dev_data_path=config.eval_data_path,
                                vocab_size=config.vocab_size, config=config)
    logger.info('-' * 10 + "set up data done!" + '-' * 10)
    return datainfo


def run_train(config):
    train_dir, model_dir = initial_dir('train', config)
    config.train_path = train_dir
    config.model_path = model_dir
    print_config(config, train_dir)
    datainfo = set_up_data('train', config)
    train_sampler = BucketSampler(batch_size=config.batch_size, seq_len_field_name='enc_len')
    criterion = MyLoss(config=config, padding_idx=datainfo.vocabs["train"].to_index(PAD_TOKEN))

    model = Model(vocab=datainfo.vocabs["train"], config=config)
    params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
             list(model.reduce_state.parameters())
    initial_lr = config.lr_coverage if config.is_coverage else config.lr
    optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

    train_loader = datainfo.datasets["train"]
    valid_loader = datainfo.datasets["dev"]
    summary_writer = tf.compat.v1.summary.FileWriter(train_dir)
    trainer = Trainer(model=model, train_data=train_loader, optimizer=optimizer, loss=criterion,
                      batch_size=config.batch_size, check_code_level=-1,
                      n_epochs=config.n_epochs, print_every=100, dev_data=valid_loader,
                      metrics=FastRougeMetric(pred='prediction', art_oovs='article_oovs',
                                              abstract_sentences='abstract_sentences', config=config,
                                              vocab=datainfo.vocabs["train"]),
                      metric_key="rouge-l-f", validate_every=-1, save_path=model_dir,
                      callbacks=[TrainCallback(config, summary_writer, patience=10)], use_tqdm=False,
                      device=config.visible_gpu)

    logger.info("-" * 5 + "start training" + "-" * 5)

    traininfo = trainer.train(load_best_model=True)
    logger.info('   | end of Train | time: {:5.2f}s | '.format(traininfo["seconds"]))
    logger.info('[INFO] best eval model in epoch %d and iter %d', traininfo["best_epoch"], traininfo["best_step"])
    logger.info(traininfo["best_eval"])

    bestmodel_save_path = os.path.join(config.model_path,
                                       'bestmodel.pkl')  # this is where checkpoints of best models are saved
    state = {
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'reduce_state_dict': model.reduce_state.state_dict()
    }
    torch.save(state, bestmodel_save_path)
    # 不是作为形参传入到Trainer里面的么，怎么里面的model变化会影响到外面的？
    logger.info('[INFO] Saving eval best model to %s', bestmodel_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m", dest="model_file_path", required=False, default=None,
                        help="Model file for retraining (default: None).")
    parser.add_argument('-visible_gpu', default=-1, type=int, required=True)
    parser.add_argument('-dataset_path', default="/remote-home/yrchen/Datasets")
    parser.add_argument('-train_data_path',
                        default="CNNDM/finished_files_new1/CNNDM.train.json", required=True)
    parser.add_argument('-eval_data_path',
                        default="CNNDM/finished_files_new1/CNNDM.val.json", required=True)
    # parser.add_argument('-decode_data_path',
    #                     default="CNNDM/finished_files_new1/CNNDM.test.json", required=True)
    # parser.add_argument('-vocab_path', default='CNNDM/finished_files_new1/vocab.pkl')
    parser.add_argument('-root',
                        default='/remote-home/yrchen/tasks/fastnlp-relevant/summarization/my-pnt-sum/log')
    parser.add_argument('-log_root', default='CNNDM', required=True)

    parser.add_argument('-hidden_dim', default=256, type=int)
    parser.add_argument('-emb_dim', default=128, type=int)
    # parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-batch_size', default=16, type=int)
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
    parser.add_argument("-n_epochs", default=33, type=int, required=True)

    parser.add_argument('-lr_coverage', default=0.15, type=float)
    args = parser.parse_args()

    args.train_data_path = os.path.join(args.dataset_path, args.train_data_path)
    args.eval_data_path = os.path.join(args.dataset_path, args.eval_data_path)
    # args.decode_data_path = os.path.join(args.dataset_path, args.decode_data_path)
    # args.vocab_path = os.path.join(args.dataset_path, args.vocab_path)

    args.log_root = os.path.join(args.root, args.log_root)

    if args.visible_gpu != -1:
        args.use_gpu = True
        torch.cuda.set_device(args.visible_gpu)
        print("using gpu: ", args.visible_gpu)
    else:
        args.use_gpu = False

    logger.info("------start mode train------")
    run_train(args)
