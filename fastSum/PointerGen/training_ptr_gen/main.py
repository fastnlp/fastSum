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

config = Config()


def initial_dir(mode, model_file_path=None):
    if mode == 'train':
        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
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


def set_up_data(mode):
    datainfo = prepare_dataInfo(mode, config.train_data_path, config.eval_data_path, config.decode_data_path,
                                config.vocab_path, config.vocab_size, config)
    logger.info('-' * 10 + "set up data done!" + '-' * 10)
    return datainfo


def run_train():
    train_dir, model_dir = initial_dir('train')
    config.train_path = train_dir
    config.model_path = model_dir
    print_config(config, train_dir)
    datainfo = set_up_data('train')
    train_sampler = BucketSampler(batch_size=config.batch_size, seq_len_field_name='enc_len')
    criterion = MyLoss(config=config, padding_idx=datainfo.vocabs["train"].to_index(PAD_TOKEN))

    model = Model(vocab=datainfo.vocabs["train"])
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
                      callbacks=[TrainCallback(config, summary_writer, patience=5)], use_tqdm=False)

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


def run_test(model_file_path):
    decode_path = initial_dir('test', model_file_path)
    datainfo = set_up_data('test')
    model = Model(vocab=datainfo.vocabs["train"])
    tester = Tester(datainfo.datasets['test'], model=model, metrics=PyRougeMetric(pred='prediction',
                                                                                  art_oovs='article_oovs',
                                                                                  abstract_sentences='abstract_sentences',
                                                                                  config=config,
                                                                                  vocab=datainfo.vocabs["train"]), batch_size=1)
    eval_results = tester.test()
    write_eval_results(decode_path, eval_results)


if __name__ == '__main__':
    torch.cuda.set_device(4)
    mode = sys.argv[1]
    if mode == 'train':
        logger.info("------start mode train------")
        run_train()
    elif mode == 'test':
        logger.info("------start mode test-------")
        model_filename = sys.argv[2]
        run_test(model_filename)
    else:
        logger.error("error: none of the mode is in train or test!")
        raise Exception("wrong mode! neither train nor test!", mode)
