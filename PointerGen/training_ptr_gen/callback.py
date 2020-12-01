#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import time
import numpy as np

import torch
from fastNLP.core.callback import Callback, EarlyStopError

from data_util.logging import logger
from data_util.utils import calc_running_avg_loss


class TrainCallback(Callback):
    def __init__(self, config, summary_writer, patience=3, quit_all=True):
        super().__init__()
        self.config = config
        self.patience = patience
        self.wait = 0
        self.running_avg_loss = 0
        self.summary_writer = summary_writer

        if type(quit_all) != bool:
            raise ValueError("In KeyBoardInterrupt, quit_all arguemnt must be a bool.")
        self.quit_all = quit_all

    def on_epoch_begin(self):
        self.epoch_start_time = time.time()

    def on_step_end(self):
        if self.step % 100 == 0:
            self.summary_writer.flush()

        if self.step % 1000 == 0:
            state = {
                'iter': self.step,
                'encoder_state_dict': self.model.encoder.state_dict(),
                'decoder_state_dict': self.model.decoder.state_dict(),
                'reduce_state_dict': self.model.reduce_state.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_loss': self.running_avg_loss
            }
            # model_save_path = os.path.join(self.config.model_path,
            #                                'model_%d_%d' % (self.step, int(time.time())))
            # model_save_path = os.path.join(self.config.model_path,
            #                                'model_%d_%d_loss_%f' % (self.step, int(time.time()), self.running_avg_loss))
            model_save_path = os.path.join(self.config.model_path,
                                           'model_%d_loss_%f' % (self.step, self.running_avg_loss))
            #torch.save(state, model_save_path)
            #self.model.cpu()
            torch.save(self.model, model_save_path)
            #if self.config.use_gpu:
            #    self.model.cuda()

    def on_backward_begin(self, loss):
        """
        :param loss: []
        :return:
        """
        print("|epoch: %d  step: %d  loss: %.4f|" % (self.epoch, self.step, loss.item()))
        if not np.isfinite(loss.item()):
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss.item())
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
                    logger.info(param.grad.data.sum())
            raise Exception("train Loss is not finite. Stopping.")

        self.running_avg_loss = calc_running_avg_loss(loss.item(), self.running_avg_loss, self.summary_writer, self.step)

    def on_backward_end(self):
        torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.reduce_state.parameters(), self.config.max_grad_norm)

    def on_epoch_end(self):
        logger.info(
            '   | end of epoch {:3d} | time: {:5.2f}s | '.format(self.epoch, (time.time() - self.epoch_start_time)))

    def on_valid_begin(self):
        self.valid_start_time = time.time()

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        logger.info(
            '   | end of valid {:3d} | time: {:5.2f}s | '.format(self.epoch, (time.time() - self.valid_start_time)))

        # early stop
        if not is_better_eval:
            if self.wait == self.patience:

                state = {
                    'iter': self.step,
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'decoder_state_dict': self.model.decoder.state_dict(),
                    'reduce_state_dict': self.model.reduce_state.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'current_loss': self.running_avg_loss
                }

                model_save_path = os.path.join(self.config.model_path,
                                               'earlystop_step_%d.pkl' % self.step)

                # torch.save(state, model_save_path)

                #self.model.cpu()
                torch.save(self.model, model_save_path)
                #if self.config.use_gpu:
                #    self.model.cuda()

                logger.info('[INFO] Saving early stop model to %s', model_save_path)
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
            state = {
                'iter': self.step,
                'encoder_state_dict': self.model.encoder.state_dict(),
                'decoder_state_dict': self.model.decoder.state_dict(),
                'reduce_state_dict': self.model.reduce_state.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_loss': self.running_avg_loss
            }

            model_save_path = os.path.join(self.config.model_path,
                                           'earlystop_step_%d.pkl' % self.step)

            # torch.save(state, model_save_path)

            #self.model.cpu()
            torch.save(self.model, model_save_path)
            #if self.config.use_gpu:
            #    self.model.cuda()

            logger.info('[INFO] Saving early stop model to %s', model_save_path)

            if self.quit_all is True:
                sys.exit(0)  # 直接退出程序
            else:
                pass
        else:
            raise exception  # 抛出陌生Error
