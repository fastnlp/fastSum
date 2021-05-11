<<<<<<< HEAD
import numpy as np
import json
from os.path import join
import torch
import logging
import tempfile
import subprocess as sp
from datetime import timedelta
from time import time

from pyrouge import Rouge155
from pyrouge.utils import log

from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase

import torch.nn.functional as F
import torchsnooper

_ROUGE_PATH = '/path/to/RELEASE-1.5.5'
ROOT = "root/path"


class LabelSmoothingLoss(torch.nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    # @torchsnooper.snoop()
    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # 我的修改：在原始代码基础上增加.to(torch.device(target.get_device())),注意这里要迁移到target在的特定device上，否则计算会出错
        # model_prob = self.one_hot.repeat(target.size(0), 1)

        model_prob = self.one_hot.repeat(target.size(0), 1).to(torch.device(target.get_device()))
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
        # print("LabelSmoothingLoss")
        # print(output.shape, model_prob.shape)
        return F.kl_div(output, model_prob, reduction='sum')


class MyNLLLoss(LossBase):
    def __init__(self, generator, vocab_size, pred=None, target=None, label_smoothing=0.0, pad_id=0):
        super(MyNLLLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = pad_id
        self.generator = generator
        # self.loss_func = torch.nn.BCELoss(reduction='none')
        if label_smoothing > 0:
            self.loss_func = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.loss_func = torch.nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def get_loss(self, pred, target):
        # print("MyNLLLoss")
        # print(pred.shape, target.shape)
        bottled_output = self._bottle(pred)
        scores = self.generator(bottled_output)
        # print(target.size())
        target = target[:, 1:]
        gtruth = target.contiguous().view(-1)

        loss = self.loss_func(scores, gtruth)
        # 不知道原来的这个mask是干什么用的 ?至少presum代码中没有用到mask
        # loss = (loss * mask.float()).sum()
        return loss


class ABSLossMetric(MetricBase):
    def __init__(self, generator, vocab_size, pred=None, target=None, label_smoothing=0.0, pad_id=0):
        super(ABSLossMetric, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = pad_id
        self.generator = generator
        # self.loss_func = torch.nn.BCELoss(reduction='none')
        if label_smoothing > 0:
            self.loss_func = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.loss_func = torch.nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
        self.avg_loss = 0.0
        self.nsamples = 0

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def evaluate(self, pred, target):
        bottled_output = self._bottle(pred)
        scores = self.generator(bottled_output)
        target = target[:, 1:]
        gtruth = target.contiguous().view(-1)
        loss = self.loss_func(scores, gtruth)

        batch_size = pred.size(0)

        self.avg_loss += loss.item()
        self.nsamples += batch_size

    def get_metric(self, reset=True):
        self.avg_loss = self.avg_loss / self.nsamples
        eval_result = {'loss': self.avg_loss}
        if reset:
            self.avg_loss = 0
            self.nsamples = 0
        return eval_result


class MyBCELoss(LossBase):

    def __init__(self, pred=None, target=None, mask=None):
        super(MyBCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')

    def get_loss(self, pred, target, mask):
        loss = self.loss_func(pred, target.float())
        loss = (loss * mask.float()).sum()
        return loss


class EXTLossMetric(MetricBase):
    def __init__(self, pred=None, target=None, mask=None):
        super(EXTLossMetric, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')
        self.avg_loss = 0.0
        self.nsamples = 0

    def evaluate(self, pred, target, mask):
        batch_size = pred.size(0)
        loss = self.loss_func(pred, target.float())
        loss = (loss * mask.float()).sum()
        self.avg_loss += loss
        self.nsamples += batch_size

    def get_metric(self, reset=True):
        self.avg_loss = self.avg_loss / self.nsamples
        eval_result = {'loss': self.avg_loss}
        if reset:
            self.avg_loss = 0
            self.nsamples = 0
        return eval_result


import os
import datetime
from rouge import Rouge
from others.utils import pyrouge_score_all, pyrouge_score_all_multi, ranstr


def remend_score(scores_all):
    remend_score = {}
    for key, value in scores_all.items():
        for subkey, subvalue in value.items():
            remend_score[key + "-" + subkey] = subvalue
    return remend_score


def make_html_safe(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if "<" in s or ">" in s:
        print("-------html not safe sent:")
        print(s)
    return s


class RougeMetricEXT(MetricBase):
    def __init__(self, n_ext=3, ngram_block=3, pred=None, src_txt=None, tgt_txt=None, mask=None, logger=None,
                 config=None):
        super(RougeMetricEXT, self).__init__()
        self._init_param_map(pred=pred, src_txt=src_txt, tgt_txt=tgt_txt, mask=mask)

        self.n_ext = n_ext
        self.ngram_block = ngram_block

        self.referece = []
        self.prediction = []

        self.logger = logger
        self.config = config

    # Set model in validating mode.
    def _get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(self, c, p):
        tri_c = self._get_ngrams(self.ngram_block, c.split())
        for s in p:
            tri_s = self._get_ngrams(self.ngram_block, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def evaluate(self, pred, src_txt, tgt_txt, mask):
        sent_scores = pred + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        # selected_ids = np.sort(selected_ids,1)

        for i, idx in enumerate(selected_ids):
            _pred = []
            if (len(src_txt[i]) == 0):
                continue
            for j in selected_ids[i][:len(src_txt[i])]:
                if (j >= len(src_txt[i])):
                    continue
                candidate = src_txt[i][j].strip()
                if (self.ngram_block):
                    if (not self._block_tri(candidate, _pred)):
                        _pred.append(candidate)
                else:
                    _pred.append(candidate)

                if len(_pred) == self.n_ext:
                    break

            _pred = '\n'.join([make_html_safe(sent) for sent in _pred])
            self.prediction.append(_pred)

            gold_sents = []
            for sent in tgt_txt[i].split('<q>'):
                if len(sent.strip()) > 0:
                    gold_sents.append(sent.strip())
            self.referece.append("\n".join(gold_sents))

    def get_metric(self, reset=True):
        pass


class RougeMetricABS(MetricBase):
    def __init__(self, pred=None, tgt_txt=None, config=None, vocab=None, logger=None):
        super().__init__()

        self.vocab = vocab
        self.config = config
        self._init_param_map(pred=pred, tgt_txt=tgt_txt)

        self.prediction = []
        self.referece = []

        self.logger = logger

    def evaluate(self, pred, tgt_txt):
        """

        :param prediction: [batch, N]
        :param text: [batch, N]
        :param summary: [batch, N]
        :return:
        """

        batch_size = len(pred)

        for b in range(batch_size):
            # print(b,"----------------------",pred[b])
            # output_ids = [int(id) for id in pred[b]]

            pred_str = self.vocab.convert_ids_to_tokens([int(n) for n in pred[b][0]])
            pred_str = ' '.join(pred_str).replace(' ##', '').replace('[unused0]', '').replace('[unused3]', '').replace(
                '[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace(
                '[unused2]', '').strip()
            gold_str = tgt_txt[b]

            pred_sents = []
            for sent in pred_str.split('<q>'):
                if len(sent.strip()) > 0:
                    pred_sents.append(sent.strip())

            abstract_sentences = []
            for sent in gold_str.split('<q>'):
                if len(sent.strip()) > 0:
                    abstract_sentences.append(sent.strip())

            self.prediction.append("\n".join([make_html_safe(sent) for sent in pred_sents]))
            self.referece.append("\n".join([make_html_safe(sent) for sent in abstract_sentences]))

    def get_metric(self, reset=True):
        pass


class FastRougeMetricABS(RougeMetricABS):
    def __init__(self, pred=None, tgt_txt=None, config=None, vocab=None, logger=None):
        super().__init__(pred, tgt_txt, config, vocab, logger)

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(self.prediction, self.referece, avg=True)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        scores_all = remend_score(scores_all)
        return scores_all


class PyRougeMetricABS(RougeMetricABS):
    def __init__(self, pred=None, tgt_txt=None, config=None, vocab=None, logger=None):
        super().__init__(pred, tgt_txt, config, vocab, logger)
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tmp_path = os.path.join(os.path.join(ROOT,'tmp'), "tmp_" + nowTime + "_" + ranstr(6))

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        if isinstance(self.referece[0], list):
            self.logger.info("Multi Reference summaries!")
            scores_all = pyrouge_score_all_multi(self.prediction, self.referece, self.config, self.tmp_path)
        else:
            scores_all = pyrouge_score_all(self.prediction, self.referece, self.config, self.tmp_path)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        return scores_all


class FastRougeMetricEXT(RougeMetricEXT):
    def __init__(self, n_ext=3, ngram_block=3, pred=None, src_txt=None, tgt_txt=None, mask=None, logger=None,
                 config=None):
        super().__init__(n_ext, ngram_block, pred, src_txt, tgt_txt, mask, logger, config)

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(self.prediction, self.referece, avg=True)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        scores_all = remend_score(scores_all)
        return scores_all


class PyRougeMetricEXT(RougeMetricEXT):
    def __init__(self, n_ext=3, ngram_block=3, pred=None, src_txt=None, tgt_txt=None, mask=None, logger=None,
                 config=None):
        super().__init__(n_ext, ngram_block, pred, src_txt, tgt_txt, mask, logger, config)
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tmp_path = os.path.join(os.path.join(ROOT,'tmp'), "tmp_" + nowTime + "_" + ranstr(6))

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        if isinstance(self.referece[0], list):
            self.logger.info("Multi Reference summaries!")
            scores_all = pyrouge_score_all_multi(self.prediction, self.referece, self.config, self.tmp_path)
        else:
            scores_all = pyrouge_score_all(self.prediction, self.referece, self.config, self.tmp_path)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        return scores_all

import numpy as np
import json
from os.path import join
import torch
import logging
import tempfile
import subprocess as sp
from datetime import timedelta
from time import time

from pyrouge import Rouge155
from pyrouge.utils import log

from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase

import torch.nn.functional as F
import torchsnooper

_ROUGE_PATH = '/path/to/RELEASE-1.5.5'
ROOT = "root/path"


class LabelSmoothingLoss(torch.nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    # @torchsnooper.snoop()
    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # 我的修改：在原始代码基础上增加.to(torch.device(target.get_device())),注意这里要迁移到target在的特定device上，否则计算会出错
        # model_prob = self.one_hot.repeat(target.size(0), 1)

        model_prob = self.one_hot.repeat(target.size(0), 1).to(torch.device(target.get_device()))
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
        # print("LabelSmoothingLoss")
        # print(output.shape, model_prob.shape)
        return F.kl_div(output, model_prob, reduction='sum')


class MyNLLLoss(LossBase):
    def __init__(self, generator, vocab_size, pred=None, target=None, label_smoothing=0.0, pad_id=0):
        super(MyNLLLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = pad_id
        self.generator = generator
        # self.loss_func = torch.nn.BCELoss(reduction='none')
        if label_smoothing > 0:
            self.loss_func = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.loss_func = torch.nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def get_loss(self, pred, target):
        # print("MyNLLLoss")
        # print(pred.shape, target.shape)
        bottled_output = self._bottle(pred)
        scores = self.generator(bottled_output)
        # print(target.size())
        target = target[:, 1:]
        gtruth = target.contiguous().view(-1)

        loss = self.loss_func(scores, gtruth)
        # 不知道原来的这个mask是干什么用的 ?至少presum代码中没有用到mask
        # loss = (loss * mask.float()).sum()
        return loss


class ABSLossMetric(MetricBase):
    def __init__(self, generator, vocab_size, pred=None, target=None, label_smoothing=0.0, pad_id=0):
        super(ABSLossMetric, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = pad_id
        self.generator = generator
        # self.loss_func = torch.nn.BCELoss(reduction='none')
        if label_smoothing > 0:
            self.loss_func = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.loss_func = torch.nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
        self.avg_loss = 0.0
        self.nsamples = 0

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def evaluate(self, pred, target):
        bottled_output = self._bottle(pred)
        scores = self.generator(bottled_output)
        target = target[:, 1:]
        gtruth = target.contiguous().view(-1)
        loss = self.loss_func(scores, gtruth)

        batch_size = pred.size(0)

        self.avg_loss += loss.item()
        self.nsamples += batch_size

    def get_metric(self, reset=True):
        self.avg_loss = self.avg_loss / self.nsamples
        eval_result = {'loss': self.avg_loss}
        if reset:
            self.avg_loss = 0
            self.nsamples = 0
        return eval_result


class MyBCELoss(LossBase):

    def __init__(self, pred=None, target=None, mask=None):
        super(MyBCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')

    def get_loss(self, pred, target, mask):
        loss = self.loss_func(pred, target.float())
        loss = (loss * mask.float()).sum()
        return loss


class EXTLossMetric(MetricBase):
    def __init__(self, pred=None, target=None, mask=None):
        super(EXTLossMetric, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')
        self.avg_loss = 0.0
        self.nsamples = 0

    def evaluate(self, pred, target, mask):
        batch_size = pred.size(0)
        loss = self.loss_func(pred, target.float())
        loss = (loss * mask.float()).sum()
        self.avg_loss += loss
        self.nsamples += batch_size

    def get_metric(self, reset=True):
        self.avg_loss = self.avg_loss / self.nsamples
        eval_result = {'loss': self.avg_loss}
        if reset:
            self.avg_loss = 0
            self.nsamples = 0
        return eval_result


import os
import datetime
from rouge import Rouge
from others.utils import pyrouge_score_all, pyrouge_score_all_multi, ranstr


def remend_score(scores_all):
    remend_score = {}
    for key, value in scores_all.items():
        for subkey, subvalue in value.items():
            remend_score[key + "-" + subkey] = subvalue
    return remend_score


def make_html_safe(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if "<" in s or ">" in s:
        print("-------html not safe sent:")
        print(s)
    return s


class RougeMetricEXT(MetricBase):
    def __init__(self, n_ext=3, ngram_block=3, pred=None, src_txt=None, tgt_txt=None, mask=None, logger=None,
                 config=None):
        super(RougeMetricEXT, self).__init__()
        self._init_param_map(pred=pred, src_txt=src_txt, tgt_txt=tgt_txt, mask=mask)

        self.n_ext = n_ext
        self.ngram_block = ngram_block

        self.referece = []
        self.prediction = []

        self.logger = logger
        self.config = config

    # Set model in validating mode.
    def _get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(self, c, p):
        tri_c = self._get_ngrams(self.ngram_block, c.split())
        for s in p:
            tri_s = self._get_ngrams(self.ngram_block, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def evaluate(self, pred, src_txt, tgt_txt, mask):
        sent_scores = pred + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        # selected_ids = np.sort(selected_ids,1)

        for i, idx in enumerate(selected_ids):
            _pred = []
            if (len(src_txt[i]) == 0):
                continue
            for j in selected_ids[i][:len(src_txt[i])]:
                if (j >= len(src_txt[i])):
                    continue
                candidate = src_txt[i][j].strip()
                if (self.ngram_block):
                    if (not self._block_tri(candidate, _pred)):
                        _pred.append(candidate)
                else:
                    _pred.append(candidate)

                if len(_pred) == self.n_ext:
                    break

            _pred = '\n'.join([make_html_safe(sent) for sent in _pred])
            self.prediction.append(_pred)

            gold_sents = []
            for sent in tgt_txt[i].split('<q>'):
                if len(sent.strip()) > 0:
                    gold_sents.append(sent.strip())
            self.referece.append("\n".join(gold_sents))

    def get_metric(self, reset=True):
        pass


class RougeMetricABS(MetricBase):
    def __init__(self, pred=None, tgt_txt=None, config=None, vocab=None, logger=None):
        super().__init__()

        self.vocab = vocab
        self.config = config
        self._init_param_map(pred=pred, tgt_txt=tgt_txt)

        self.prediction = []
        self.referece = []

        self.logger = logger

    def evaluate(self, pred, tgt_txt):
        """

        :param prediction: [batch, N]
        :param text: [batch, N]
        :param summary: [batch, N]
        :return:
        """

        batch_size = len(pred)

        for b in range(batch_size):
            # print(b,"----------------------",pred[b])
            # output_ids = [int(id) for id in pred[b]]

            pred_str = self.vocab.convert_ids_to_tokens([int(n) for n in pred[b][0]])
            pred_str = ' '.join(pred_str).replace(' ##', '').replace('[unused0]', '').replace('[unused3]', '').replace(
                '[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace(
                '[unused2]', '').strip()
            gold_str = tgt_txt[b]

            pred_sents = []
            for sent in pred_str.split('<q>'):
                if len(sent.strip()) > 0:
                    pred_sents.append(sent.strip())

            abstract_sentences = []
            for sent in gold_str.split('<q>'):
                if len(sent.strip()) > 0:
                    abstract_sentences.append(sent.strip())

            self.prediction.append("\n".join([make_html_safe(sent) for sent in pred_sents]))
            self.referece.append("\n".join([make_html_safe(sent) for sent in abstract_sentences]))

    def get_metric(self, reset=True):
        pass


class FastRougeMetricABS(RougeMetricABS):
    def __init__(self, pred=None, tgt_txt=None, config=None, vocab=None, logger=None):
        super().__init__(pred, tgt_txt, config, vocab, logger)

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(self.prediction, self.referece, avg=True)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        scores_all = remend_score(scores_all)
        return scores_all


class PyRougeMetricABS(RougeMetricABS):
    def __init__(self, pred=None, tgt_txt=None, config=None, vocab=None, logger=None):
        super().__init__(pred, tgt_txt, config, vocab, logger)
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tmp_path = os.path.join(os.path.join(ROOT,'tmp'), "tmp_" + nowTime + "_" + ranstr(6))

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        if isinstance(self.referece[0], list):
            self.logger.info("Multi Reference summaries!")
            scores_all = pyrouge_score_all_multi(self.prediction, self.referece, self.config, self.tmp_path)
        else:
            scores_all = pyrouge_score_all(self.prediction, self.referece, self.config, self.tmp_path)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        return scores_all


class FastRougeMetricEXT(RougeMetricEXT):
    def __init__(self, n_ext=3, ngram_block=3, pred=None, src_txt=None, tgt_txt=None, mask=None, logger=None,
                 config=None):
        super().__init__(n_ext, ngram_block, pred, src_txt, tgt_txt, mask, logger, config)

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(self.prediction, self.referece, avg=True)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        scores_all = remend_score(scores_all)
        return scores_all


class PyRougeMetricEXT(RougeMetricEXT):
    def __init__(self, n_ext=3, ngram_block=3, pred=None, src_txt=None, tgt_txt=None, mask=None, logger=None,
                 config=None):
        super().__init__(n_ext, ngram_block, pred, src_txt, tgt_txt, mask, logger, config)
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tmp_path = os.path.join(os.path.join(ROOT,'tmp'), "tmp_" + nowTime + "_" + ranstr(6))

    def get_metric(self, reset=True):
        self.logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            self.logger.error("During testing, no hyps or refers is selected!")
            return
        if isinstance(self.referece[0], list):
            self.logger.info("Multi Reference summaries!")
            scores_all = pyrouge_score_all_multi(self.prediction, self.referece, self.config, self.tmp_path)
        else:
            scores_all = pyrouge_score_all(self.prediction, self.referece, self.config, self.tmp_path)
        if reset:
            self.prediction = []
            self.referece = []
        self.logger.info(scores_all)
        return scores_all
>>>>>>> bcb618ae9faf5d35f3147b05b34268a73808238a
