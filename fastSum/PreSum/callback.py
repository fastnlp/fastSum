import os
import torch
import sys
from torch import nn

from fastNLP.core.callback import Callback, EarlyStopError
from others.utils import mkdir
from fastNLP.core.utils import _get_model_device
import fitlog
from copy import deepcopy
from fastNLP import Tester, DataSet


class MyCallback(Callback):
    def __init__(self, args, optims):
        super(MyCallback, self).__init__()
        self.args = args
        self.real_step = 0
        self.optims = optims

    def on_step_end(self):
        if self.step % self.update_every == 0 and self.step > 0:
            self.real_step += 1
            cur_lr = []
            for o in self.optims:
                cur_lr.append("{:.8f}".format(o.optimizer.param_groups[0]['lr']))
                o.step()

            if self.real_step % 1000 == 0:
                self.pbar.write('Current learning rate is {}, real_step: {}'.format("|".join(cur_lr), self.real_step))

    def on_epoch_end(self):
        self.pbar.write('Epoch {} is done !!!'.format(self.epoch))


def _save_model(checkpoint, model_name, save_dir, only_param=True):
    """ 存储不含有显卡信息的 state_dict 或 model
    这里还没有实现保存optims, 相关实现可以继续
    :param model:
    :param model_name:
    :param save_dir: 保存的 directory
    :param only_param:
    :return:
    """

    model_path = os.path.join(save_dir, model_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(checkpoint['model'], nn.DataParallel):
        checkpoint['model'] = checkpoint['model'].module
    if only_param:
        state_dict = checkpoint['model'].state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        checkpoint['model'] = state_dict
        torch.save(checkpoint, model_path)
    else:
        _model_device = _get_model_device(checkpoint['model'])
        checkpoint['model'].cpu()
        torch.save(checkpoint, model_path)
        checkpoint['model'].to(_model_device)


class SaveModelCallback(Callback):
    """
    由于Trainer在训练过程中只会保存最佳的模型， 该 callback 可实现多种方式的结果存储。
    会根据训练开始的时间戳在 save_dir 下建立文件夹，在再文件夹下存放多个模型
    -save_dir
        -2019-07-03-15-06-36
            -epoch0step20{metric_key}{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_perfomance是性能
            -epoch1step40
        -2019-07-03-15-10-00
            -epoch:0step:20{metric_key}:{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_perfomance是性能
    :param str save_dir: 将模型存放在哪个目录下，会在该目录下创建以时间戳命名的目录，并存放模型
    :param int top: 保存dev表现top多少模型。-1为保存所有模型
    :param bool only_param: 是否只保存模型权重
    :param save_on_exception: 发生exception时，是否保存一份当时的模型
    """

    def __init__(self, save_dir, optims, args, top=5, only_param=False, save_on_exception=False):
        super().__init__()

        if not os.path.isdir(save_dir):
            raise IsADirectoryError("{} is not a directory.".format(save_dir))
        self.save_dir = save_dir
        if top < 0:
            self.top = sys.maxsize
        else:
            self.top = top
        self.optims = optims
        self._ordered_save_models = []  # List[Tuple], Tuple[0]是metric， Tuple[1]是path。metric是依次变好的，所以从头删

        self.only_param = only_param
        self.save_on_exception = save_on_exception
        self.args = args

    def on_train_begin(self):
        self.save_dir = os.path.join(self.save_dir, self.trainer.start_time)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        metric_value = list(eval_result.values())[0][metric_key]
        self._save_this_model(metric_value)

    def _insert_into_ordered_save_models(self, pair):
        # pair:(metric_value, model_name)
        # 返回save的模型pair与删除的模型pair. pair中第一个元素是metric的值，第二个元素是模型的名称
        index = -1
        for _pair in self._ordered_save_models:
            if _pair[0] >= pair[0] and self.trainer.increase_better:
                break
            if not self.trainer.increase_better and _pair[0] <= pair[0]:
                break
            index += 1
        save_pair = None
        if len(self._ordered_save_models) < self.top or (len(self._ordered_save_models) >= self.top and index != -1):
            save_pair = pair
            self._ordered_save_models.insert(index + 1, pair)
        delete_pair = None
        if len(self._ordered_save_models) > self.top:
            delete_pair = self._ordered_save_models.pop(0)
        return save_pair, delete_pair

    def _save_this_model(self, metric_value):
        name = "epoch:{}_step:{}_{}:{:.6f}.pt".format(self.epoch, self.step, self.trainer.metric_key, metric_value)
        save_pair, delete_pair = self._insert_into_ordered_save_models((metric_value, name))
        checkpoint = {'model': self.model, 'optims': self.optims, 'opt': self.args}
        if save_pair:
            try:
                _save_model(checkpoint, model_name=name, save_dir=self.save_dir)
            except Exception as e:
                print(f"The following exception:{e} happens when saves model to {self.save_dir}.")
        if delete_pair:
            try:
                delete_model_path = os.path.join(self.save_dir, delete_pair[1])
                if os.path.exists(delete_model_path):
                    os.remove(delete_model_path)
            except Exception as e:
                print(f"Fail to delete model {name} at {self.save_dir} caused by exception:{e}.")

    def on_exception(self, exception):
        if self.save_on_exception:
            checkpoint = {'model': self.model, 'optims': self.optims, 'opt': self.args}
            name = "epoch:{}_step:{}_Exception:{}.pt".format(self.epoch, self.step, exception.__class__.__name__)
            _save_model(checkpoint, model_name=name, save_dir=self.save_dir)


class EarlyStopCallback(Callback):
    r"""
    多少个epoch没有变好就停止训练，相关类 :class:`~fastNLP.core.callback.EarlyStopError`
    """

    def __init__(self, patience=10):
        r"""

        :param int patience: epoch的数量
        """
        super(EarlyStopCallback, self).__init__()
        self.patience = patience
        self.wait = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            print("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error


class FitlogCallback(Callback):
    r"""
    该callback可将loss和progress写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
    一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
    并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
    fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=False):
        r"""

        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要
            传入多个DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。data的结果的名称以'data'开头。
        :param ~fastNLP.Tester,Dict[~fastNLP.Tester] tester: Tester对象，将在on_valid_end时调用。tester的结果的名称以'tester'开头
        :param int log_loss_every: 多少个step记录一次loss(记录的是这几个batch的loss平均值)，如果数据集较大建议将该值设置得
            大一些，不然会导致log文件巨大。默认为0, 即不要记录loss。
        :param int verbose: 是否在终端打印evaluation的结果，0不打印。
        :param bool log_exception: fitlog是否记录发生的exception信息
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        assert isinstance(log_loss_every, int) and log_loss_every >= 0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm,
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

    def on_backward_begin(self, loss):
        if self._log_loss_every > 0:
            self._avg_loss += loss.item()
            if self.step % self._log_loss_every == 0:
                fitlog.add_loss(self._avg_loss / self._log_loss_every * self.update_every, name='loss',
                                step=int(self.step / self.update_every),
                                epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')
