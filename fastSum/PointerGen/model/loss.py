import torch
import numpy as np

from fastNLP.core.losses import LossBase

def is_infinite_loss(losses):
    for loss in losses:
        if not (np.isfinite(loss.data.cpu())).numpy():
            # print("the infinite loss is :", loss)
            return True
    return False

class MyLoss(LossBase):
    def __init__(self, config=None, target=None, list_final_dist=None, list_coverage=None, list_attn=None,
                 max_dec_len=None, dec_padding_mask=None, dec_lens_var=None, padding_idx=-100, reduce='mean'):
        super().__init__()
        self._init_param_map(target=target, list_final_dist=list_final_dist, list_coverage=list_coverage,
                             list_attn=list_attn, max_dec_len=max_dec_len, dec_padding_mask=dec_padding_mask,
                             dec_lens_var=dec_lens_var)
        self.padding_idx = padding_idx
        self.reduce = reduce
        self.config = config

    def get_loss(self, target, list_final_dist, list_coverage, list_attn, max_dec_len, dec_padding_mask, dec_lens_var):
        step_losses = []
        config = self.config
        target_batch = target
        if config.use_gpu:
            target = target.cuda()
            target_batch = target_batch.cuda()

        for di in range(min(max_dec_len, config.max_dec_steps)):

            target = target_batch[:, di]
            final_dist = list_final_dist[di]
            attn_dist = list_attn[di]

            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)

            if config.is_coverage:
                coverage = list_coverage[di]
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            # if is_infinite_loss(step_loss):
                # print("catch inifite loss: ", step_loss)
                # print("gold_probs: ", gold_probs)
                # print("final_dist: ", final_dist)
                # print("target: ", target)
            step_losses.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var.float()

        loss = torch.mean(batch_avg_loss)

        # if not (np.isfinite(loss.data.cpu())).numpy():
            # print("dec_lens_var: ", dec_lens_var)

        return loss
