import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class ImbalancedLoss(_Loss):
    def __init__(self, size_average=None, reduce=None,
                 reduction: str = 'mean', mode: str = 'info',
                 eps = 1e-7):
        super(ImbalancedLoss, self).__init__(size_average, reduce, reduction)
        self.mode = mode
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # making target into batch x 1 x 2
        new_target = torch.ones_like(input, device=input.device)
        new_target[:, 1] = target
        new_target[:, 0] = new_target[:, 0] - target

        get_metric = lambda score_type: self._get_score(input, new_target, score_type)

        fp = get_metric('fp')
        fn = get_metric('fn')
        tp = get_metric('tp')
        tn = get_metric('tn')

        prec_sur = tp / (tp + fp + self.eps)
        rec_sur = tp / (tp + fn + self.eps)
        spec_sur = tn / (tn + fp + self.eps)
        return - 1 * prec_sur * rec_sur * spec_sur

    def _get_score(self, input: Tensor, target: Tensor, score_type: str) -> Tensor:
        assert score_type in ['fp', 'fn', 'tn', 'tp']
        dummy = torch.ones(size=(target.size(0)), device=target.device)
        if score_type == 'tp':
            input_row = input[:, 1]
            target_row = target[:, 1]
        elif score_type == 'fp':
            input_row = input[:, 1]
            target_row = target[:, 0]
        elif score_type == 'tn':
            input_row = input[:, 0]
            target_row = target[:, 0]
        elif score_type == 'fn':
            input_row = input[:, 0]
            target_row = target[:, 1]

        if self.mode == 'geo':
            num = torch.linalg.vecdot(input, target_row)
            denom = (torch.linalg.norm(input_row) * torch.linalg.norm(target_row)) + self.eps

        elif self.mode == 'info':
            num = -1 * torch.linalg.vecdot(torch.log(input_row), target_row)
            denom = torch.linalg.vector_norm(torch.linalg.norm(target_row)) + self.eps

        return num / denom




