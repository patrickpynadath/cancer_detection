import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class ImbalancedLoss(_Loss):
    def __init__(self, size_average=None, reduce=None,
                 reduction: str = 'mean', mode: str = 'geo',
                 eps = .01):
        super(ImbalancedLoss, self).__init__(size_average, reduce, reduction)
        self.mode = mode
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # making target into batch x 1 x 2
        new_target = torch.ones_like(input, device=input.device)
        new_target[:, 1] = target
        new_target[:, 0] = new_target[:, 0] - target

        get_metric = lambda score_type: self._get_score(input, new_target, score_type)

        fp = .1 * get_metric('fp')
        fn = 2 * get_metric('fn')
        tp = 2 * get_metric('tp')
        tn = .1 * get_metric('tn')


        return fp + fn - tp - tn

    def _get_score(self, input: Tensor, target: Tensor, score_type: str) -> Tensor:
        assert score_type in ['fp', 'fn', 'tn', 'tp']
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
            num = torch.matmul(input_row, target_row)
            denom = (torch.linalg.norm(input_row) * torch.linalg.norm(target_row)) + self.eps

        elif self.mode == 'info':
            num = -1 * torch.matmul(torch.log(input_row), target_row)
            denom = torch.linalg.vector_norm(torch.linalg.norm(target_row)) + self.eps
        return num / denom




