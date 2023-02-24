import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# naive implementation of paper http://203.170.84.89/~idawis33/DataScienceLab/publication/IJCNN15.wang.final.pdf
class MSFELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MSFELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        neg_idx = (target == 0).nonzero()
        pos_idx = (target == 1).nonzero()
        FPE = F.mse_loss(input[pos_idx], target[pos_idx])
        FNE = F.mse_loss(input[neg_idx], target[neg_idx])
        return FPE ** 2 + FNE ** 2