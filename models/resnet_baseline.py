from __future__ import absolute_import

'''
This file is from: https://raw.githubusercontent.com/bearpaw/pytorch-classification/master/models/cifar/resnet.py
by Wei Yang
'''
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule
import math
from .normalization_layers import get_normalize_layer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3T(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleNeckTranspose(nn.Module):
    expansion = 4

    def __init__(self, outplanes, planes, stride=1, downsample=None):
        super(BottleNeckTranspose, self).__init__()
        self.convT1 = nn.ConvTranspose2d(planes * 4, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.convT2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride,
                                         padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.convT3 = nn.ConvTranspose2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convT1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.convT2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.convT3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 depth,
                 tag,
                 input_size,
                 block_name='BottleNeck', device='cuda'):
        super(ResNet, self).__init__()
        self.label = tag
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.depth = depth
        self.norm_layer = get_normalize_layer()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.LazyLinear(2)

        # needed for lazy init
        dummy = torch.zeros(64, 1, input_size[0], input_size[1])
        self.forward(dummy)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PL_ResNet(LightningModule):
    def __init__(self, model, lr, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.epoch_val = 0
        self.train_pred = []
        self.train_actual = []
        self.val_pred = []
        self.val_actual = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        orig, _, y = batch
        logits = self.forward(orig)
        loss = self.criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        # appending pred and actual values to field for access at epoch end
        self.train_pred += [pred[i].item() for i in range(len(pred))]
        self.train_actual += [y[i].item() for i in range(len(pred))]
        self.log('train/loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        orig, jigsaw, y = batch
        logits = self.forward(jigsaw)
        loss = self.criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        self.val_pred += [pred[i].item() for i in range(len(pred))]
        self.val_actual += [y[i].item() for i in range(len(pred))]
        tb = self.logger.experiment
        tb.add_scalar(f'val/loss', loss.item(), self.epoch_val)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr, eps=.01)
        return optimizer

    def on_train_epoch_end(self) -> None:
        tb = self.logger.experiment
        res_dct = get_metrics(self.train_actual, self.train_pred)
        for k in res_dct.keys():
            tb.add_scalar(f'train/{k}', res_dct[k], self.epoch_val)
        self.epoch_val += 1
        self.train_pred = []
        self.train_actual = []
        return

    def on_validation_epoch_end(self) -> None:
        tb = self.logger.experiment
        res_dct = get_metrics(self.val_actual, self.val_pred)
        for k in res_dct.keys():
            tb.add_scalar(f'val/{k}', res_dct[k], self.epoch_val)
        self.val_pred = []
        self.val_actual = []
        return

def get_metrics(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    roc = roc_auc_score(true, pred)
    num_pos_pred = sum(pred)
    return {'acc' : acc, 'f1' : f1, 'roc_auc' : roc, 'num_pos_pred' : num_pos_pred}


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)