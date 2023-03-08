import torch.nn as nn
from .resnet_baseline import Bottleneck, conv3x3
import math


class OrigResDecoder(nn.Module):

    def __init__(self,
                 depth):
        super(OrigResDecoder, self).__init__()

        assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
        n = (depth - 2) // 9
        block = Bottleneck

        self.depth = depth
        self.inplanes = 256 # planes * block.expansion

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, n, stride=2)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 16, n)
        self.final_conv_T = nn.ConvTranspose2d(64, 1, 3, padding=1,
                                                bias=False)
        self.bn1 = nn.BatchNorm2d(1)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv_T(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x