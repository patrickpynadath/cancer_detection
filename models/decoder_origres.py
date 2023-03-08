import torch.nn as nn
from .resnet_baseline import BottleNeckTranspose
import math


class OrigResDecoder(nn.Module):

    def __init__(self,
                 depth):
        super(OrigResDecoder, self).__init__()

        assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
        n = (depth - 2) // 9

        self.depth = depth
        self.inplanes = 64 # planes * block.expansion

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, n, stride=2)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(16, n)
        self.final_conv_T = nn.ConvTranspose2d(16, 1, 3,
                                                padding=1,
                                                bias=False)
        self.bn1 = nn.BatchNorm2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * BottleNeckTranspose.expansion:
            print('asd')
            upsample = nn.Sequential(
                nn.ConvTranspose2d(planes * BottleNeckTranspose.expansion, self.inplanes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(BottleNeckTranspose(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * BottleNeckTranspose.expansion
        for i in range(1, blocks):
            layers.append(BottleNeckTranspose(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv_T(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x