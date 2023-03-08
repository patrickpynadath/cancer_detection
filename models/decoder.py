import torch.nn as nn
import torch.nn.functional as F
from .res_stack import ResidualStack

class Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens):
        super(Decoder, self).__init__()


        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens,
                                                kernel_size=2,
                                                stride=1, padding=0, bias=False)
        self.rl1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_hiddens)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1, bias=False)
        self.rl2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_hiddens//2)

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens //2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1, bias=False)
        self.rl3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = self.rl1(x)
        x = self.bn1(x)
        # #x = F.relu(x)
        x = self._conv_trans_2(x)
        x = self.rl2(x)
        x = self.bn2(x)
        #x = F.relu(x)
        x = self._conv_trans_3(x)
        x = self.rl3(x)
        x = self.bn3(x)


        return x