import torch
import torch.nn as nn
from .resnet_clf import Bottleneck
import math
from torchvision.transforms import Pad
from skimage.measure import shannon_entropy


class WindowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.LazyConv2d(out_channels=32, kernel_size=3)
        self.inplanes = 32
        self.group1 = self._make_layer(planes=16, blocks=3)
        self.group2 = self._make_layer(planes=32, blocks=3)
        self.fc_out = nn.LazyLinear(2)


    def forward(self, x):
        x = self.input_conv(x)
        x = self.group1(x)
        x = self.group2(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc_out(x)
        return out

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)


class EnsembleModel(nn.Module):
    def __init__(self, window_size, input_size):
        super().__init__()
        img_ensemble_dct = {}
        gap_size = window_size // 2
        x_windows = math.ceil(input_size[0]/gap_size) - 1
        y_windows = math.ceil(input_size[1]/gap_size) - 1
        for i in range(x_windows):
            for j in range(y_windows):
                img_ensemble_dct[(i, j)] = WindowModel()
        self.window_size = window_size
        self.x_windows = x_windows
        self.y_windows = y_windows
        x_pad = ((x_windows-1) * gap_size) + window_size - input_size[0]
        y_pad = ((y_windows-1) * gap_size) + window_size - input_size[1]
        self.gap_size = gap_size
        self.pad = Pad(padding=(x_pad, y_pad))
        self.network_ensemble = img_ensemble_dct

    def _get_window(self, img, x_idx, y_idx):
        return img[:, :, x_idx * self.gap_size:(x_idx * self.gap_size) + self.window_size,
               y_idx * self.gap_size: (y_idx * self.gap_size) + self.window_size]

    def _get_entropy_weights(self, img):
        total_entropies = []
        for k in range(img.size(0)):
            tmp_entropies = []
            for i in range(self.x_windows):
                for j in range(self.y_windows):
                    window = self._get_window(img, i, j)
                    tmp_entropies.append(shannon_entropy(window.to('cpu').numpy()))
            total_entropies.append(tmp_entropies)
        return torch.tensor(total_entropies, device= img.device)

    # NOTE: the padding size is in the forward size because it depends on parameters of the model itself
    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        window_out = []
        for i in range(self.x_windows):
            for j in range(self.y_windows):
                tmp_window = self._get_window(x, i, j)
                out = self.network_ensemble[(i, j)](tmp_window)
                window_out.append(out)
        window_out = torch.stack(window_out, dim=1)
        entropy = self._get_entropy_weights(x)
        weights = nn.functional.softmax(entropy, dim=1)
        final_out = torch.zeros(x.size(0), 2)
        print(weights.size())
        print(window_out.size())
        for batch_idx in range(x.size(0)):
            for k in range(weights.size(1)):
                final_out[batch_idx, :] += weights[batch_idx, k] * window_out[batch_idx, k, :]
        return final_out




