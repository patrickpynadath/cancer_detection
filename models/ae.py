import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Generative.encoder import Encoder
from Models.Generative.decoder import Decoder
from Utils import timestamp


class AE(nn.Module):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 latent_size):
        super(AE, self).__init__()
        self.res_layers = num_residual_layers
        self.latent_size=latent_size
        self.num_hiddens = num_hiddens
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._fc_latent = nn.Linear(num_hiddens * 8 * 8, latent_size)

        self._fc_dec = nn.Linear(latent_size, num_hiddens * 8 * 8)
        self._decoder = Decoder(num_hiddens,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.label = f'AE_{timestamp()}'

    def encode(self, x):
        enc = self._encoder(x)
        pre_latent = enc.flatten(start_dim=1)
        z = self._fc_latent(pre_latent)
        return z

    def decode(self, z):
        dec = self._fc_dec(z)
        pre_recon = dec.view(-1, self.num_hiddens, 8, 8)
        x_recon = self._decoder(pre_recon)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)

    def get_encoding(self, x):
        return self.encode(x)


def get_latent_code_ae(ae: AE, x):
    return ae.encode(x)
