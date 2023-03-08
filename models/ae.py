import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .encoder_origres import OrigResEncoder
from .decoder_origres import OrigResDecoder
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torchvision.transforms import Pad
from torch.nn import ModuleDict
import math
import torch

# TODO: make able to use original resnet arch for enc and dec
class PLAutoEncoder(pl.LightningModule):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 latent_size,
                 lr,
                 input_size, tag, encoder, decoder):
        super().__init__()
        self.label = tag
        self.res_layers = num_residual_layers
        self.latent_size = latent_size
        self.num_hiddens = num_hiddens
        self._encoder = encoder
        self._fc_latent = nn.LazyLinear(latent_size)
        self.fc_rl = nn.ReLU()
        self.fc_bn = nn.LazyBatchNorm1d()

        # initializing enc and lazy linear
        dummy = torch.zeros(64, 1, input_size[0], input_size[1])
        print(dummy.size())
        dummy = self._encoder(dummy)
        self.enc_dim = dummy.size()
        dummy = torch.flatten(dummy, start_dim=1)
        pre_latent_dim = dummy.size(1)
        #dummy = torch.cat((dummy, torch.zeros(size=(64, 4))), dim=1)
        self._fc_dec = nn.LazyLinear(pre_latent_dim)
        self.dec_rl = nn.ReLU()
        self.dec_bn = nn.LazyBatchNorm1d()

        dummy = self._fc_latent(dummy)
        dummy = self.fc_rl(dummy)
        dummy = self.fc_bn(dummy)
        self._fc_dec(dummy)
        dummy = self.dec_rl(dummy)
        dummy = self.dec_bn(dummy)
        self._decoder = decoder
        self.criterion = nn.MSELoss()
        self.lr = lr

    def encode(self, x, qual_values):
        enc = self._encoder(x)
        pre_latent = enc.flatten(start_dim=1)
        #pre_latent = torch.cat((pre_latent, qual_values), dim=1)
        z = self._fc_latent(pre_latent)
        return z

    def get_encoder(self):
        return {'encoder': self._encoder, 'fc_latent': self._fc_latent}

    def decode(self, z):
        dec = self._fc_dec(z)
        pre_recon = dec.view(-1, self.enc_dim[1], self.enc_dim[2], self.enc_dim[3])
        x_recon = self._decoder(pre_recon)
        return x_recon

    def forward(self, x, qual_values):
       # print(torch.isnan(qual_values).any())
        z = self.encode(x, qual_values)
        out = self.decode(z)
        return out

    def generate(self, x, qual_values):
        return self.forward(x, qual_values)

    def get_encoding(self, x, qual_values):
        return self.encode(x, qual_values)

    def training_step(self, batch, batch_idx):
        orig_img, jigsaw_img, qual_labels = batch
        recon = self.forward(jigsaw_img, qual_labels)
        loss = self.criterion(recon, orig_img)
        self.log('train_loss', loss, on_epoch=True)
        self.log('recon_pixel_max', recon[-1, :, :, :].max(), on_epoch=True)
        self.log('recon_pixel_min', recon[-1, :, :, :].min(), on_epoch=True)
        tensorboard = self.logger.experiment
        if batch_idx % 20:
            orig_grid = make_grid(orig_img[0, 0, :, :])
            jigsaw_grid = make_grid(jigsaw_img[0, 0, :, :])
            recon_grid = make_grid(torch.clamp(recon[0, 0, :, :].detach(), 0, 1))
            tensorboard.add_image('train_jigsaw_images', jigsaw_grid)
            tensorboard.add_image('train_orig_images', orig_grid)
            tensorboard.add_image('train_recon_images', recon_grid)
        return loss

    def validation_step(self, batch, batch_idx):
        orig_img, jigsaw_img, qual_labels = batch
        recon = self.forward(jigsaw_img, qual_labels)
        loss = self.criterion(recon, orig_img)
        self.log('val_loss', loss, on_epoch=True)
        self.orig_img = orig_img[0, 0, :, :]
        self.recon = recon[0, 0, :, :]
        self.jigsaw_img = jigsaw_img[0, 0, :, :]
        return loss

    def validation_step_end(self, outputs):
        orig_grid = make_grid(torch.clamp(self.orig_img, 0, 1))
        recon_grid = make_grid(torch.clamp(self.recon, 0, 1))
        jigsaw_grid = make_grid(torch.clamp(self.recon, 0, 1))
        tb = self.logger.experiment
        tb.add_image('val_end_orig_grid', orig_grid)
        tb.add_image('val_end_recon_grid', recon_grid)
        tb.add_image('val_end_jigsaw_grid', jigsaw_grid)
        return



    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        return optimizer


class PLWindowAE(pl.LightningModule):
    def __init__(self,
                 num_channels,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 total_bottleneck_size,
                 lr,
                 input_size,
                 tile_size=32):
        super().__init__()
        self.tile_length = tile_size
        self.tiling = (math.ceil(input_size[0] / tile_size), math.ceil(input_size[1] / tile_size))
        self.pad_dim = (self.tiling[0] * self.tile_length - input_size[0],
                        self.tiling[1] * self.tile_length - input_size[1])
        self.pad = Pad(padding=self.pad_dim)


        # calculate window sizes and stuff
        # make dict of fc layers for the window models

def get_ae(num_channels,
           num_hiddens,
           num_residual_layers,
           num_residual_hiddens,
           latent_size,
           lr,
           input_size,
           tag,
           res_type):
    assert res_type in ['custom', 'orig']
    if res_type == 'custom':
        encoder = Encoder(in_channels=num_channels,
                          num_hiddens=num_hiddens,
                          num_residual_layers=num_residual_layers,
                          num_residual_hiddens=num_residual_hiddens)
        decoder = Decoder(out_channels=num_channels,
                          num_hiddens=num_hiddens,
                          num_residual_layers=num_residual_layers,
                          num_residual_hiddens=num_residual_hiddens)
    elif res_type == 'orig':
        encoder = OrigResEncoder(num_residual_layers)
        decoder = OrigResDecoder(num_residual_layers)
    ae = PLAutoEncoder(num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
                       latent_size=latent_size,
                       lr=lr,
                       input_size=input_size,
                       tag=tag,
                       encoder=encoder,
                       decoder=decoder)
    return ae



