import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
import pytorch_lightning as pl
from torchvision.utils import make_grid
import torch


class PLAutoEncoder(pl.LightningModule):
    def __init__(self,
                 num_channels,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 latent_size, lr, input_size):
        super().__init__()
        self.res_layers = num_residual_layers
        self.latent_size = latent_size
        self.num_hiddens = num_hiddens
        self._encoder = Encoder(num_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._fc_latent = nn.LazyLinear(latent_size)
        # initializing enc and lazy linear
        dummy_in = torch.zeros(64, 5, input_size[0], input_size[1])
        dummy_out = torch.flatten(self._encoder(dummy_in), start_dim=1)
        dummy_fc_in = torch.cat((dummy_out, torch.zeros(size=(64, 4))), dim=1)
        self._fc_latent(dummy_fc_in)

        self._fc_dec = nn.Linear(latent_size, num_hiddens * 8 * 8)
        self._decoder = Decoder(num_hiddens,
                                num_channels,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def encode(self, x, qual_values):
        enc = self._encoder(x)
        pre_latent = enc.flatten(start_dim=1)
        pre_latent = torch.cat((pre_latent, qual_values), dim=1)
        z = self._fc_latent(pre_latent)
        return z

    def decode(self, z):
        dec = self._fc_dec(z)
        pre_recon = dec.view(-1, self.num_hiddens, 8, 8)
        x_recon = self._decoder(pre_recon)
        return x_recon

    def forward(self, x, qual_values):
        z = self.encode(x, qual_values)
        return self.decode(z)

    def generate(self, x, qual_values):
        return self.forward(x, qual_values)

    def get_encoding(self, x, qual_values):
        return self.encode(x, qual_values)

    def training_step(self, batch, batch_idx):
        orig_img, jigsaw_img, qual_labels = batch
        recon = self.forward(jigsaw_img, qual_labels)
        loss = self.criterion(recon, orig_img)
        self.log('train_loss', loss, on_epoch=True)
        tensorboard = self.logger.experiment
        orig_grid = make_grid(orig_img[0, 0, :, :])
        jigsaw_grid = make_grid(jigsaw_img[0, 0, :, :])
        recon_grid = make_grid(recon[0, 0, :, :])
        tensorboard.add_image('train_jigsaw_images', jigsaw_grid)
        tensorboard.add_image('train_orig_images', orig_grid)
        tensorboard.add_image('train_recon_images', recon_grid)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        orig_img, jigsaw_img, qual_labels = batch
        recon = self.forward(jigsaw_img, qual_labels)
        loss = self.criterion(recon, orig_img)
        self.log('val_loss', loss, on_epoch=True)
        tensorboard = self.logger.experiment
        orig_grid = make_grid(orig_img[0, 0, :, :])
        jigsaw_grid = make_grid(jigsaw_img[0, 0, :, :])
        recon_grid = make_grid(recon[0, 0, :, :])
        tensorboard.add_image('val_jigsaw_images', jigsaw_grid)
        tensorboard.add_image('val_orig_images', orig_grid)
        tensorboard.add_image('val_recon_images', recon_grid)
        print(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


def get_pl_ae(num_channels,
             num_hiddens,
             num_residual_layers,
             num_residual_hiddens,
             latent_size, lr):
    return PLAutoEncoder(num_channels,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 latent_size, lr, (128, 64))
