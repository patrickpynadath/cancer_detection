from argparse import ArgumentParser
from pytorch_lightning import Trainer


def config_transfer_learn_ae(args: ArgumentParser):
    args = Trainer.add_argparse_args(args)
    args.add_argument('--device', default='cpu')
    args.add_argument('--input_height', default=128, type=int, help='input img height')
    args.add_argument('--input_width', default=64, type=int, help='input img width')
    args.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    args.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    args.add_argument('--latent_size', help='size of latent dim', default=1024, type=int)
    args.add_argument('--learning_mode', help='task to use for training', default='normal', type=str)
    args.add_argument('--num_hiddens', help='num hiddens for resstack', default=256, type=int)
    args.add_argument('--num_residual_layers', help='num residual layers', default=20, type=int)
    args.add_argument('--num_residual_hiddens', help='num hiddens for residuals', default=256,
                                         type=int)
    args.add_argument('--tile_size', help='size of tiles for fillin/jigsaw tasks', type=int,
                                         default=16)
    args.add_argument('--num_channels', help='num input channels', default=1, type=int)
    args.add_argument('--lr', help='lr for model', default=1e-5, type=float)
    args.add_argument('--res_type', help='type of residual blocks to use', default='custom')
    return args
