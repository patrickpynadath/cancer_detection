from argparse import ArgumentParser
from pytorch_lightning import Trainer


def config_resnet_train_cmd(args: ArgumentParser):
    model_flags = args.add_argument_group('model_flags')
    data_flags = args.add_argument_group('data_flags')

    model_flags.add_argument('--device', help='device to train model on', default='cpu', type=str)
    model_flags.add_argument('--lr', help='lr to use for adam optimizer', default=1e-3, type=float)
    model_flags.add_argument('--depth', help='layers for resnet', default=56, type=int)
    model_flags.add_argument('--block_name', help='block name to use for resnet',
                                    choices=['BasicBlock', 'BottleNeck'], default='BottleNeck')

    Trainer.add_argparse_args(args)
    data_flags.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    data_flags.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    data_flags.add_argument('--num_pos', help = 'number of positive samples for training', default = 8000, type=int)
    data_flags.add_argument('--loader_workers', help='workers for dataloader', type = int, default = 1)
    data_flags.add_argument('--synthetic_dir', help='dir for synthetic data', default=None)
    return args
