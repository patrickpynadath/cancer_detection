from argparse import ArgumentParser
from pytorch_lightning import Trainer


def config_resnet_train_cmd(args: ArgumentParser):
    model_flags = args.add_argument_group('model_flags')
    data_flags = args.add_argument_group('data_flags')

    model_flags.add_argument('--device', help='device to train model on', default='cpu', type=str)
    model_flags.add_argument('--lr', help='lr to use for adam optimizer', default=1e-3, type=float)
    model_flags.add_argument('--depth', help='layers for resnet', default=110, type=int)
    model_flags.add_argument('--block_name', help='block name to use for resnet',
                                    choices=['BasicBlock', 'BottleNeck'], default='BottleNeck')
    model_flags.add_argument('--criterion', help='criterion to use', default = 'CE', type=str)
    model_flags.add_argument('--use_encoder_params', help='include encoder params for optim', type=bool, default=False)
    model_flags.add_argument('--epochs', help='epochs to run for', default=200,  type=int)

    Trainer.add_argparse_args(args)
    data_flags.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    data_flags.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    data_flags.add_argument('--num_pos', help = 'number of positive samples for training', default = 8000, type=int)
    data_flags.add_argument('--loader_workers', help='workers for dataloader', type = int, default = 1)
    data_flags.add_argument('--synthetic_dir', help='dir for synthetic data', default=None)
    data_flags.add_argument('--tile_size', help='tile size for jigsaw', type=int, default=32)
    data_flags.add_argument('--input_height', default=128, type=int, help='input img height')
    data_flags.add_argument('--input_width', default=64, type=int, help='input img width')
    data_flags.add_argument('--sample_strat', help='flag for sampling strat', default='none')
    data_flags.add_argument('--sim_calc', help='flag for vector sim metric', default='geo')
    data_flags.add_argument('--learning_mode', help='training mode for clf', type=str, default='normal')
    data_flags.add_argument('--kmeans_clusters', help='number of clusters for kmeans', default=20)
    data_flags.add_argument('--balancing_beta', help='hyper param for dynamic resampling, controls how much the resampling ratio can change by',
                            type=float, default=.25)

    return args
