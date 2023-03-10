from argparse import ArgumentParser


def config_data_processing_cmd(args: ArgumentParser):
    args.add_argument('--finalheight', default=128, type=int, help='final img height for processing')
    args.add_argument('--finalwidth', default=64, type=int, help='final img width for processing')
    args.add_argument('--par', help='run processing in parralel', action='store_true')
    args.add_argument('--base_data_dir', help='base dir for data', type=str, default='data')
    return args


def config_split_data_cmd(args: ArgumentParser):
    args.add_argument('--test_size', default=.2, type=float, help='ratio to use for train-test splits')
    args.add_argument('--base_data_dir', help='base dir for data', type=str, default='data')
    return args


def config_split_data_CIFAR_cmd(args: ArgumentParser):
    args.add_argument('--base_dir', default='cifar-10-batches-py', help='base dir for cifar')
    args.add_argument('--minority_sample_ratio', default=1, type=float, help='ratio of total positive class to include in train/test datasets')