from processing import MammographyPreprocessor, get_paths
import argparse
from pytorch_lightning import Trainer




# data preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")
    subparsers = parser.add_subparsers(dest='command')
    process_data = subparsers.add_parser('process_data', help = 'command for processing data')

    train_resnet = subparsers.add_parser('train_resnet', help = 'command to train resnet clf')
    model_flags = train_resnet.add_argument_group('model_flags')
    data_flags = train_resnet.add_argument_group('data_flags')

    # preprocessing data args
    process_data.add_argument('--finalheight', default=256, type=int, help='final img height for processing')
    process_data.add_argument('--finalwidth', default=128, type=int, help='final img width for processing')
    process_data.add_argument('--par', help='run processing in parralel', action='store_false')
    process_data.add_argument('--base_data_dir', help='base dir for data', type=str, default='data')

    # training resnet data args
    model_flags.add_argument('--lr', help = 'lr to use for adam optimizer', default = 1e-3, type=float)
    model_flags.add_argument('--depth', help = 'layers for resnet', default=56, type=int)
    model_flags.add_argument('--block_name', help = 'block name to use for resnet', choices=['BasicBlock', 'BottleNeck'], default='BottleNeck')

    data_flags.add_argument('--target_col', help='target column for training', default='cancer', type=str)
    data_flags.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    data_flags.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    data_flags.add_argument('--test_size', help='ratio for size of validation set', default=.25, type=float)

    trainer_flags = Trainer.add_argparse_args(train_resnet)
    # training diffusion models args

    # training VAE args


    # training XGBoost args

    args = parser.parse_args()

    if args.command == 'process_data':
        final_size = (args.finalwidth, args.finalheight)
        base_dir = args.base_data_dir
        mp = MammographyPreprocessor(size=final_size, csv_path = f'{base_dir}/train.csv', train_path=f'{base_dir}/train_images')

        paths = get_paths()
        mp.preprocess_all(paths, parallel=args.par)
    elif args.command == 'train_resnet':
        print(args)


