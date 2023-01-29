from processing import MammographyPreprocessor, get_paths, get_loaders_from_args, get_num_classes
import argparse
from models import resnet_from_args, get_diffusion_model_from_args
from pytorch_lightning import Trainer
from training import resnet_training_loop, diffusion_training_loop

# data preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")
    subparsers = parser.add_subparsers(dest='command')
    process_data = subparsers.add_parser('process_data', help = 'command for processing data')

    train_resnet = subparsers.add_parser('train_resnet', help = 'command to train resnet clf')
    resnet_model_flags = train_resnet.add_argument_group('model_flags')
    resnet_data_flags = train_resnet.add_argument_group('data_flags')

    train_diffusion = subparsers.add_parser('train_diffusion', help='command for training diffusion models')
    diffusion_model_flags = train_diffusion.add_argument_group('model_flags')
    diffusion_data_flags = train_diffusion.add_argument_group('data_flags')

    # preprocessing data args
    process_data.add_argument('--finalheight', default=128, type=int, help='final img height for processing')
    process_data.add_argument('--finalwidth', default=64, type=int, help='final img width for processing')
    process_data.add_argument('--par', help='run processing in parralel', action='store_true')
    process_data.add_argument('--base_data_dir', help='base dir for data', type=str, default='data')

    # training resnet data args
    resnet_model_flags.add_argument('--device', help ='device to train model on', default='cpu', type=str)
    resnet_model_flags.add_argument('--lr', help ='lr to use for adam optimizer', default = 1e-3, type=float)
    resnet_model_flags.add_argument('--depth', help ='layers for resnet', default=56, type=int)
    resnet_model_flags.add_argument('--block_name', help ='block name to use for resnet', choices=['BasicBlock', 'BottleNeck'], default='BottleNeck')

    resnet_data_flags.add_argument('--target_col', help='target column for training', default='cancer', type=str)
    resnet_data_flags.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    resnet_data_flags.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    resnet_data_flags.add_argument('--test_size', help='ratio for size of validation set', default=.25, type=float)
    resnet_data_flags.add_argument('--loader_workers', help='workers for dataloader', type = int, default = 1)

    trainer_flags = Trainer.add_argparse_args(train_resnet)

    # training diffusion models args
    diffusion_model_flags.add_argument('--img_height', default = 128, type=int, help='input img height')
    diffusion_model_flags.add_argument('--img_width', default=64, type=int, help='input img width')
    diffusion_model_flags.add_argument('--timesteps', default=1000, type=int, help='timesteps for diffusion model')
    diffusion_model_flags.add_argument('--loss_type', default='l2', choices=['l2', 'l1'], help='loss type for diffusion output')

    diffusion_data_flags.add_argument('--mimic_col', default='cancer', type=str, help='target col to train diffusion model to generate')
    diffusion_data_flags.add_argument('--mimic_val', default=1, help='the desired value the samples should mimic from target_col')
    diffusion_data_flags.add_argument('--base_dir', default='data', type=str, help='base dir for data')
    diffusion_data_flags.add_argument('--test_size', default=.1, type=float, help='test ratio size')
    diffusion_data_flags.add_argument('--batch_size', default=32, type=int,help='batch size')
    diffusion_data_flags.add_argument('--loader_workers', default=32, type=int,help='num workers for data loader')
    diffusion_data_flags.add_argument('--target_col', default='cancer', type=str, help='target col')
    # training VAE args


    # training XGBoost args

    args = parser.parse_args()

    if args.command == 'process_data':
        final_size = (args.finalwidth, args.finalheight)
        base_dir = args.base_data_dir
        mp = MammographyPreprocessor(size=final_size, csv_path = f'{base_dir}/train.csv', train_path=f'{base_dir}/train_images')
        paths = get_paths()
        mp.preprocess_all(paths, parallel=args.par, save=True, save_dir=f'{base_dir}/train_images')

    elif args.command == 'train_resnet':
        train_loader, val_loader, test_loader = get_loaders_from_args(args)
        pl_resnet = resnet_from_args(args, get_num_classes(args.target_col, args.base_dir))
        resnet_training_loop(args, pl_resnet, train_loader, val_loader)

    elif args.command == 'train_diffusion':
        train_loader, val_loader, test_loader = get_loaders_from_args(args, mimic_col=args.mimic_col, mimic_val=args.mimic_val)
        diffusion_model = get_diffusion_model_from_args(args)
        diffusion_training_loop(diffusion_model, train_loader)




