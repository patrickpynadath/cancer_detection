import os

from processing import MammographyPreprocessor, get_paths, get_diffusion_dataloaders, get_clf_dataloaders,\
    split_data, get_ae_loaders
import argparse
from models import resnet_from_args, get_diffusion_model_from_args, get_trained_diff_model, \
    create_save_artificial_samples, get_window_model, get_pl_ae
from pytorch_lightning import Trainer
from training import generic_training_loop, diffusion_training_loop
import torch
from vit_pytorch.cait import CaiT
# data preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")
    subparsers = parser.add_subparsers(dest='command')
    process_data = subparsers.add_parser('process_data', help = 'command for processing data')
    train_resnet = subparsers.add_parser('train_resnet', help = 'command to train resnet clf')
    train_window_model = subparsers.add_parser('train_window_model', help = 'command to run training of window based model')
    train_transfer_learn_ae = subparsers.add_parser('train_transfer_learn_ae', help='train transfer learning autoencoder')
    resnet_model_flags = train_resnet.add_argument_group('model_flags')
    resnet_data_flags = train_resnet.add_argument_group('data_flags')

    train_diffusion = subparsers.add_parser('train_diffusion', help='command for training diffusion models')
    diffusion_model_flags = train_diffusion.add_argument_group('model_flags')
    diffusion_data_flags = train_diffusion.add_argument_group('data_flags')

    generate_imgs = subparsers.add_parser('generate_imgs', help='command for creating artificial positive samples')

    generate_splits = subparsers.add_parser('generate_splits', help = 'generating the splits to use for resnet and diffusion')
    generate_splits.add_argument('--test_size', default=.2, type=float, help='ratio to use for train-test splits')
    generate_splits.add_argument('--base_data_dir', help='base dir for data', type=str, default='data')

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

    resnet_data_flags.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    resnet_data_flags.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    resnet_data_flags.add_argument('--num_pos', help = 'number of positive samples for training', default = 8000, type=int)
    resnet_data_flags.add_argument('--loader_workers', help='workers for dataloader', type = int, default = 1)
    resnet_data_flags.add_argument('--synthetic_dir', help='dir for synthetic data', default=None)

    train_window_model.add_argument('--window_size', help = 'window size to use for model', default = 16, type=int)
    train_window_model = Trainer.add_argparse_args(train_window_model)
    train_window_model.add_argument('--synthetic_dir', help='dir for synthetic data', default=None)
    train_window_model.add_argument('--input_height', default = 128, type=int, help='input img height')
    train_window_model.add_argument('--input_width', default=64, type=int, help='input img width')
    train_window_model.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    train_window_model.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    train_window_model.add_argument('--num_pos', help='number of positive samples for training', default=8000, type=int)
    train_window_model.add_argument('--loader_workers', help='workers for dataloader', type=int, default=1)
    trainer_flags = Trainer.add_argparse_args(train_resnet)

    # train jigsaw autoencoder args
    train_transfer_learn_ae = Trainer.add_argparse_args(train_transfer_learn_ae)
    train_transfer_learn_ae.add_argument('--input_height', default = 128, type=int, help='input img height')
    train_transfer_learn_ae.add_argument('--input_width', default=64, type=int, help='input img width')
    train_transfer_learn_ae.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    train_transfer_learn_ae.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    train_transfer_learn_ae.add_argument('--latent_size', help='size of latent dim', default = 200, type=int)
    train_transfer_learn_ae.add_argument('--learning_mode', help='task to use for training', default='normal', type=str)
    train_transfer_learn_ae.add_argument('--num_hiddens', help='num hiddens for resstack', default = 512, type=int)
    train_transfer_learn_ae.add_argument('--num_residual_layers', help='num residual layers', default=20, type=int)
    train_transfer_learn_ae.add_argument('--num_residual_hiddens', help='num hiddens for residuals', default=64, type=int)
    train_transfer_learn_ae.add_argument('--tile_size', help='size of tiles for fillin/jigsaw tasks', type=int, default=16)


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

    generate_imgs.add_argument('--img_height', default=128, type=int, help='input img height')
    generate_imgs.add_argument('--img_width', default=64, type=int, help='input img width')
    generate_imgs.add_argument('--save_name', default='total_cancer_results/model-99.pt', type=str, help='name of stored state dict for diffusion model')
    generate_imgs.add_argument('--num_samples', default=2000, type=int, help = 'num of samples to be generated')
    generate_imgs.add_argument('--batch_size', default=64, type=int, help = 'batch size for generating imgs')
    generate_imgs.add_argument('--device', default='cpu', type=str, help = 'device to use')


    # training XGBoost args

    args = parser.parse_args()

    if args.command == 'process_data':
        final_size = (args.finalwidth, args.finalheight)
        base_dir = args.base_data_dir
        mp = MammographyPreprocessor(size=final_size, csv_path = f'{base_dir}/train.csv', train_path=f'{base_dir}/train_images')
        paths = get_paths()
        mp.preprocess_all(paths, parallel=args.par, save=True, save_dir=f'{base_dir}/train_images')

    elif args.command == 'train_resnet':
        train_loader, test_loader = get_clf_dataloaders(args.base_dir, args.num_pos, args.batch_size,
                                                        synthetic_dir=args.synthetic_dir, grad_data=True)

        pl_resnet = resnet_from_args(args, 2)
        generic_training_loop(args, pl_resnet, train_loader, test_loader)
        if args.synthetic_dir:
            torch.save(pl_resnet.resnet.state_dict(), 'pl_resnet_synthetic.pickle')
        else:
            torch.save(pl_resnet.resnet.state_dict(), 'pl_resnet_oversample.pickle')


        # train_loader, val_loader, test_loader = over_sample_loader(args, 250, 100, 100)
        # pl_resnet = resnet_from_args(args, get_num_classes(args.target_col, args.base_dir))
        # resnet_training_loop(args, pl_resnet, train_loader, val_loader)
        # torch.save(pl_resnet.resnet.state_dict(), 'resnet_500samples.pickle')
        #
        # train_loader, val_loader, test_loader = get_balanced_loaders(args, 250, 100, 100)
        # pl_resnet = resnet_from_args(args, get_num_classes(args.target_col, args.base_dir))
        # resnet_training_loop(args, pl_resnet, train_loader, val_loader)
        # torch.save(pl_resnet.resnet.state_dict(), 'resnet_250samples.pickle')
        #
        # train_loader, val_loader, test_loader = get_balanced_loaders(args, 500, 100, 100)
        # pl_resnet = resnet_from_args(args, get_num_classes(args.target_col, args.base_dir))
        # resnet_training_loop(args, pl_resnet, train_loader, val_loader)
        # torch.save(pl_resnet.resnet.state_dict(), 'resnet_500samples.pickle')
    elif args.command == 'train_window_model':
        train_loader, test_loader = get_clf_dataloaders(args.base_dir, args.num_pos, args.batch_size,
                                                        synthetic_dir=args.synthetic_dir, grad_data=True)
        window_model = get_window_model(args.window_size, (args.input_height, args.input_width))
        generic_training_loop(args, window_model, train_loader, test_loader)
        torch.save(window_model.model.state_dict(), 'window model state dict')

    elif args.command == 'train_diffusion':
        train_loader, test_loader = get_diffusion_dataloaders(args.base_dir, args.batch_size)
        diffusion_model = get_diffusion_model_from_args(args)
        diffusion_training_loop(diffusion_model, train_loader, 'total_cancer_results')
        torch.save(diffusion_model.model.state_dict(), 'diff_cancer_model.pickle')

    elif args.command == 'train_transfer_learn_ae':
        train_loader, test_loader = get_ae_loaders(args.base_dir, args.tile_size, (args.input_height, args.input_width), args.batch_size, args.learning_mode)
        ae = get_pl_ae(args.num_channels,
                 args.num_hiddens,
                 args.num_residual_layers,
                 args.num_residual_hiddens,
                 args.latent_size)
        generic_training_loop(args, ae, train_loader, test_loader)
        torch.save(ae.model.state_dict(), f'ae_tl_{args.learning_mode}.pickle')


    elif args.command == 'generate_imgs':
        os.makedirs('artificial_pos_samples', exist_ok = True)
        diff_model = get_trained_diff_model(args.save_name, (args.img_height, args.img_width))
        create_save_artificial_samples(diff_model, args.num_samples, 'artificial_pos_samples', device=args.device, batch_size=args.batch_size)


    elif args.command == 'generate_splits':
        test_size = args.test_size
        base_dir = args.base_data_dir
        split_data(test_size, base_dir)
