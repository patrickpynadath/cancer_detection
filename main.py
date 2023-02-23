import os
from cmd_utils import config_diffusion_train_cmd, config_diffusion_generate_cmd, \
    config_data_processing_cmd, config_split_data_cmd, \
    config_resnet_train_cmd, config_transfer_learn_ae
from processing import MammographyPreprocessor, get_paths, get_diffusion_dataloaders, get_clf_dataloaders,\
    split_data, get_ae_loaders
import argparse
from models import resnet_from_args, get_diffusion_model_from_args, get_trained_diff_model, \
    create_save_artificial_samples, get_window_model, get_pl_ae, PLAutoEncoder
from pytorch_lightning import Trainer
from training import generic_training_loop, diffusion_training_loop
from imbalanced_rl_clf import ImbalancedClfEnv, RLTrainer, Agent
import torch
# data preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")
    subparsers = parser.add_subparsers(dest='command')

    process_data = subparsers.add_parser('process_data', help = 'command for processing data')
    process_data = config_data_processing_cmd(process_data)

    train_resnet = subparsers.add_parser('train_resnet', help = 'command to train resnet clf')
    train_resnet = config_resnet_train_cmd(train_resnet)

    train_transfer_learn_ae = subparsers.add_parser('train_transfer_learn_ae', help='train transfer learning autoencoder')
    train_transfer_learn_ae = config_transfer_learn_ae(train_transfer_learn_ae)

    train_diffusion = subparsers.add_parser('train_diffusion', help='command for training diffusion models')
    train_diffusion = config_diffusion_train_cmd(train_diffusion)

    generate_imgs = subparsers.add_parser('generate_imgs', help='command for creating artificial positive samples')
    generate_imgs = config_diffusion_generate_cmd(generate_imgs)

    generate_splits = subparsers.add_parser('generate_splits', help = 'generating the splits to use for resnet and diffusion')
    generate_splits = config_split_data_cmd(generate_splits)

    train_rl = subparsers.add_parser('train_rl', help = 'train rl policy net')

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
                 args.latent_size, args.lr)
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

    if args.command == 'train_rl':
        device = 'cuda'
        trained_jigsaw_ae = PLAutoEncoder.load_from_checkpoint(
            'lightning_logs/version_188/checkpoints/epoch=168-step=115596.ckpt',
            num_channels=1,
            num_hiddens=256,
            num_residual_layers=20,
            num_residual_hiddens=256,
            latent_size=1024, lr=.01, input_size=(128, 64))
        trainloader, test_loader = get_ae_loaders('data', 32, (128, 64), 32, 'jigsaw')
        trained_jigsaw_ae.to(device)
        encoder = trained_jigsaw_ae.encode
        env = ImbalancedClfEnv(trainloader.dataset, device)
        agent = Agent(2, 0.05, 0.9, 1000, encoder, device, 10000, 64, .01)
        trainer = RLTrainer(.99, .005, env, agent, device)
        trainer.train_loop(600)