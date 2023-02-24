import os
from cmd_utils import config_diffusion_train_cmd, config_diffusion_generate_cmd, \
    config_data_processing_cmd, config_split_data_cmd, \
    config_resnet_train_cmd, config_transfer_learn_ae, \
    config_rl_train_cmd
from processing import MammographyPreprocessor, get_paths, get_diffusion_dataloaders, get_clf_dataloaders,\
    split_data, get_ae_loaders
import argparse
from models import get_diffusion_model_from_args, get_trained_diff_model, \
    create_save_artificial_samples, get_pl_ae, PLAutoEncoder, MSFELoss
from training import generic_training_loop, diffusion_training_loop, DynamicSamplingTrainer
from imbalanced_rl_clf import ImbalancedClfEnv, RLTrainer, Agent, Generic_MLP, PL_MLP_clf
import torch
from torch.nn import CrossEntropyLoss
# data preprocessing

TRAINED_JIGSAW_PATH = 'lightning_logs/version_188/checkpoints/epoch=168-step=115596.ckpt'
TRAINED_NORMAL_PATH = 'lightning_logs/version_187/checkpoints/epoch=108-step=74556.ckpt'
LOG_DIR = 'lightning_logs/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")
    subparsers = parser.add_subparsers(dest='command')

    process_data = subparsers.add_parser('process_data', help = 'command for processing data')
    process_data = config_data_processing_cmd(process_data)

    train_clf = subparsers.add_parser('train_clf', help ='command to train clf')
    train_clf = config_resnet_train_cmd(train_clf)

    train_transfer_learn_ae = subparsers.add_parser('train_transfer_learn_ae', help='train transfer learning autoencoder')
    train_transfer_learn_ae = config_transfer_learn_ae(train_transfer_learn_ae)

    train_diffusion = subparsers.add_parser('train_diffusion', help='command for training diffusion models')
    train_diffusion = config_diffusion_train_cmd(train_diffusion)

    generate_imgs = subparsers.add_parser('generate_imgs', help='command for creating artificial positive samples')
    generate_imgs = config_diffusion_generate_cmd(generate_imgs)

    generate_splits = subparsers.add_parser('generate_splits', help = 'generating the splits to use for resnet and diffusion')
    generate_splits = config_split_data_cmd(generate_splits)

    train_rl = subparsers.add_parser('train_rl', help = 'train rl policy net')
    train_rl = config_rl_train_cmd(train_rl)

    args = parser.parse_args()

    if args.command == 'process_data':
        final_size = (args.finalwidth, args.finalheight)
        base_dir = args.base_data_dir
        mp = MammographyPreprocessor(size=final_size, csv_path = f'{base_dir}/train.csv', train_path=f'{base_dir}/train_images')
        paths = get_paths()
        mp.preprocess_all(paths, parallel=args.par, save=True, save_dir=f'{base_dir}/train_images')

    elif args.command == 'train_clf':
        assert args.oversample_method in ['none', 'normal_ros', 'dynamic_ros', 'dynamic_kmeans_ros']
        input_size = (args.input_height, args.input_width)

        tag = '' + args.oversample_method

        path = None

        if args.training_mode == 'normal':
            path = TRAINED_NORMAL_PATH
            tag += 'normal/'
        elif args.training_mode == 'jigsaw':
            path = TRAINED_JIGSAW_PATH
            tag += 'jigsaw/'
        trained_ae = PLAutoEncoder.load_from_checkpoint(
            path,
            num_channels=1,
            num_hiddens=256,
            num_residual_layers=20,
            num_residual_hiddens=256,
            latent_size=1024, lr=.01, input_size=(128, 64))
        encoder = trained_ae.encode
        mlp = Generic_MLP(encoder)
        criterion = None
        if args.criterion == 'CE':
            criterion = CrossEntropyLoss()
            tag += 'CE/'
        elif args.criterion == 'MSFE':
            criterion = MSFELoss()
            tag += 'MSFE/'
        train_loader, test_loader = get_clf_dataloaders(args.base_dir, args.num_pos, args.batch_size,
                                                        tile_length=args.tile_size,

                                                        synthetic_dir=args.synthetic_dir,
                                                        oversample=args.oversample_method,
                                                        input_size=input_size,
                                                        device=args.device,
                                                        learning_mode=args.learning_mode,
                                                        kmeans_clusters=args.kmeans_clusters,
                                                        encoder=encoder)
        tag += 'mlp_clf'
        clf = PL_MLP_clf(mlp, criterion)
        if 'dynamic' in args.oversample_method:
            trainer = DynamicSamplingTrainer(mlp, args.device, tag, train_loader, test_loader, LOG_DIR, args.lr)
            trainer.training_loop(args.epochs)

        generic_training_loop(args, clf, train_loader, test_loader, tag)
        torch.save(clf.model.mp.state_dict(), f'{tag}.pickle')

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
        create_save_artificial_samples(diff_model, args.num_samples, 'artificial_pos_samples',
                                       device=args.device, batch_size=args.batch_size)


    elif args.command == 'generate_splits':
        test_size = args.test_size
        base_dir = args.base_data_dir
        split_data(test_size, base_dir)

    if args.command == 'train_rl':
        device = 'cuda'
        path = None
        if args.training_mode == 'normal':
            path = TRAINED_NORMAL_PATH
        elif args.training_mode == 'jigsaw':
            path = TRAINED_JIGSAW_PATH
        trained_jigsaw_ae = PLAutoEncoder.load_from_checkpoint(
            path,
            num_channels=1,
            num_hiddens=256,
            num_residual_layers=20,
            num_residual_hiddens=256,
            latent_size=1024, lr=.01, input_size=(128, 64))
        size = (args.input_height, args.input_width)
        trainloader, test_loader = get_ae_loaders(args.base_dir, 32, size, args.batch_size, 'jigsaw')
        trained_jigsaw_ae.to(device)
        encoder = trained_jigsaw_ae.encode
        env = ImbalancedClfEnv(trainloader.dataset, device)
        agent = Agent(2, args.eps_end, args.eps_start, args.eps_decay, encoder, device, 10000, args.batch_size, args.lr)
        trainer = RLTrainer(args.gamma, args.tau, env, agent, device, test_loader)
        trainer.train_loop(args.updates)