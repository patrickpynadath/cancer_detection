import os
from cmd_utils import config_diffusion_train_cmd, config_diffusion_generate_cmd, \
    config_data_processing_cmd, config_split_data_cmd, \
    config_resnet_train_cmd, config_transfer_learn_ae, \
    config_rl_train_cmd
from processing import MammographyPreprocessor, get_paths, get_diffusion_dataloaders, get_clf_dataloaders,\
    split_data, get_ae_loaders
import argparse
from models import get_diffusion_model_from_args, get_trained_diff_model, \
    create_save_artificial_samples, get_pl_ae, PLAutoEncoder, MSFELoss, ImbalancedLoss, \
    ResNet, PL_ResNet
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

    train_transfer_learn_clf = subparsers.add_parser('train_transfer_learn_clf', help ='command to train clf')
    train_transfer_learn_clf = config_resnet_train_cmd(train_transfer_learn_clf)

    train_resnet = subparsers.add_parser('train_resnet_clf', help = 'train resnet clf')
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
    train_rl = config_rl_train_cmd(train_rl)

    args = parser.parse_args()

    if args.command == 'process_data':
        final_size = (args.finalwidth, args.finalheight)
        base_dir = args.base_data_dir
        mp = MammographyPreprocessor(size=final_size, csv_path = f'{base_dir}/train.csv', train_path=f'{base_dir}/train_images')
        paths = get_paths()
        mp.preprocess_all(paths, parallel=args.par, save=True, save_dir=f'{base_dir}/train_images')

    if args.command == 'train_resnet_clf':
        tag = f'{args.sample_strat}/{args.criterion}/resnet_baseline'
        input_size =(args.input_height, args.input_width)
        resnet = ResNet(depth=args.depth, tag=tag, input_size=input_size).to(args.device)
        if args.criterion == 'CE':
            criterion = torch.nn.CrossEntropyLoss()
        elif args.criterion == 'MSFE':
            criterion = MSFELoss()
        clf = PL_ResNet(resnet, .001, criterion=criterion)
        train_loader, test_loader = get_clf_dataloaders(args.base_dir,
                                                        args.batch_size,
                                                        32,
                                                        input_size,
                                                        sample_strat=args.sample_strat)
        generic_training_loop(args, clf, train_loader, test_loader, tag)




    elif args.command == 'train_transfer_learn_clf':
        assert args.sample_strat in ['none', 'rus', 'ros', 'dynamic_ros', 'dynamic_kmeans_ros']
        input_size = (args.input_height, args.input_width)
        device = 'cpu'
        if args.accelerator == 'gpu':
            device = 'cuda'

        tag = 'samplestrat_' + args.sample_strat + '/'

        path = None

        if args.learning_mode == 'normal':
            path = TRAINED_NORMAL_PATH
            tag += 'normal/'
        elif args.learning_mode == 'jigsaw':
            path = TRAINED_JIGSAW_PATH
            tag += 'jigsaw/'
        trained_ae = PLAutoEncoder.load_from_checkpoint(
            path,
            num_channels=1,
            num_hiddens=256,
            num_residual_layers=20,
            num_residual_hiddens=256,
            latent_size=1024, lr=.01, input_size=(128, 64)).to(device)
        pretrained = trained_ae.get_encoder()
        mlp = Generic_MLP(encoder=pretrained['encoder'], fc_latent=pretrained['fc_latent'])
        criterion = None
        labels_dtype = torch.long
        tag += f'use_encoder_params_{args.use_encoder_params}'
        if args.criterion == 'CE':
            if args.sample_strat == 'none':
                criterion = CrossEntropyLoss(weight=torch.Tensor([.05, 1]))
            else:
                criterion = CrossEntropyLoss()
            tag += 'CE/'
        elif args.criterion == 'MSFE':
            criterion = MSFELoss()
            tag += 'MSFE/'
            labels_dtype = torch.float32

        elif args.criterion == 'IMB':
            criterion = ImbalancedLoss(mode=args.sim_calc)
            tag += 'IMB/'
            labels_dtype = torch.float32

        train_loader, test_loader = get_clf_dataloaders(args.base_dir,
                                                        args.batch_size,
                                                        tile_length=args.tile_size,
                                                        sample_strat=args.sample_strat,
                                                        input_size=input_size,
                                                        device=args.device,
                                                        learning_mode=args.learning_mode,
                                                        kmeans_clusters=args.kmeans_clusters,
                                                        encoder=trained_ae.encode,
                                                        label_dtype=labels_dtype,
                                                        update_beta=args.balancing_beta)
        tag += 'mlp_clf'
        if 'dynamic' in args.sample_strat:
            print(tag)

            trainer = DynamicSamplingTrainer(model=mlp,
                                             device=device,
                                             tag=tag,
                                             train_loader=train_loader,
                                             test_loader=test_loader,
                                             log_dir=LOG_DIR,
                                             lr=args.lr,
                                             use_encoder_params=args.use_encoder_params,
                                             criterion=criterion)
            trainer.training_loop(args.epochs)
            torch.save(mlp.state_dict(), f'{args.sample_strat}_model_sd.pickle')

        else:
            clf = PL_MLP_clf(mlp, criterion, args.lr, use_encoder_params=args.use_encoder_params)
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
        generic_training_loop(args, ae, train_loader, test_loader,
                              model_name=f'ae_lz_{args.latent_size}_learnmode_{args.learning_mode}')
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
        agent = Agent(2, args.eps_end, args.eps_start, args.eps_decay, encoder, device, 10000, args.batch_size, args.lr, QModel=Generic_MLP)
        trainer = RLTrainer(args.gamma, args.tau, env, agent, device, test_loader)
        trainer.train_loop(args.updates)