import os
from cmd_utils import config_diffusion_train_cmd, config_diffusion_generate_cmd, \
    config_data_processing_cmd, config_split_data_cmd, \
    config_resnet_train_cmd, config_transfer_learn_ae, \
    config_rl_train_cmd, config_split_data_CIFAR_cmd
from processing import MammographyPreprocessor, get_paths, get_diffusion_dataloaders, get_clf_dataloaders,\
    split_data_RSNA, get_ae_loaders_RSNA, split_data_CIFAR, get_cifar_sets
import argparse
from models import get_diffusion_model_from_args, get_trained_diff_model, \
    create_save_artificial_samples, get_ae, PLAutoEncoder, MSFELoss, ImbalancedLoss, \
    ResNet, PL_ResNet, Generic_MLP, PL_MLP_clf
from training import generic_training_loop, diffusion_training_loop, DynamicSamplingTrainer
from imbalanced_rl_clf import ImbalancedClfEnv, RLTrainer, Agent
import torch
from torch.nn import CrossEntropyLoss
# data preprocessing

TRAINED_JIGSAW_PATH = 'lightning_logs/version_188/checkpoints/epoch=168-step=115596.ckpt'
TRAINED_NORMAL_PATH = 'lightning_logs/version_187/checkpoints/epoch=108-step=74556.ckpt'
LOG_DIR = 'lightning_logs/'


# cleaning up code
def process_data(cmd_args):
    final_size = (cmd_args.finalwidth, cmd_args.finalheight)
    base_dir = cmd_args.base_data_dir
    mp = MammographyPreprocessor(size=final_size, csv_path=f'{base_dir}/train.csv',
                                 train_path=f'{base_dir}/train_images')
    paths = get_paths()
    mp.preprocess_all(paths, parallel=cmd_args.par, save=True, save_dir=f'{base_dir}/train_images')
    return


def train_resnet_clf(cmd_args):
    tag = f'{cmd_args.sample_strat}/{cmd_args.criterion}/resnet_baseline'
    input_size = (cmd_args.input_height, cmd_args.input_width)
    resnet = ResNet(depth=cmd_args.depth, tag=tag, input_size=input_size).to(cmd_args.device)
    if cmd_args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif cmd_args.criterion == 'MSFE':
        criterion = MSFELoss()
    clf = PL_ResNet(resnet, .001, criterion=criterion)
    train_loader, test_loader = get_clf_dataloaders(cmd_args.base_dir,
                                                    cmd_args.batch_size,
                                                    32,
                                                    input_size,
                                                    sample_strat=cmd_args.sample_strat)
    generic_training_loop(cmd_args, clf, train_loader, test_loader, tag)
    return


def make_cifar_splits(cmd_args):
    for r in [1, .8, .5, .25, .1, .05]:
        base_dir = cmd_args.base_dir
        train, test = get_cifar_sets()
        split_data_CIFAR(r, base_dir, train, 'train')
        split_data_CIFAR(r, base_dir, test, 'test')
    return


def train_transfer_learn_clf(cmd_args):
    assert cmd_args.sample_strat in ['none', 'rus', 'ros', 'dynamic_ros', 'dynamic_kmeans_ros']
    input_size = (cmd_args.input_height, cmd_args.input_width)
    device = 'cpu'
    if cmd_args.accelerator == 'gpu':
        device = 'cuda'

    tag = 'samplestrat_' + cmd_args.sample_strat + '/'

    path = None
    if args.fast_dev_run:
        epochs = 1
    else:
        epochs = cmd_args.epochs
    if cmd_args.learning_mode == 'normal':
        path = TRAINED_NORMAL_PATH
        tag += 'normal/'
    elif cmd_args.learning_mode == 'jigsaw':
        path = TRAINED_JIGSAW_PATH
        tag += 'jigsaw/'
    blank_ae = get_ae(num_channels=1,
                      num_hiddens=256,
                      num_residual_layers=20,
                      num_residual_hiddens=256,
                      latent_size=1024,
                      lr=.01,
                      input_size=(128, 64),
                      res_type=args.res_type,
                      tag=tag)
    trained_ae = PLAutoEncoder.load_from_checkpoint(
        path,
        latent_size=1024, lr=.01, input_size=(128, 64), tag=tag, encoder=blank_ae._encoder, decoder=blank_ae._decoder).to(device)
    pretrained = trained_ae.get_encoder()
    mlp = Generic_MLP(encoder=pretrained['encoder'],
                      fc_latent=pretrained['fc_latent'])
    criterion = None
    labels_dtype = torch.long
    tag += f'use_encoder_params_{cmd_args.use_encoder_params}'
    if cmd_args.criterion == 'CE':
        if cmd_args.sample_strat == 'none':
            criterion = CrossEntropyLoss(weight=torch.Tensor([.05, 1]))
        else:
            criterion = CrossEntropyLoss()
        tag += 'CE/'
    elif cmd_args.criterion == 'MSFE':
        criterion = MSFELoss()
        tag += 'MSFE/'
        labels_dtype = torch.float32

    elif cmd_args.criterion == 'IMB':
        criterion = ImbalancedLoss(mode=cmd_args.sim_calc)
        tag += 'IMB/'
        labels_dtype = torch.float32

    train_loader, test_loader = get_clf_dataloaders(cmd_args.base_dir,
                                                    cmd_args.batch_size,
                                                    tile_length=cmd_args.tile_size,
                                                    sample_strat=cmd_args.sample_strat,
                                                    input_size=input_size,
                                                    device=cmd_args.device,
                                                    learning_mode=cmd_args.learning_mode,
                                                    kmeans_clusters=cmd_args.kmeans_clusters,
                                                    encoder=trained_ae.encode,
                                                    label_dtype=labels_dtype,
                                                    update_beta=cmd_args.balancing_beta)
    tag += 'mlp_clf'
    if 'dynamic' in cmd_args.sample_strat:
        print(tag)

        trainer = DynamicSamplingTrainer(model=mlp,
                                         device=device,
                                         tag=tag,
                                         train_loader=train_loader,
                                         test_loader=test_loader,
                                         log_dir=LOG_DIR,
                                         lr=cmd_args.lr,
                                         use_encoder_params=cmd_args.use_encoder_params,
                                         criterion=criterion)
        trainer.training_loop(epochs)
        torch.save(mlp.state_dict(), f'{cmd_args.sample_strat}_model_sd.pickle')

    else:
        clf = PL_MLP_clf(mlp, criterion, cmd_args.lr, use_encoder_params=cmd_args.use_encoder_params)
        generic_training_loop(cmd_args, clf, train_loader, test_loader, tag)
        torch.save(clf.model.mp.state_dict(), f'{tag}.pickle')
    return


def train_diffusion(cmd_args):
    train_loader, test_loader = get_diffusion_dataloaders(cmd_args.base_dir, cmd_args.batch_size)
    diffusion_model = get_diffusion_model_from_args(cmd_args)
    diffusion_training_loop(diffusion_model, train_loader, 'total_cancer_results')
    torch.save(diffusion_model.model.state_dict(), 'diff_cancer_model.pickle')
    return


def train_transfer_learn_ae(cmd_args):
    input_size = (cmd_args.input_height, cmd_args.input_width)
    train_loader, test_loader = get_ae_loaders_RSNA(cmd_args.base_dir, cmd_args.tile_size, (cmd_args.input_height, cmd_args.input_width),
                                                    cmd_args.batch_size, cmd_args.learning_mode)
    tag = f'ae_lz_{cmd_args.latent_size}_learnmode_{cmd_args.learning_mode}_res_{args.res_type}'
    ae = get_ae(cmd_args.num_channels,
                cmd_args.num_hiddens,
                cmd_args.num_residual_layers,
                cmd_args.num_residual_hiddens,
                cmd_args.latent_size, cmd_args.lr, input_size=input_size, tag=tag, res_type=cmd_args.res_type)
    generic_training_loop(cmd_args, ae, train_loader, test_loader,
                          model_name=tag)
    return


def generate_imgs(cmd_args):
    os.makedirs('artificial_pos_samples', exist_ok=True)
    diff_model = get_trained_diff_model(cmd_args.save_name, (cmd_args.img_height, cmd_args.img_width))
    create_save_artificial_samples(diff_model, cmd_args.num_samples, 'artificial_pos_samples',
                                   device=cmd_args.device, batch_size=cmd_args.batch_size)
    return


def generate_splits(cmd_args):
    test_size = cmd_args.test_size
    base_dir = cmd_args.base_data_dir
    split_data_RSNA(test_size, base_dir)
    return


def train_rl(cmd_args):
    device = 'cuda'
    path = None
    if cmd_args.training_mode == 'normal':
        path = TRAINED_NORMAL_PATH
    elif cmd_args.training_mode == 'jigsaw':
        path = TRAINED_JIGSAW_PATH
    trained_jigsaw_ae = PLAutoEncoder.load_from_checkpoint(
        path,
        num_channels=1,
        num_hiddens=256,
        num_residual_layers=20,
        num_residual_hiddens=256,
        latent_size=1024, lr=.01, input_size=(128, 64))
    size = (cmd_args.input_height, cmd_args.input_width)
    trainloader, test_loader = get_ae_loaders_RSNA(cmd_args.base_dir, 32, size, cmd_args.batch_size, 'jigsaw')
    trained_jigsaw_ae.to(device)
    encoder = trained_jigsaw_ae.encode
    env = ImbalancedClfEnv(trainloader.dataset, device)
    agent = Agent(2, cmd_args.eps_end, cmd_args.eps_start, cmd_args.eps_decay, encoder, device, 10000, cmd_args.batch_size, cmd_args.lr,
                  QModel=Generic_MLP)
    trainer = RLTrainer(cmd_args.gamma, cmd_args.tau, env, agent, device, test_loader)
    trainer.train_loop(cmd_args.updates)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")
    subparsers = parser.add_subparsers(dest='command')

    process_data_args = subparsers.add_parser('process_data', help ='command for processing data')
    process_data_args = config_data_processing_cmd(process_data_args)

    train_transfer_learn_clf_args = subparsers.add_parser('train_transfer_learn_clf', help ='command to train clf')
    train_transfer_learn_clf_args = config_resnet_train_cmd(train_transfer_learn_clf_args)

    train_resnet_args = subparsers.add_parser('train_resnet_clf', help ='train resnet clf')
    train_resnet_args = config_resnet_train_cmd(train_resnet_args)

    train_transfer_learn_ae_args = subparsers.add_parser('train_transfer_learn_ae', help='train transfer learning autoencoder')
    train_transfer_learn_ae_args = config_transfer_learn_ae(train_transfer_learn_ae_args)

    train_diffusion_args = subparsers.add_parser('train_diffusion', help='command for training diffusion models')
    train_diffusion_args = config_diffusion_train_cmd(train_diffusion_args)

    generate_imgs_args = subparsers.add_parser('generate_imgs', help='command for creating artificial positive samples')
    generate_imgs_args = config_diffusion_generate_cmd(generate_imgs_args)

    generate_splits_args = subparsers.add_parser('generate_splits', help ='generating the splits to use for resnet and diffusion')
    generate_splits_args = config_split_data_cmd(generate_splits_args)

    generate_CIFAR_splits_args = subparsers.add_parser('generate_CIFAR_splits', help='make cifar splits for train/test')
    generate_CIFAR_splits_args = config_split_data_CIFAR_cmd(generate_CIFAR_splits_args)

    train_rl_args = subparsers.add_parser('train_rl', help ='train rl policy net')
    train_rl_args = config_rl_train_cmd(train_rl_args)

    args = parser.parse_args()

    if args.command == 'process_data':
        process_data(args)

    if args.command == 'train_resnet_clf':
        train_resnet_clf(args)

    elif args.command == 'train_transfer_learn_clf':
        train_transfer_learn_clf(args)

    elif args.command == 'train_diffusion':
        train_diffusion(args)

    elif args.command == 'train_transfer_learn_ae':
        train_transfer_learn_ae(args)

    elif args.command == 'generate_imgs':
        generate_imgs(args)

    elif args.command == 'generate_splits':
        generate_splits(args)

    elif args.command == 'generate_CIFAR_splits':
        make_cifar_splits(args)

    elif args.command == 'train_rl':
        train_rl(args)


