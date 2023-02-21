from argparse import ArgumentParser


def config_diffusion_train_cmd(args: ArgumentParser):
    model_flags = args.add_argument_group('model_flags')
    data_flags = args.add_argument_group('data_flags')
    model_flags.add_argument('--img_height', default=128, type=int, help='input img height')
    model_flags.add_argument('--img_width', default=64, type=int, help='input img width')
    model_flags.add_argument('--timesteps', default=1000, type=int, help='timesteps for diffusion model')
    model_flags.add_argument('--loss_type', default='l2', choices=['l2', 'l1'],
                                       help='loss type for diffusion output')
    data_flags.add_argument('--mimic_col', default='cancer', type=str,
                                      help='target col to train diffusion model to generate')
    data_flags.add_argument('--mimic_val', default=1,
                                      help='the desired value the samples should mimic from target_col')
    data_flags.add_argument('--base_dir', default='data', type=str, help='base dir for data')
    data_flags.add_argument('--test_size', default=.1, type=float, help='test ratio size')
    data_flags.add_argument('--batch_size', default=32, type=int, help='batch size')
    data_flags.add_argument('--loader_workers', default=32, type=int, help='num workers for data loader')
    data_flags.add_argument('--target_col', default='cancer', type=str, help='target col')
    return args


def config_diffusion_generate_cmd(args: ArgumentParser):
    args.add_argument('--img_height', default=128, type=int, help='input img height')
    args.add_argument('--img_width', default=64, type=int, help='input img width')
    args.add_argument('--save_name', default='total_cancer_results/model-99.pt', type=str,
                               help='name of stored state dict for diffusion model')
    args.add_argument('--num_samples', default=2000, type=int, help='num of samples to be generated')
    args.add_argument('--batch_size', default=64, type=int, help='batch size for generating imgs')
    args.add_argument('--device', default='cpu', type=str, help='device to use')
    return args
