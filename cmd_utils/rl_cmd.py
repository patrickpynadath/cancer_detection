from argparse import ArgumentParser



def config_rl_train_cmd(args : ArgumentParser):
    args.add_argument("--device", type=str, help='device to run training on',default='cpu')
    args.add_argument('--input_height', default=128, type=int, help='input img height')
    args.add_argument('--input_width', default=64, type=int, help='input img width')
    args.add_argument('--batch_size', help='batch size to use for dataloader', default=64, type=int)
    args.add_argument('--base_dir', help='base dir for data', default='data', type=str)
    args.add_argument('--eps_start', help='epsilon starting val', default=.95, type = float)
    args.add_argument('--eps_end', help='epsilon ending val', default=.05, type=float)
    args.add_argument('--eps_decay', help='decay val for eps', default = 10000)
    args.add_argument('--gamma', help='gamma val', default = .99)
    args.add_argument('--tau', help='tau val', default=.005)
    args.add_argument('--lr', help='lr for training', default=.001)
    args.add_argument('--mem_capacity', help='capacity for agent mem', type=int, default=10000)
    args.add_argument('--updates', help='num gradient updates for model', default=160000)
    args.add_argument('--training_mode', help='training mode', default='normal', type=str)
    return args


