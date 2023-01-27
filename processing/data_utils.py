import os


def get_paths(base_dir='data', train=True):
    if train:
        return [f.path for f in os.scandir(f'{base_dir}/train_images') if f.is_dir()]
    else:
        return [f.path for f in os.scandir(f'{base_dir}/test_images') if f.is_dir()]
