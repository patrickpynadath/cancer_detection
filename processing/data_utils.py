import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import ShuffleSplit
import numpy as np


def get_paths(base_dir='data', train=True):
    if train:
        root_dir = f'{base_dir}/train_images'
    else:
        root_dir = f'{base_dir}/test_images'
    train_csv = pd.read_csv(f'{base_dir}/train.csv')
    train_csv.index = train_csv['image_id']
    paths = []
    for i, image_id in enumerate(train_csv.index):
        patient_id = train_csv.loc[image_id]['patient_id']
        paths.append(f'{root_dir}/{patient_id}/{image_id}.dcm')
    return paths


# torch dataset class for mammographs
# target col can be any column included in the given csv
class XRayDataset(Dataset):
    def __init__(self, base_dir, image_ids, target_col='cancer'):

        self.base_dir = base_dir
        self.image_ids = image_ids
        train_csv = pd.read_csv(f'{base_dir}/train.csv')
        if target_col not in list(train_csv.columns):
            raise Exception("provided target column not in csv columns, check spelling")
        train_csv.index = train_csv['image_id']
        self.classes = list(pd.unique(train_csv[target_col]))
        tmp_dct = {}
        for i, val in enumerate(self.classes):
            tmp_dct[val] = i
        # target column needs to be list of class index to use optimized computation for CE loss
        self.target_col = target_col
        self.target_df = train_csv[target_col].map(lambda class_name: tmp_dct[class_name])
        self.data_df = train_csv

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        patient_id = self.data_df.loc[image_id]['patient_id']
        xray = Image.open(f'{self.base_dir}/train_images/{patient_id}/{image_id}.png')
        label = self.target_df.loc[image_id]
        return torch.tensor(np.array(xray) / 255, dtype=torch.float)[None, :], torch.tensor(label, dtype=torch.long)


def get_loaders_from_args(args):
    base_dir = args.base_dir
    train_csv = pd.read_csv(f'{base_dir}/train.csv')

    total_ids = train_csv['image_id']
    rs = ShuffleSplit(n_splits=2, test_size=args.test_size)
    total_train_ids, test_ids = next(rs.split(total_ids))
    train_ids, val_ids = next(rs.split(total_ids))
    target_col = args.target_col
    train_set = XRayDataset(base_dir, train_ids, target_col)
    val_set = XRayDataset(base_dir, val_ids, target_col)
    test_set = XRayDataset(base_dir, test_ids, target_col)
    batch_size = args.batch_size
    return DataLoader(train_set, batch_size=batch_size), \
           DataLoader(val_set, batch_size=batch_size), \
           DataLoader(test_set, batch_size=batch_size)


def get_num_classes(target_col, base_dir):
    return len(pd.unique(pd.read_csv(f'{base_dir}/train.csv')[target_col]))
