from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
import random
from sklearn.model_selection import ShuffleSplit
import numpy as np

random.seed(1)

def get_paths(base_dir='data', train=True, target_col=None, target_val = None):
    if train:
        root_dir = f'{base_dir}/train_images'
    else:
        root_dir = f'{base_dir}/test_images'
    train_csv = pd.read_csv(f'{base_dir}/train.csv')
    train_csv.index = train_csv['image_id']
    paths = []
    if not target_val and not target_col:
        for i, image_id in enumerate(train_csv.index):
            patient_id = train_csv.loc[image_id]['patient_id']
            paths.append(f'{root_dir}/{patient_id}/{image_id}.dcm')
    else:
        for i, image_id in enumerate(train_csv.index):
            if train_csv.loc[image_id][target_col] == target_val:
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


# TODO: need to explicitly control the size of the training set
def get_loaders_from_args(args, to_mimic=None):
    base_dir = args.base_dir
    train_csv = pd.read_csv(f'{base_dir}/train.csv')
    if to_mimic:
        tmp = train_csv
        for col_name, val in to_mimic:
            tmp = tmp[tmp[col_name].isin(val)]
        total_ids = tmp['image_id']
    else:
        total_ids = train_csv['image_id']
    rs = ShuffleSplit(n_splits=1, test_size=args.test_size)
    total_idx, test_idx = next(rs.split(total_ids))
    train_idx, val_idx = next(rs.split(total_ids))
    target_col = args.target_col
    train_set = XRayDataset(base_dir, list(total_ids.iloc[train_idx]), target_col)
    val_set = XRayDataset(base_dir, list(total_ids.iloc[val_idx]), target_col)
    test_set = XRayDataset(base_dir, list(total_ids.iloc[test_idx]), target_col)
    batch_size = args.batch_size
    return DataLoader(train_set, batch_size=batch_size, num_workers=args.loader_workers), \
           DataLoader(val_set, batch_size=batch_size, num_workers=args.loader_workers), \
           DataLoader(test_set, batch_size=batch_size, num_workers=args.loader_workers)


def get_balanced_loaders(args, train_size, val_size, test_size):
    base_dir = args.base_dir
    train_csv = pd.read_csv(f'{base_dir}/train.csv')
    total_ids = train_csv['image_id']
    index_pos = list(train_csv[train_csv['cancer'].isin([1])].index)
    index_neg = list(train_csv[train_csv['cancer'].isin([0])].index)
    sampled_pos = random.sample(index_pos, (train_size + val_size + test_size)/2)
    sampled_neg = random.sample(index_neg, (train_size + val_size + test_size)/2)

    sampled_pos_train = sampled_pos[:train_size/2]
    sampled_neg_train = sampled_neg[:train_size/2]

    sampled_pos_val = sampled_pos[train_size/2 : (train_size + val_size)/2]
    sampled_neg_val = sampled_neg[train_size / 2: (train_size + val_size) / 2]

    sampled_pos_test = sampled_pos[(train_size + val_size)/2 :]
    sampled_neg_test = sampled_neg[(train_size + val_size)/2 :]
    batch_size = args.batch_size
    train_idx = sampled_pos_train + sampled_neg_train
    val_idx = sampled_pos_val + sampled_neg_val
    test_idx = sampled_pos_test + sampled_neg_test

    train_set = XRayDataset(base_dir, list(total_ids.iloc[train_idx]), 'cancer')
    val_set = XRayDataset(base_dir, list(total_ids.iloc[val_idx]), 'cancer')
    test_set = XRayDataset(base_dir, list(total_ids.iloc[test_idx]), 'cancer')
    return DataLoader(train_set, batch_size=batch_size, num_workers=args.loader_workers), \
           DataLoader(val_set, batch_size=batch_size, num_workers=args.loader_workers), \
           DataLoader(test_set, batch_size=batch_size, num_workers=args.loader_workers)


def get_num_classes(target_col, base_dir):
    return len(pd.unique(pd.read_csv(f'{base_dir}/train.csv')[target_col]))
