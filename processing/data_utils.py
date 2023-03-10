from torch.utils.data import DataLoader
import pandas as pd
import random
from sklearn.model_selection import ShuffleSplit
import pickle
import os
import torch
from .custom_dataset_classes import ImgloaderDataSet, TransferLearningDatasetRSNA, TransferLearningDatasetCIFAR
from .dynamic_dataset import DynamicDatasetRSNA
from .cifar_dataset import get_cifar_sets, ANIMAL_CLASS_IDX, get_values

os.environ["NCCL_DEBUG"] = "INFO"
random.seed(1)

QUAL_COL_NAMES = ['age', 'implant', 'laterality']


def get_paths(base_dir='data', train=True, target_col=None, target_val=None):
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


def split_data_RSNA(test_ratio, base_dir):
    train_csv = pd.read_csv(f'{base_dir}/train.csv')
    neg_df = train_csv[train_csv['cancer'].isin([0])]
    pos_df = train_csv[train_csv['cancer'].isin([1])]

    rs = ShuffleSplit(n_splits=1, test_size=test_ratio)

    neg_train, neg_test = next(rs.split(neg_df['image_id']))
    pos_train, pos_test = next(rs.split(pos_df['image_id']))

    with open(f'{base_dir}/neg_train_imgid.pickle', 'wb') as f:
        pickle.dump(neg_df['image_id'].iloc[neg_train], f)
    with open(f'{base_dir}/neg_test_imgid.pickle', 'wb') as f:
        pickle.dump(neg_df['image_id'].iloc[neg_test], f)
    with open(f'{base_dir}/pos_train_imgid.pickle', 'wb') as f:
        pickle.dump(pos_df['image_id'].iloc[pos_train], f)
    with open(f'{base_dir}/pos_test_imgid.pickle', 'wb') as f:
        pickle.dump(pos_df['image_id'].iloc[pos_test], f)
    return


def split_data_CIFAR(minority_sample_ratio, base_dir, dataset, dataset_label):
    pos_idx = []
    neg_idx = []
    for i in range(len(dataset)):
        if dataset[i][1] in ANIMAL_CLASS_IDX:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    # sample from pos_train, pos_test after calculating the number to sample
    num_pos = int(minority_sample_ratio * len(pos_idx))

    pos_train = random.sample(pos_idx, num_pos)
    print(f'imbalance ratio for {dataset_label}: {num_pos/len(dataset)}')
    base_dir += f'/{minority_sample_ratio}'
    if not os.path.exists(f'{base_dir}'):
        os.mkdir(f'{base_dir}')

    with open(f'{base_dir}/neg_{dataset_label}_imgid.pickle', 'wb') as f:
        pickle.dump(neg_idx, f)
    with open(f'{base_dir}/pos_{dataset_label}_imgid.pickle', 'wb') as f:
        pickle.dump(pos_train, f)
    return


def get_stored_splits(base_dir):
    with open(f'{base_dir}/neg_train_imgid.pickle', 'rb') as f:
        neg_train = pickle.load(f)
    with open(f'{base_dir}/neg_test_imgid.pickle', 'rb') as f:
        neg_test = pickle.load(f)
    with open(f'{base_dir}/pos_train_imgid.pickle', 'rb') as f:
        pos_train = pickle.load(f)
    with open(f'{base_dir}/pos_test_imgid.pickle', 'rb') as f:
        pos_test = pickle.load(f)
    return {'train': (neg_train, pos_train),
            'test': (neg_test, pos_test)}



def get_diffusion_dataloaders(base_dir, batch_size):
    split_dct = get_stored_splits(base_dir)
    total_df = pd.read_csv(f'{base_dir}/train.csv')
    total_df.index = total_df['image_id']
    train_paths = get_img_paths(split_dct['train'][1], total_df, base_dir)
    test_paths = get_img_paths(split_dct['test'][1], total_df, base_dir)

    train_set = ImgloaderDataSet(train_paths, values=[1 for _ in train_paths])
    test_set = ImgloaderDataSet(test_paths, values=[1 for _ in test_paths])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader


def get_clf_dataloaders(base_dir,
                        batch_size,
                        tile_length,
                        input_size,
                        label_dtype=torch.long,
                        kmeans_clusters=None,
                        device=None,
                        encoder=None,
                        learning_mode='normal',
                        sample_strat='none',
                        update_beta=.25,
                        w = 'natural'):
    split_dct = get_stored_splits(base_dir)
    total_df = pd.read_csv(f'{base_dir}/train.csv')
    total_df.index = total_df['image_id']
    if sample_strat == 'ros':
        # how many times to concat list
        pos_size = len(split_dct['train'][0])
        pos_train_imgids = list(split_dct['train'][1])
        n = 1 + pos_size// len(pos_train_imgids)
        pos_train_paths = get_img_paths(pos_train_imgids, total_df, base_dir) * n
        pos_train_paths = pos_train_paths[:pos_size]
        neg_train_imgids = list(split_dct['train'][0])
        neg_train_paths = get_img_paths(neg_train_imgids, total_df, base_dir)
    elif sample_strat == 'rus':
        pos_size = len(split_dct['train'][1])
        pos_train_imgids = list(split_dct['train'][1])
        pos_train_paths = get_img_paths(pos_train_imgids, total_df, base_dir)
        neg_train_imgids = random.sample(list(split_dct['train'][0]), pos_size)
        neg_train_paths = get_img_paths(neg_train_imgids, total_df, base_dir)
    else:
        pos_train_imgids = list(split_dct['train'][1])
        neg_train_imgids = list(split_dct['train'][0])
        pos_train_paths = get_img_paths(pos_train_imgids, total_df, base_dir)
        neg_train_paths = get_img_paths(neg_train_imgids, total_df, base_dir)

    # if 'test_ratio' != 'natural':



    neg_test_imgids = list(split_dct['test'][0])
    pos_test_imgids = list(split_dct['test'][1])

    neg_test_paths = get_img_paths(neg_test_imgids, total_df, base_dir)
    pos_test_paths = get_img_paths(pos_test_imgids, total_df, base_dir)

    train_paths = pos_train_paths + neg_train_paths
    train_values = [1 for _ in pos_train_paths] + [0 for _ in neg_train_paths]

    test_paths = pos_test_paths + neg_test_paths
    test_values = [1 for _ in pos_test_paths] + [0 for _ in neg_test_paths]

    if 'dynamic' in sample_strat:
        train_set = DynamicDatasetRSNA(train_paths, train_values, tile_length=tile_length, input_size=input_size,
                                       learning_mode=learning_mode,
                                       use_kmeans=sample_strat == 'dynamic_kmeans_ros',
                                       kmeans_clusters=kmeans_clusters,
                                       encoder=encoder, device=device, label_dtype=label_dtype,
                                       update_beta=update_beta)
    else:
        train_set = TransferLearningDatasetRSNA(train_paths, train_values,
                                                tile_length=tile_length,
                                                input_size=input_size,
                                                learning_mode=learning_mode, label_dtype=label_dtype)

    test_set = TransferLearningDatasetRSNA(test_paths, test_values,
                                           tile_length=tile_length,
                                           input_size=input_size,
                                           learning_mode=learning_mode, label_dtype=label_dtype)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_ae_loaders_RSNA(base_dir='data',
                        tile_length=16,
                        input_size=(128, 64),
                        batch_size=32,
                        learning_mode='normal'):
    split_dct = get_stored_splits(base_dir)
    total_df = pd.read_csv(f'{base_dir}/train.csv')
    total_df.index = total_df['image_id']

    train_img_ids = list(split_dct['train'][0]) + list(split_dct['train'][1])
    test_img_ids = list(split_dct['test'][0]) + list(split_dct['test'][1])

    train_labels = [0 for _ in split_dct['train'][0]] + [1 for _ in split_dct['train'][1]]
    test_labels = [0 for _ in split_dct['test'][0]] + [1 for _ in split_dct['test'][1]]

    train_values = get_qual_values(total_df, list(train_img_ids))
    test_values = get_qual_values(total_df, list(test_img_ids))

    train_paths = get_img_paths(train_img_ids, total_df, base_dir)
    test_paths = get_img_paths(test_img_ids, total_df, base_dir)

    train_set = TransferLearningDatasetRSNA(train_paths, train_labels, tile_length, input_size, learning_mode=learning_mode)
    test_set = TransferLearningDatasetRSNA(test_paths, test_labels, tile_length, input_size, learning_mode=learning_mode)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_ae_loaders_CIFAR(tile_length,
                         base_dir,
                         batch_size,
                         learning_mode,
                         minority_sample_ratio):
    train, test = get_cifar_sets()
    base_dir = base_dir + f'/{round(minority_sample_ratio, 3)}'
    stored_splits = get_stored_splits(base_dir)
    train_idx = stored_splits['train'][0] + stored_splits['train'][1]
    test_idx = stored_splits['test'][0] + stored_splits['test'][1]
    tv_train = get_values(train)
    tv_test = get_values(test)
    train_values = [tv_train[idx] for idx in train_idx]
    test_values = [tv_test[idx] for idx in test_idx]

    train_ds = TransferLearningDatasetCIFAR(train_idx, train_values, train, tile_length, (32, 32), learning_mode)
    test_ds = TransferLearningDatasetCIFAR(test_idx, test_values, test, tile_length, (32, 32), learning_mode)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    return train_dl, test_dl


def get_qual_values(df, image_ids):
    age = df.loc[image_ids]['age']
    laterality_one_hot = pd.get_dummies(df.loc[image_ids]['laterality'], prefix='laterality_')
    implant = df.loc[image_ids]['implant']
    out = pd.DataFrame()
    out.index = image_ids
    out['age'] = age
    for col_name in list(laterality_one_hot.columns):
        out[col_name] = laterality_one_hot[col_name]
    out['implant'] = implant
    return out


def get_img_paths(img_ids, train_csv, base_dir):
    paths = []
    for id in img_ids:
        patient_id = train_csv.loc[id]['patient_id']
        pth = f'{base_dir}/train_images/{patient_id}/{id}.png'
        paths.append(pth)
    return paths


def get_num_classes(target_col, base_dir):
    return len(pd.unique(pd.read_csv(f'{base_dir}/train.csv')[target_col]))


