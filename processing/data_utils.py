import math
import os

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
import random
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pickle
from torchvision.transforms import Pad
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os

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
# TODO: make so that it samples from the file paths so I can include the directory to the artificial data as well
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


class ImgloaderDataSet(Dataset):
    def __init__(self, paths, values):
        self.paths = paths
        self.values = values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        xray = Image.open(path)
        img_array = np.array(xray)
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        return torch.tensor(img_array / 255, dtype=torch.float)[None, :], torch.tensor(self.values[i], dtype=torch.long)


def split_data(test_ratio, base_dir):
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


class AugmentedImgDataset(ImgloaderDataSet):
    def __init__(self, paths, values,
                 small_entropy_rad=2,
                 big_entropy_rad=5,
                 pad_val=4):
        super().__init__(paths, values)
        self.small_entropy_rad = small_entropy_rad
        self.big_entropy_rad = big_entropy_rad

    def __getitem__(self, i):
        path = self.paths[i]
        xray = Image.open(path)
        img_array = np.array(xray)
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        entropy_big = torch.tensor(get_img_entropy(img_array, self.big_entropy_rad), dtype=torch.float)
        entropy_small = torch.tensor(get_img_entropy(img_array, self.small_entropy_rad), dtype=torch.float)
        img_array = torch.tensor(img_array, dtype=torch.float) / 255
        x_grad, y_grad = get_img_gradient(img_array)
        final_img = torch.stack((img_array, x_grad, y_grad, entropy_big, entropy_small), dim=0)

        return final_img, torch.tensor(self.values[i], dtype=torch.long)


class TransferLearningDataset(AugmentedImgDataset):
    def __init__(self, paths, values : pd.DataFrame,
                 tile_length,
                 input_size,
                 learning_mode = 'normal'):
        super().__init__(paths, values)
        self.tile_length = tile_length
        self.tiling = (math.ceil(input_size[0] / tile_length), math.ceil(input_size[1] / tile_length))
        self.pad_dim = (self.tiling[0] * self.tile_length - input_size[0],
                        self.tiling[1] * self.tile_length - input_size[1])
        self.pad = Pad(padding=self.pad_dim)
        self.learning_mode = learning_mode
        self.input_size = input_size

    def _make_jigsaw(self, img: torch.Tensor):
        img = self.pad(img)
        x_indices = [i for i in range(self.tiling[0])]
        y_indices = [i for i in range(self.tiling[1])]
        random.shuffle(x_indices)
        random.shuffle(y_indices)
        jigsaw_img = torch.zeros_like(img)
        for out_x_idx, orig_x_idx in enumerate(x_indices):
            for out_y_idx, orig_y_idx in enumerate(y_indices):
                jigsaw_img[:,
                out_x_idx * self.tile_length: (out_x_idx + 1) * self.tile_length,
                out_y_idx * self.tile_length: (out_y_idx + 1) * self.tile_length] \
                    = self._get_tile(img, orig_x_idx, orig_y_idx)
        return jigsaw_img

    def _make_fillin(self, img: torch.Tensor):
        img = self.pad(img.clone())
        x_indices = [i for i in range(self.tiling[0])]
        y_indices = [i for i in range(self.tiling[1])]
        to_omit = torch.randint(low=1, high=self.tiling[0] * self.tiling[1] // 4)

        to_omit_x= random.sample(x_indices, k=to_omit)
        to_omit_y= random.sample(y_indices, k=to_omit)
        for out_x_idx, orig_x_idx in enumerate(to_omit_x):
            for out_y_idx, orig_y_idx in enumerate(to_omit_y):
                img[:,
                out_x_idx * self.tile_length: (out_x_idx + 1) * self.tile_length,
                out_y_idx * self.tile_length: (out_y_idx + 1) * self.tile_length] \
                    = torch.zeros(size=(img.size(0), self.tile_length, self.tile_length))
        return img

    def _get_tile(self, img, tile_x_idx, tile_y_idx):
        return img[:, tile_x_idx * self.tile_length: (tile_x_idx + 1) * self.tile_length,
               tile_y_idx * self.tile_length: (tile_y_idx + 1) * self.tile_length]

    def __getitem__(self, i):
        path = self.paths[i]
        xray = Image.open(path)
        img_array = np.array(xray)
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        entropy_big = torch.tensor(get_img_entropy(img_array, self.big_entropy_rad), dtype=torch.float)
        entropy_small = torch.tensor(get_img_entropy(img_array, self.small_entropy_rad), dtype=torch.float)
        img_array = torch.tensor(img_array, dtype=torch.float) / 255
        x_grad, y_grad = get_img_gradient(img_array)
        final_img = torch.stack((img_array, x_grad, y_grad, entropy_big, entropy_small), dim=0)
        if self.learning_mode == 'jigsaw':
            input_img = self._make_jigsaw(final_img)
        elif self.learning_mode == 'fillin':
            input_img = self._make_fillin(final_img)
        else:
            input_img = final_img.clone()
        # also get the labels
        return final_img, input_img, torch.tensor(self.values.iloc[i].to_numpy(), dtype=torch.float)


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


def get_clf_dataloaders(base_dir, pos_size, batch_size, synthetic_dir=None, grad_data=False):
    split_dct = get_stored_splits(base_dir)
    total_df = pd.read_csv(f'{base_dir}/train.csv')
    total_df.index = total_df['image_id']

    # getting the synthetic paths
    pos_train_paths = []
    if synthetic_dir:
        for i in range(pos_size):
            pos_train_paths.append(f'{synthetic_dir}/img{i}.png')
    else:
        # how many times to concat list
        pos_train_imgids = list(split_dct['train'][1])
        n = 1 + pos_size // len(pos_train_imgids)
        pos_train_paths = get_img_paths(pos_train_imgids, total_df, base_dir) * n
        pos_train_paths = pos_train_paths[:pos_size]

    num_pos_test = len(split_dct['test'][1])
    neg_test_imgids = random.sample(list(split_dct['test'][0]), num_pos_test)
    neg_train_imgids = random.sample(list(split_dct['train'][0]), pos_size)

    neg_test_paths = get_img_paths(neg_test_imgids, total_df, base_dir)
    pos_test_paths = get_img_paths(list(split_dct['test'][1]), total_df, base_dir)
    neg_train_paths = get_img_paths(neg_train_imgids, total_df, base_dir)

    if grad_data:
        train_set = AugmentedImgDataset(pos_train_paths + neg_train_paths,
                                        [1 for _ in pos_train_paths] + [0 for _ in neg_train_paths])
        test_set = AugmentedImgDataset(pos_test_paths + neg_test_paths,
                                       [1 for _ in pos_test_paths] + [0 for _ in neg_test_paths])
    else:
        train_set = ImgloaderDataSet(pos_train_paths + neg_train_paths,
                                     [1 for _ in pos_train_paths] + [0 for _ in neg_train_paths])
        test_set = ImgloaderDataSet(pos_test_paths + neg_test_paths,
                                    [1 for _ in pos_test_paths] + [0 for _ in neg_test_paths])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_ae_loaders(base_dir='data',tile_length=16, input_size=(128, 64), batch_size=32):
    split_dct = get_stored_splits(base_dir)
    total_df = pd.read_csv(f'{base_dir}/train.csv')
    total_df.index = total_df['image_id']

    train_img_ids = split_dct['train'][0] + split_dct['train'][1]
    test_img_ids = split_dct['test'][0]  + split_dct['test'][1]

    train_values = get_qual_values(total_df, train_img_ids)
    test_values = get_qual_values(total_df, test_img_ids)

    train_paths = get_img_paths(train_img_ids, total_df, base_dir)
    test_paths = get_img_paths(test_img_ids, total_df, base_dir)

    train_set = TransferLearningDataset(train_paths, train_values, tile_length, input_size)
    test_set = TransferLearningDataset(test_paths, test_values, tile_length, input_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


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


# given a tensor representing an image, return both the x-gradient and y-gradient
def get_img_gradient(img):
    pad_img = Pad(padding=1)(img)
    x_grad = pad_img[2:, 1:img.size(1) + 1] - img
    y_grad = pad_img[1:img.size(0) + 1:, 2:] - img
    return x_grad, y_grad


# img needs to be in NUMPY format
def get_img_entropy(img, rad):
    return entropy(img, disk(rad))
