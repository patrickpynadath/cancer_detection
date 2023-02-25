import math
import random
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange
from sklearn.decomposition import IncrementalPCA
from PIL import Image
from skimage.filters.rank import entropy
from skimage.morphology import disk
from torch.utils.data import Dataset
from torchvision.transforms import Pad
from sklearn.preprocessing import normalize
import hdbscan


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
    def __init__(self, paths, values,
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
        # entropy_big = torch.tensor(get_img_entropy(img_array, self.big_entropy_rad), dtype=torch.float)
        # entropy_small = torch.tensor(get_img_entropy(img_array, self.small_entropy_rad), dtype=torch.float)
        # img_array = torch.tensor(img_array, dtype=torch.float) / 255
        # x_grad, y_grad = get_img_gradient(img_array)
        # final_img = torch.stack((img_array, x_grad, y_grad, entropy_big, entropy_small), dim=0)
        final_img = torch.tensor(img_array, dtype=torch.float32)[None, :] / 255
        if self.learning_mode == 'jigsaw':
            input_img = self._make_jigsaw(final_img)
        elif self.learning_mode == 'fillin':
            input_img = self._make_fillin(final_img)
        else:
            input_img = final_img.clone()
        # also get the labels
        return final_img, input_img, torch.tensor(self.values[i], dtype=torch.long)


class DynamicDataset(TransferLearningDataset):
    def __init__(self, paths, values,
                 tile_length,
                 input_size,
                 learning_mode = 'normal',
                 use_kmeans = False,
                 kmeans_clusters=8,
                 encoder = None,
                 device='cpu'):
        super().__init__(paths,
                         values, tile_length,
                         input_size,
                         learning_mode)

        self.orig_paths = paths
        self.orig_values = values
        self.class_map = get_label_idx_dct(values)  # dct with class_idx : sample_idx
        if use_kmeans:
            if encoder:
                print("training kmeans")
                self.class_map = self._get_kmeans_class_dct(encoder, kmeans_clusters, device)
            else:
                raise Exception
        class_ratios = {}
        for k in self.class_map.keys():
            class_ratios[k] = .5
        self.use_kmeans= use_kmeans
        self.class_ratios = class_ratios

    def adjust_sample_size(self, f1_class_scores):
        denom = 0
        for i, score in enumerate(f1_class_scores):
            denom += 1 - score
        avg_size = len(self.orig_values) / len(self.class_map.keys())
        sample_idx = []
        print(f'initial size: {len(self.paths)}')
        for k in self.class_map.keys():
            print(f"adjusting class size: {k}")
            ratio = (1-f1_class_scores[k])/denom
            num_to_sample = int(avg_size * ratio)
            print(f"ratio: {ratio}")
            print(f"num to sample: {num_to_sample}")
            print(f"orig: {len(self.class_map[k])}")
            multiple = int(1 + num_to_sample / len(self.class_map[k]))
            to_append = self.class_map[k] * multiple
            to_append = to_append[:num_to_sample]
            print(len(to_append))
            sample_idx += to_append
        self.paths = [self.orig_paths[idx] for idx in sample_idx]
        self.values = [self.orig_values[idx] for idx in sample_idx]
        print(f"new size: {len(self.paths)}")
        return

    def _get_encoder_lv(self, encoder, device='cpu'):
        to_stack = []
        print('getting encoded lv')
        with torch.no_grad():
            pg = trange(len(self.paths))
            for i in pg:
                sample = self.__getitem__(i)[1][None, :].to(device)
                lv = encoder(sample, None)[0, :].cpu().numpy()
                to_stack.append(lv)
        return np.stack(to_stack, axis=0)

    def _get_kmeans_class_dct(self, encoder, num_clusters, device):
        X = self._get_encoder_lv(encoder, device)
        X_normalized = normalize(X)
        pca = IncrementalPCA(n_components=100)
        print("fitting pca")
        pca.fit(X_normalized)
        X_reduced = pca.transform(X)
        print(X_reduced)
        # print("fitting kmeans")
        # kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=1024, verbose=1)
        # kmeans.fit(X_reduced)
        clutering_alg = hdbscan.HDBSCAN(min_samples=50)
        clutering_alg.fit(X_reduced)

        #pred = kmeans.predict(X_reduced)
        pred = clutering_alg.labels_
        num_clusters = len(np.unique(pred))
        class_map = {}
        for i in range(num_clusters):
            class_map[i] = []
        for idx, cluster_idx in enumerate(list(pred)):
            class_map[cluster_idx].append(idx)
        return class_map

    def _get_label_dct(self):
        class_map = {0 : [], 1 : []}
        for idx, label in enumerate(self.values):
            class_map[int(label)].append(idx)
        return class_map


def get_label_idx_dct(labels):
    class_map = {0 : [], 1 : []}
    for i, val in enumerate(labels):
        class_map[int(val)].append(i)
    return class_map


def get_img_gradient(img):
    pad_img = Pad(padding=1)(img)
    x_grad = pad_img[2:, 1:img.size(1) + 1] - img
    y_grad = pad_img[1:img.size(0) + 1:, 2:] - img
    return x_grad, y_grad


def get_img_entropy(img, rad):
    return entropy(img, disk(rad))
