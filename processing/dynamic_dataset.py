import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import trange
from torch.utils.data import Dataset
from .custom_dataset_classes import TransferLearningDatasetRSNA, TransferLearningDatasetCIFAR
from processing.custom_dataset_classes import get_label_idx_dct
import copy

class DynamicDatasetRSNA(TransferLearningDatasetRSNA):
    def __init__(self, paths, values,
                 tile_length,
                 input_size,
                 label_dtype,
                 learning_mode = 'normal',
                 use_kmeans = False,
                 kmeans_clusters=8,
                 encoder = None,
                 device='cpu',
                 update_beta=.25):
        super().__init__(paths,
                         values, tile_length,
                         input_size,
                         learning_mode, label_dtype=label_dtype)

        self.orig_paths = paths
        self.orig_values = values
        self.idx_class_map = values
        self.class_idx_map = get_label_idx_dct(values)  # dct with class_idx : sample_idx
        if use_kmeans:
            if encoder:
                print("training kmeans")
                self.class_idx_map, self.idx_class_map = self._get_kmeans_class_dct(encoder, kmeans_clusters, device)
            else:
                raise Exception
        class_ratios = {}
        for k in self.class_idx_map.keys():
            class_ratios[k] = len(self.class_idx_map[k]) / len(self.orig_paths)
        self.use_kmeans= use_kmeans
        self.class_ratios = class_ratios
        self.update_beta = update_beta
        self.cur_paths_class_map = [0 for _ in self.orig_paths]


    def _update_paths_class_map(self):
        paths_class_map = []
        for k in self.class_idx_map.keys():
            for idx in self.class_idx_map[k]:
                paths_class_map[idx] = k


    # we have a current weight,
    # calculated_weight
    # need final weights to be in between current and calculated
    def adjust_sample_size(self, f1_class_scores):
        print(f"f1 scores: {f1_class_scores}")
        f1_class_scores = np.array(f1_class_scores)
        ratios = np.ones_like(f1_class_scores) - f1_class_scores
        new_ratios = np.exp(np.array(ratios))/np.sum(np.exp(np.array(ratios)))
        print(f"new sampling ratios: {new_ratios}")
        orig_ratios = [len(self.class_idx_map[k]) / len(self.orig_values) for k in list(self.class_idx_map.keys())]
        print(f"orig ratios: {orig_ratios}")
        sample_idx = []
        print(f'initial size: {len(self.paths)}')
        print(f1_class_scores)
        new_idx_class_map = []
        for k in self.class_idx_map.keys():
            print(f"adjusting class size: {k}")
            cur_ratio = self.class_ratios[k]
            ratio = new_ratios[k] * self.update_beta + cur_ratio * (1 - self.update_beta)
            num_to_sample = int(len(self.orig_values) * ratio)
            print(f"cur sampling size: {int(self.class_ratios[k] * len(self.orig_paths))}")
            print(f"new sampling size: {num_to_sample}")
            #multiple = int(1 + num_to_sample / len(self.class_map[k]))
            #to_append = self.class_map[k] * multiple
            #to_append = to_append[:num_to_sample]
            to_append = random.choices(self.class_idx_map[k], k=num_to_sample)
            print(len(to_append))
            self.class_ratios[k] = ratio
            sample_idx += to_append
            new_idx_class_map += [k for _ in range(num_to_sample)]
        self.idx_class_map = new_idx_class_map
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
                lv = encoder(sample, None)[0, :]
                #lv = lv / torch.linalg.norm(lv, ord=2)
                to_stack.append(lv.cpu().numpy())
        return np.stack(to_stack, axis=0)

    def _get_kmeans_class_dct(self, encoder, num_clusters, device):
        X = self._get_encoder_lv(encoder, device)
        X = normalize(X)
        pca = PCA(n_components=20)
        print("fitting pca")
        pca.fit(X)

        X_reduced = pca.transform(X)
        print(X_reduced)
        print("fitting kmeans")
        kmeans = KMeans(n_clusters=20, verbose=1)
        kmeans.fit(X_reduced)

        idx_to_cluster = kmeans.predict(X_reduced)
        print(np.unique(idx_to_cluster, return_counts=True))

        cluster_to_idx = {}
        #print(np.unique(pred, return_counts=True))
        for i in list(np.unique(idx_to_cluster)):
            cluster_to_idx[i] = []
        for idx, cluster_idx in enumerate(list(idx_to_cluster)):
            cluster_to_idx[cluster_idx].append(idx)
        return cluster_to_idx, idx_to_cluster

    def _get_label_dct(self):
        class_map = {0 : [], 1 : []}
        for idx, label in enumerate(self.values):
            class_map[int(label)].append(idx)
        return class_map


# instead of making something entirely new, just going to make a wrapper class since most of the heavy lifting is already done
class DynamicDatasetCIFAR(TransferLearningDatasetCIFAR):
    def __init__(self,
                 dataset,
                 tile_length,
                 input_size,
                 anomoly_class_idx,
                 learning_mode,
                 label_dtype,
                 use_kmeans,
                 update_beta,
                 encoder,
                 device,
                 kmeans_clusters=50):
        super().__init__(dataset, tile_length, input_size, anomoly_class_idx=anomoly_class_idx, learning_mode=learning_mode, label_dtype=label_dtype)
        self.orig_values = copy.copy(self.values)
        self.idx_class_map = copy.copy(self.values)
        self.class_idx_map = get_label_idx_dct(self.values)  # dct with class_idx : sample_idx
        if use_kmeans:
            if encoder:
                print("training kmeans")
                self.class_idx_map, self.idx_class_map = self._get_kmeans_class_dct(encoder, kmeans_clusters,
                                                                                    device)
            else:
                raise Exception
        class_ratios = {}
        for k in self.class_idx_map.keys():
            class_ratios[k] = len(self.class_idx_map[k]) / len(self.dataset)
        self.use_kmeans = use_kmeans
        self.class_ratios = class_ratios
        self.update_beta = update_beta
        self.new_idx_orig_idx_map = [i for i in range(len(dataset))]

    def _get_values(self):
        new_values = []
        for i in range(len(self.dataset)):
            if self.dataset[i][1] == self.anomoly_class_idx:
                new_values.append(1)
            else:
                new_values.append(0)
        return new_values

    # we have a current weight,
    # calculated_weight
    # need final weights to be in between current and calculated
    def adjust_sample_size(self, f1_class_scores):
        print(f"f1 scores: {f1_class_scores}")
        f1_class_scores = np.array(f1_class_scores)
        ratios = np.ones_like(f1_class_scores) - f1_class_scores
        new_ratios = np.exp(np.array(ratios)) / np.sum(np.exp(np.array(ratios)))
        print(f"new sampling ratios: {new_ratios}")
        orig_ratios = [len(self.class_idx_map[k]) / len(self.orig_values) for k in list(self.class_idx_map.keys())]
        print(f"orig ratios: {orig_ratios}")
        sample_idx = []
        print(f'initial size: {len(self.new_idx_orig_idx_map)}')
        print(f1_class_scores)
        new_idx_class_map = []
        for k in self.class_idx_map.keys():
            print(f"adjusting class size: {k}")
            cur_ratio = self.class_ratios[k]
            ratio = new_ratios[k] * self.update_beta + cur_ratio * (1 - self.update_beta)
            num_to_sample = int(len(self.orig_values) * ratio)
            print(f"cur sampling size: {int(self.class_ratios[k] * len(self.dataset))}")
            print(f"new sampling size: {num_to_sample}")
            # multiple = int(1 + num_to_sample / len(self.class_map[k]))
            # to_append = self.class_map[k] * multiple
            # to_append = to_append[:num_to_sample]
            to_append = random.choices(self.class_idx_map[k], k=num_to_sample)
            print(len(to_append))
            self.class_ratios[k] = ratio
            sample_idx += to_append
            new_idx_class_map += [k for _ in range(num_to_sample)]
        self.idx_class_map = new_idx_class_map
        self.new_idx_orig_idx_map = [idx for idx in sample_idx]
        self.values = [self.orig_values[idx] for idx in sample_idx]
        print(f"new size: {len(self.new_idx_orig_idx_map)}")
        return

    def __getitem__(self, i):
        orig_idx = self.new_idx_orig_idx_map[i]
        return super().__getitem__(orig_idx)

    def _get_encoder_lv(self, encoder, device='cpu'):
        to_stack = []
        print('getting encoded lv')
        with torch.no_grad():
            pg = trange(len(self.new_idx_orig_idx_map))
            for i in pg:
                sample = self.__getitem__(i)[1][None, :].to(device)
                lv = encoder(sample, None)[0, :]
                # lv = lv / torch.linalg.norm(lv, ord=2)
                to_stack.append(lv.cpu().numpy())
        return np.stack(to_stack, axis=0)

    def _get_kmeans_class_dct(self, encoder, device):
        X = self._get_encoder_lv(encoder, device)
        X = normalize(X)
        pca = PCA(n_components=20)
        print("fitting pca")
        pca.fit(X)

        X_reduced = pca.transform(X)
        print(X_reduced)
        print("fitting kmeans")
        kmeans = KMeans(n_clusters=20, verbose=1)
        kmeans.fit(X_reduced)

        idx_to_cluster = kmeans.predict(X_reduced)
        print(np.unique(idx_to_cluster, return_counts=True))

        cluster_to_idx = {}
        # print(np.unique(pred, return_counts=True))
        for i in list(np.unique(idx_to_cluster)):
            cluster_to_idx[i] = []
        for idx, cluster_idx in enumerate(list(idx_to_cluster)):
            cluster_to_idx[cluster_idx].append(idx)
        return cluster_to_idx, idx_to_cluster

    def _get_label_dct(self):
        class_map = {0: [], 1: []}
        for idx, label in enumerate(self.values):
            class_map[int(label)].append(idx)
        return class_map



