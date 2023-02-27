import torch.nn as nn
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


class DynamicSamplingTrainer:
    def __init__(self, model: nn.Module,
                 device: str,
                 tag: str,
                 train_loader,
                 test_loader,
                 log_dir,
                 use_encoder_params,
                 criterion,
                 lr: float):

        self.tag = tag
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_dir = log_dir
        self.use_encoder_params = use_encoder_params
        self.lr = lr

        self.optimizer = self.configure_optimizer()

        self.criterion = criterion
        self.train_pred = []
        self.train_actual = []
        self.val_pred = []
        self.val_actual = []
        self.epoch_val = 0
        self.use_true_labels = not self.train_loader.dataset.use_kmeans
        self.logger = SummaryWriter(log_dir=f'{self.log_dir}/{tag}')

    def train_step(self, data: torch.Tensor):
        """
        :param data: torch.Tensor of [batch_size x channel x height x width]
        :return: dictionary where key, value pairs are metrics to be stored for batch step
        """
        orig, jigsaw, labels = data
        orig, jigsaw, labels = orig.to(self.device), jigsaw.to(self.device), labels.to(self.device)

        self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion

        optimizer.zero_grad()
        outputs = self.model(jigsaw)
        pred = torch.argmax(outputs, dim=1)

        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        self.train_actual += [labels[j].item() for j in range(len(labels))]
        self.train_pred += [pred[j].item() for j in range(len(labels))]

        return batch_loss.item()

    def val_loop(self):
        with torch.no_grad():
            loss = 0
            pg = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc='val loop running')
            for i, data in pg:
                orig, jigsaw, labels = data
                orig, jigsaw, labels = orig.to(self.device), jigsaw.to(self.device), labels.to(self.device)
                self.model.eval()
                criterion = self.criterion

                outputs = self.model(jigsaw)
                loss += criterion(outputs, labels).item()

                # get accuracy values
                pred = torch.argmax(outputs, dim=1)

                self.val_actual += [labels[j].item() for j in range(len(labels))]
                self.val_pred += [pred[j].item() for j in range(len(labels))]
            self.logger.add_scalar('val/loss', loss/len(self.test_loader), self.epoch_val)
        self.on_val_epoch_end()
        return

    def training_loop(self, epochs: int):
        """
        runs a training loop for given epochs
        :param epochs: number of epochs to train self.classifier
        :return:
        """
        for epoch in range(epochs):  # loop over the dataset multiple times
            loss = 0
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader)) as datastream:
                for i, data in datastream:
                    loss += self.train_step(data)
                    datastream.set_description(
                        f"Epoch {epoch + 1} / {epochs} | Iteration {i + 1} / {len(self.train_loader)}")
            loss /= len(self.train_loader)
            self.logger.add_scalar('train/loss', loss, self.epoch_val)
            self.on_train_epoch_end()
            self.val_loop()
            for k in self.train_loader.dataset.class_ratios.keys():
                self.logger.add_scalar(f'class_{k}_sample_ratio', self.train_loader.dataset.class_ratios[k], self.epoch_val)
            self.epoch_val += 1

        return

    def on_train_epoch_end(self):
        metric_dct = get_metrics(self.train_actual, self.train_pred)
        for k in metric_dct.keys():
            self.logger.add_scalar(f'train/{k}', metric_dct[k], self.epoch_val)
        self.logger.add_scalar('train/num_pos_pred', sum(self.train_pred), self.epoch_val)
        f1_scores = get_class_f1_scores(self.train_actual, self.train_pred,
                                        self.train_loader.dataset.idx_class_map, self.use_true_labels)
        self.train_loader.dataset.adjust_sample_size(f1_scores)
        self.train_pred = []
        self.train_actual = []
        return

    def on_val_epoch_end(self):
        metric_dct = get_metrics(self.val_actual, self.val_pred)
        for k in metric_dct.keys():
            self.logger.add_scalar(f'val/{k}', metric_dct[k], self.epoch_val)
        self.logger.add_scalar('val/num_pos_pred', sum(self.val_pred), self.epoch_val)
        self.val_pred = []
        self.val_actual = []
        return

    def configure_optimizer(self):
        if self.use_encoder_params:
            optimizer = Adam(self.model.parameters(),
                              lr=self.lr)
        else:
            optimizer = Adam(self.model.mp.parameters(),
                             lr=self.lr)
        return optimizer

# because I want to extend the resampling to be based on k-means as well, the class_map is needed
# just a list that has what the classes for the i-th sample is
def get_class_f1_scores(true, pred, idx_class_map, use_true_classes=True):
    if use_true_classes:
        scores = f1_score(true, pred, average=None)
        return scores
    else:
        num_classes = len(np.unique(idx_class_map))
        f1_dct = [0 for _ in range(num_classes)]
        for k in range(num_classes):
            tmp_pred = []
            tmp_actual = []
            for idx, class_val in enumerate(list(idx_class_map)):
                if class_val == k:
                    tmp_pred.append(pred[idx])
                    tmp_actual.append(true[idx])
            if tmp_pred:
                score = f1_score(tmp_actual, tmp_pred)
                f1_dct[k] = score
            else:
                f1_dct[k] = 0
        return f1_dct



def get_metrics(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    roc = roc_auc_score(true, pred)
    return {'acc' : acc, 'f1' : f1, 'roc_auc' : roc}