import torch.nn as nn
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DynamicSamplingTrainer:
    def __init__(self, model: nn.Module,
                 device: str,
                 tag: str,
                 train_loader,
                 test_loader,
                 log_dir,
                 lr: float):

        self.tag = tag
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_dir = log_dir
        self.optimizer = Adam(self.model.parameters(),
                                  lr=lr)

        self.criterion = nn.CrossEntropyLoss()
        self.train_pred = []
        self.train_actual = []
        self.val_pred = []
        self.val_actual = []
        self.epoch_val = 0
        self.logger = SummaryWriter(log_dir=f'{self.log_dir}/tag')

    def train_step(self, data: torch.Tensor):
        """
        :param data: torch.Tensor of [batch_size x channel x height x width]
        :return: dictionary where key, value pairs are metrics to be stored for batch step
        """
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion

        optimizer.zero_grad()
        outputs = self.model(inputs)
        pred = torch.argmax(outputs, dim=1)

        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()
        self.train_actual += [labels[j].item() for j in range(len(labels))]
        self.train_pred += [pred[j].item() for j in range(len(labels))]

        return

    def val_loop(self):
        """

        :return: dictionary where key, value pairs are metrics for validation step to be stored
        """
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                total_samples += len(inputs)
                self.model.eval()
                criterion = self.criterion

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # get accuracy values
                pred = torch.argmax(outputs, dim=1)

                self.val_actual += [labels[j].item() for j in range(len(labels))]
                self.val_pred += [pred[j].item() for j in range(len(labels))]
                # update statistics
                batch_loss = loss.item()
                total_loss += batch_loss
        self.on_val_epoch_end()
        return

    def training_loop(self, epochs: int):
        """
        runs a training loop for given epochs
        :param epochs: number of epochs to train self.classifier
        :return:
        """
        for epoch in range(epochs):  # loop over the dataset multiple times
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader)) as datastream:
                for i, data in datastream:
                    self.train_step(data)
                    datastream.set_description(
                        f"Epoch {epoch + 1} / {epochs} | Iteration {i + 1} / {len(self.train_loader)}")
            self.on_train_epoch_end()
            self.val_loop()
            self.epoch_val += 1

        return

    def on_train_epoch_end(self):
        print(self.train_actual, self.train_pred)
        metric_dct = get_metrics(self.train_actual, self.train_pred)
        for k in metric_dct.keys():
            self.logger.add_scalar(f'train/{k}', metric_dct[k], self.epoch_val)
        return

    def on_val_epoch_end(self):
        metric_dct = get_metrics(self.val_actual, self.val_pred)
        for k in metric_dct.keys():
            self.logger.add_scalar(f'val/{k}', metric_dct[k], self.epoch_val)
        return


# because I want to extend the resampling to be based on k-means as well, the class_map is needed
# just a list that has what the classes for the i-th sample is
def get_class_f1_scores(true, pred, class_map):
    f1_dct = {}
    for k in class_map.keys():
        tmp_pred = []
        tmp_actual = []
        for idx in class_map[k]:
            tmp_pred.append(pred[idx])
            tmp_actual.append(true[idx])
        f1_dct[k] = f1_score(tmp_actual, tmp_pred)
    return f1_dct


def get_metrics(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    roc = roc_auc_score(true, pred)
    return {'acc' : acc, 'f1' : f1, 'roc_auc' : roc}