import torch.nn as nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Generic_MLP(nn.Module):

    def __init__(self, encoder):
        super(Generic_MLP, self).__init__()
        self.encoder = encoder


        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 2)
        self.mp = nn.Sequential(self.l1, nn.ReLU(), self.l2, nn.ReLU(), self.l3)
        self.final_act = nn.ReLU()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        z = self.encoder(x, None)
        out = self.mp(z)
        return self.final_act(out)


class PL_MLP_clf(pl.LightningModule):
    def __init__(self, mlp : Generic_MLP, criterion, lr):
        super().__init__()
        self.lr = lr
        self.model = mlp
        self.softmax = nn.Softmax()
        self.epoch_val = 0
        self.criterion = criterion
        self.train_pred = []
        self.train_actual = []
        self.val_pred = []
        self.val_actual = []

    def forward(self, x):
        return self.softmax(self.model(x))

    def training_step(self, batch, batch_idx):
        orig, jigsaw, y = batch
        logits = self.forward(jigsaw)
        loss = self.criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        # appending pred and actual values to field for access at epoch end
        self.train_pred += [pred[i].item() for i in range(len(pred))]
        self.train_actual += [y[i].item() for i in range(len(pred))]
        return loss

    def validation_step(self, batch,  batch_idx):
        orig, jigsaw, y = batch
        z = self.model.encoder(jigsaw, None)
        out = self.model.mp(z)

        logits = self.softmax(out)
        loss = self.criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        self.val_pred += [pred[i].item() for i in range(len(pred))]
        self.val_actual += [y[i].item() for i in range(len(pred))]
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.mp.parameters(), lr=self.lr, eps=.01)
        return optimizer

    def on_train_epoch_end(self) -> None:
        tb = self.logger.experiment
        res_dct = get_metrics(self.train_actual, self.train_pred)
        for k in res_dct.keys():
            tb.add_scalar(f'train/{k}', res_dct[k], self.epoch_val)
        self.epoch += 1
        return

    def on_validation_epoch_end(self) -> None:
        tb = self.logger.experiment
        res_dct = get_metrics(self.val_actual, self.val_pred)
        for k in res_dct.keys():
            tb.add_scalar(f'val/{k}', res_dct[k], self.epoch_val)
        return


def get_metrics(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    roc = roc_auc_score(true, pred)
    return {'acc' : acc, 'f1' : f1, 'roc_auc' : roc}
