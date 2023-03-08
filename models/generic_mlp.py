import torch.nn as nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Generic_MLP(nn.Module):

    def s__init__(self, encoder, fc_latent):
        super(Generic_MLP, self).__init__()
        self.encoder = encoder
        self.fc_latent = fc_latent


        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 2)
        self.mp = nn.Sequential(self.l1, nn.ReLU(), self.l2, nn.ReLU(), self.l3)
        self.final_act = nn.Softmax(dim=1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        z = self.encoder(x)
        z = z.flatten(start_dim=1)
        z = self.fc_latent(z)
        out = self.mp(z)
        return self.final_act(out)


class PL_MLP_clf(pl.LightningModule):
    def __init__(self, mlp : Generic_MLP, criterion, lr, use_encoder_params):
        super().__init__()
        self.lr = lr
        self.model = mlp
        self.epoch_val = 0
        self.criterion = criterion
        self.train_pred = []
        self.train_actual = []
        self.val_pred = []
        self.val_actual = []
        self.use_encoder_params = use_encoder_params

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        orig, jigsaw, y = batch
        logits = self.forward(jigsaw)
        loss = self.criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        # appending pred and actual values to field for access at epoch end
        self.train_pred += [pred[i].item() for i in range(len(pred))]
        self.train_actual += [y[i].item() for i in range(len(pred))]
        self.log('train/loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        orig, jigsaw, y = batch
        logits = self.forward(jigsaw)
        loss = self.criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        self.val_pred += [pred[i].item() for i in range(len(pred))]
        self.val_actual += [y[i].item() for i in range(len(pred))]
        tb = self.logger.experiment
        tb.add_scalar(f'val/loss', loss.item(), self.epoch_val)
        return loss

    def configure_optimizers(self):
        if self.use_encoder_params:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=.01)
        else:
            optimizer = torch.optim.Adam(self.model.mp.parameters(), lr=self.lr, eps=.01)
        return optimizer

    def on_train_epoch_end(self) -> None:
        tb = self.logger.experiment
        res_dct = get_metrics(self.train_actual, self.train_pred)
        for k in res_dct.keys():
            tb.add_scalar(f'train/{k}', res_dct[k], self.epoch_val)
        self.epoch_val += 1
        self.train_pred = []
        self.train_actual = []
        return

    def on_validation_epoch_end(self) -> None:
        tb = self.logger.experiment
        res_dct = get_metrics(self.val_actual, self.val_pred)
        for k in res_dct.keys():
            tb.add_scalar(f'val/{k}', res_dct[k], self.epoch_val)
        self.val_pred = []
        self.val_actual = []
        return


def get_metrics(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    roc = roc_auc_score(true, pred)
    num_pos_pred = sum(pred)
    return {'acc' : acc, 'f1' : f1, 'roc_auc' : roc, 'num_pos_pred' : num_pos_pred}
