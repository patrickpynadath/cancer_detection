import pytorch_lightning as pl


# TODO: add methods for training a resnet classifier
def train_clf(args, pl_model, train_loader, val_loader):
    trainer = pl.Trainer.from_argparse_args(args)

    return




