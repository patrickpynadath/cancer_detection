import pytorch_lightning as pl


def resnet_training_loop(args, pl_model, train_loader, val_loader):
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(pl_model, train_loader, val_loader)
    return




