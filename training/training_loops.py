import pytorch_lightning as pl
from .diffusion_trainer_custom import DiffusionTrainer


def generic_training_loop(args, pl_model, train_loader, val_loader):
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(pl_model, train_loader, val_loader)
    return


def diffusion_training_loop(model, train_loader, results_folder):
    trainer = DiffusionTrainer(model, train_loader, results_folder=results_folder)
    trainer.train()
    return
