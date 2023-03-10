import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from .diffusion_trainer_custom import DiffusionTrainer
from .dynamic_trainer import DynamicSamplingTrainer


def generic_training_loop(args,
                          pl_model,
                          train_loader,
                          val_loader,
                          model_name):
    tb_logger = TensorBoardLogger(save_dir='lightning_logs', name=model_name)
    # getting the checkpoint dir
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, num_sanity_val_steps=0)
    trainer.tune(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(pl_model, train_loader, val_loader)
    return


def diffusion_training_loop(model, train_loader, results_folder):
    trainer = DiffusionTrainer(model, train_loader, results_folder=results_folder)
    trainer.train()
    return

