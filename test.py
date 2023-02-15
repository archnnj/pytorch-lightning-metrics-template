import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, NeptuneLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from datamodules.MNISTDataModule import MNISTDataModule
from experiments.MNISTExperiment import MNISTExperiment
from models.BackboneFC import BackboneFC
from core.utils import fix_lightning_logger, mkdir_ifnexists


if __name__ == '__main__':
    pl.seed_everything(1234)
    fix_lightning_logger()
    device = "cuda" if torch.cuda.is_available() else "auto"

    # Init constants
    BASE_DATASET_PATH = './data'

    LIGHTNING_PATH = './outputs/20230215164008/wandb/MNIST_test/8r82xyys'
    CKPT_PATH = LIGHTNING_PATH + '/checkpoints/epoch=1-step=3376.ckpt'

    # ------------
    # datamodule setup
    # ------------

    n_classes = 10
    in_ch = 1
    test_batch = 32
    mnist_datamodule = MNISTDataModule(data_dir=BASE_DATASET_PATH,
                                       batch_size=test_batch,
                                       split_train_valid=[1.0, 0.0])

    # ------------
    # model and experiment
    # ------------

    exp = MNISTExperiment.load_from_checkpoint(CKPT_PATH)
    exp.eval()

    # ------------
    # training setup
    # ------------

    trainer = pl.Trainer(enable_checkpointing=False, accelerator=device, devices=1) #, default_root_dir=LIGHTNING_PATH)

    # ------------
    # testing
    # ------------
    result = trainer.test(model=exp, datamodule=mnist_datamodule, verbose=True)
    print(result)
