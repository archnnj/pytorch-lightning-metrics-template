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

from core.utils import fix_lightning_logger


if __name__ == '__main__':
    pl.seed_everything(1234)
    fix_lightning_logger()
    device = "cuda" if torch.cuda.is_available() else "auto"

    neptune_args = dict(
        project="AIRLab/MNIST_test",
        name="MNIST test",
        description="MNIST test",
        log_model_checkpoints=True,
        tags=["MNIST", "test"],
        source_files=['**/*.py', '**/*.ipynb'],  # ['notebooks/2d_mnist_neptune.ipynb'] # [os.path.basename(__file__)]
        #capture_hardware_metrics=False
    )
    wandb_args = dict(
        project="MNIST_test",
        name="MNIST test",
    )
    PARAMS = dict(
        dataset_id="mnist",
        architecture="resnet18",
        optimizer="SGD",
        opt_kwargs={'lr': 1e-4, 'momentum':0.9, 'weight_decay':1e-8},  # | None
        early_stopping={'monitor': "valid/loss", 'mode': "min"},  # | None
        epochs=2,
        batch_size=64,
        hidden_dims=[100],
        activation='relu',
        cuda=torch.cuda.is_available(),
        split_train_valid=[0.9, 0.1],
        finetuning=False
    )

    # Init constants
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    BASE_DATASET_PATH = './data'
    LIGHTNING_PATH = f'./outputs/{timestamp}/lightning/'
    if not os.path.exists(LIGHTNING_PATH):
        os.makedirs(LIGHTNING_PATH, exist_ok=True)


    # ------------
    # datamodule setup
    # ------------
    n_classes = 10
    in_ch = 1
    mnist_datamodule = MNISTDataModule(data_dir=BASE_DATASET_PATH,
                                       batch_size=PARAMS['batch_size'],
                                       split_train_valid=PARAMS['split_train_valid'],
                                       num_workers=4)


    # ------------
    # model
    # ------------

    model = BackboneFC(
        in_ch=in_ch,
        fc_hidden_dims=PARAMS['hidden_dims'],
        out_dim=n_classes,
        backbone=PARAMS['architecture'],
        finetuning=PARAMS['finetuning'],
    )

    # ------------
    # experiment
    # ------------

    experiment = MNISTExperiment(
        model=model,
        loss_fcn=nn.CrossEntropyLoss(), #F.cross_entropy,
        optimizer_type=PARAMS['optimizer'],
        opt_kwargs=PARAMS['opt_kwargs'],
        metrics_train=
            MetricCollection({
                  'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=n_classes),
                  'precision': torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='micro'),
                  'recall': torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='micro'),
                  #'auroc': torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
            }),
            # MetricCollection({
            #       'mae': torchmetrics.MeanAbsoluteError(),
            #       'mse': torchmetrics.MeanSquaredError(),
            # }, prefix='regr/')
        #],
        log_loss_train='step',
        log_loss_valid='epoch',
        log_loss_test='epoch',
        log_metrics_train='step',
        log_metrics_valid='epoch',
        log_metrics_test='epoch'
    )
    print(ModelSummary(experiment))

    # ------------
    # training setup
    # ------------

    # neptune_logger = NeptuneLogger(**neptune_args)
    # neptune_logger.log_hyperparams(PARAMS)
    wandb_logger = WandbLogger(**wandb_args)
    logger = [wandb_logger] # , neptune_logger] # False | neptune_logger

    checkpoint_callback = ModelCheckpoint(monitor="valid/loss", mode="min")
    trainer_callbacks = [checkpoint_callback]
    if PARAMS['early_stopping']:
        early_stopping_callback = EarlyStopping(**PARAMS['early_stopping'])
        trainer_callbacks.append(early_stopping_callback)
    trainer = pl.Trainer(logger=logger,
                         callbacks=trainer_callbacks,
                         default_root_dir=LIGHTNING_PATH,
                         enable_checkpointing=True,
                         accelerator=device, devices=1,
                         max_epochs=PARAMS['epochs'])

    # ------------
    # training
    # ------------

    trainer.fit(experiment, datamodule=mnist_datamodule)

    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=mnist_datamodule)
    print(result)
