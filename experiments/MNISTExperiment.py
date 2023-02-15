from typing import Any, Callable, Type
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import torchmetrics
# from models._LightningMetricsModule import LightningMetricsModule
from torchmetrics import MetricCollection
from tqdm import tqdm
from core.experiment import MeteredExperiment


class MNISTExperiment(MeteredExperiment):
    def __init__(self,
                 model_class: Type[nn.Module],
                 model_args: Any,
                 loss_fcn: Type[Callable],
                 optimizer_type: str = "Adam",
                 opt_kwargs: dict = None,
                 **metered_exp_kwargs: Any):
        super().__init__(**metered_exp_kwargs)
        self.save_hyperparameters()
        self.model = self.hparams.model_class(**self.hparams.model_args)
        self.loss = self.hparams.loss_fcn()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._std_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._std_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self._std_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return nn.Softmax(self(batch))

    def configure_optimizers(self):
        if self.hparams.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), **self.hparams.opt_kwargs)
        elif self.hparams.optimizer_type == "SGD":
            return torch.optim.SGD(self.model.parameters(), **self.hparams.opt_kwargs)
        else:
            raise ValueError("MnistExperiment: optimizer_type must be either Adam or SGD")

    def _std_step(self, batch, batch_idx, phase):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        # Metrics
        self.update_metrics(phase, preds, y, loss)
        return loss
