# Metric-agnostic template for PyTorch Lightning

This is a simple boilerplate code for a metric-agnostic management of deep learning experiments using PyTorch Lightning. 

The class `core.Experiment` defines an extension of `pl.LightningModule` that abstracts the concept of deep learning experiment from its instantiation, with a specific model, optimizer and dataloader. It also automatically handles any combination of torchmetrics metrics passed at initialization, so that different metrics can be tracked without impacting the code of the experiment.<br>

Model, optimizer, dataloader, and metrics can be passed at initialization. <br>
The metrics can be passed as a dictionary or as a torchmetrics MetricCollection. It is also possible to specify whether they must be computed at each optimization step or at the end of the epoch.
It is possible to specify different sets of metrics for training, val and test phases.<br>

All metrics can then be logged using [Weights & Biases](https://wandb.ai/site) or [Neptune.ai](https://neptune.ai/).

## Usage

### Basic example

This repository contains a simple example based on MNIST. You can run it with:

`python train.py`

Additionally, once a checkpoint is save, you can specify its path in `test.py` and obtain the testing metrics with:

`python test.py`

### Extending it in your own work

To use this template in your research, simply extend `core.Experiment` instead of `pl.LightningModule` when creating the top-level class for your experiment. For submodules, you can still use `nn.Model` or `pl.LightningModule`.

For example, to define an experiment training a model on MNIST:

```
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

    def _std_step(self, batch, batch_idx, phase):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        # Metrics
        self.update_metrics(phase, preds, y, loss)
        return loss
```

Notice that the experiment is agnostic to model, optimizer, dataloader and metrics to be tracked.

Then, in your `train.py` you can indicate them when instantiating the experiment:

```
experiment = MNISTExperiment(
    model_class=model_class,
    model_args=model_args,
    loss_fcn=nn.CrossEntropyLoss,  # F.cross_entropy,
    optimizer_type=PARAMS['optimizer'],
    opt_kwargs=PARAMS['opt_kwargs'],
    metrics_train=[
        MetricCollection({
              'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=n_classes),
              'precision': torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='micro'),
              'recall': torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='micro'),
              'auroc': torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
        }),  # , prefix='regr/'),
    # you can also specify custom behavior for each single train/val/test phase using `metrics_val` and `metrics_test`; by default, when not specified, they are assumed to be duplicates of `metrics_train`
    log_loss_train='step',
    log_loss_valid='epoch',
    log_loss_test='epoch',
    log_metrics_train='step',
    log_metrics_valid='epoch',
    log_metrics_test='epoch'
)
```
Notice in particular how easy it is to add and remove a metric, without having to modify any of the internal mechanisms of the experiment.

Please check out the rest of the code in this repo for details (or feel free to [reach out](https://pcudrano.github.io/)!).
