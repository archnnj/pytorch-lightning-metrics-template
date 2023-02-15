import pytorch_lightning as pl
from torchmetrics import MetricCollection, Metric
from typing import Union, List, TypedDict, Mapping, Optional, Any
class MeteredExperiment(pl.LightningModule):
    def __init__(self,
                 metrics_train=None,
                 metrics_valid=None,
                 metrics_test=None,
                 log_loss_train: Optional[str] = 'step',
                 log_loss_valid: Optional[str] = 'step',
                 log_loss_test: Optional[str] = 'step',
                 log_metrics_train: Optional[Union[bool, str]] = 'step',
                 log_metrics_valid: Optional[Union[bool, str]] = 'epoch',
                 log_metrics_test: Optional[Union[bool, str]] = 'epoch',
                 multi_output: Optional[int] = None):
        super().__init__()

        assert (not log_loss_train or log_loss_train in ['epoch', 'step']) and \
               (not log_loss_valid or log_loss_valid in ['epoch', 'step']) and \
               (not log_loss_test or log_loss_test in ['epoch', 'step']), \
                f"{self.__class__.__name__}: log_loss_train|valid|test must be either None or 'epoch'|'step'."

        assert not multi_output or multi_output > 0, f"{self.__class__.__name__}: multi_output must be >= 1."
        self.multi_output = multi_output

        self.save_hyperparameters(ignore=['metrics_train', 'metrics_valid', 'metrics_test'])

        # Metrics settings
        self._setup_metrics(metrics_train, metrics_valid, metrics_test,
                            log_loss_train, log_loss_valid, log_loss_test,
                            log_metrics_train, log_metrics_valid, log_metrics_test)

    def update_metrics(self, split, preds, y, loss, losses=None):
        metrics_on_step = getattr(self, f'log_metrics_{split}_on_step')
        metrics_on_epoch = getattr(self, f'log_metrics_{split}_on_epoch')
        loss_on_step = getattr(self, f'log_loss_{split}_on_step')
        loss_on_epoch = getattr(self, f'log_loss_{split}_on_epoch')
        if losses is not None and losses.shape[-1] > 1:
            self.log(f'{split}/loss', loss, on_step=loss_on_step, on_epoch=loss_on_epoch, prog_bar=True, logger=True)
            for i, l in enumerate(losses):
                metric = getattr(self, f'metrics_{split}_head{i}')
                # update metrics
                metric(preds[i], y[i])
                # m.update(preds[i], y[i])
                # log losses
                self.log(f'{split}/loss{i}', l, on_step=loss_on_step, on_epoch=loss_on_epoch, prog_bar=True, logger=True)
                # print(f'\n{split}/loss{i}: {l}')
                if metrics_on_step or metrics_on_epoch:
                    # m.compute()
                    self.log_dict(metric, on_step=metrics_on_step, on_epoch=metrics_on_epoch, prog_bar=False,
                                  logger=True)
        else:
            self.log(f'{split}/loss', loss, on_step=loss_on_step, on_epoch=loss_on_epoch, prog_bar=True, logger=True)
            metric = getattr(self, f'metrics_{split}')
            metric(preds, y)
            if metrics_on_step or metrics_on_epoch:
                self.log_dict(metric, on_step=metrics_on_step, on_epoch=metrics_on_epoch, prog_bar=False,
                              logger=True)

    def _setup_metrics(self, metrics_train, metrics_valid, metrics_test,
                            log_loss_train, log_loss_valid, log_loss_test,
                            log_metrics_train, log_metrics_valid, log_metrics_test):
        self.log_loss_train_on_step = log_loss_train and log_loss_train == 'step'
        self.log_loss_train_on_epoch = self.log_loss_train_on_step or (log_loss_train and log_loss_train == 'epoch')
        self.log_loss_valid_on_step = log_loss_valid and log_loss_valid == 'step'
        self.log_loss_valid_on_epoch = self.log_loss_valid_on_step or (log_loss_valid and log_loss_valid == 'epoch')
        self.log_loss_test_on_step = log_loss_test and log_loss_test == 'step'
        self.log_loss_test_on_epoch = self.log_loss_test_on_step or (log_loss_test and log_loss_test == 'epoch')
        self.log_metrics_train_on_step = log_metrics_train and log_metrics_train == 'step'
        self.log_metrics_train_on_epoch = self.log_metrics_train_on_step or (log_metrics_train and log_metrics_train == 'epoch')
        self.log_metrics_valid_on_step = log_metrics_valid and log_metrics_valid == 'step'
        self.log_metrics_valid_on_epoch = self.log_metrics_valid_on_step or (log_metrics_valid and log_metrics_valid == 'epoch')
        self.log_metrics_test_on_step = log_metrics_test and log_metrics_test == 'step'
        self.log_metrics_test_on_epoch = self.log_metrics_test_on_step or (log_metrics_test and log_metrics_test == 'epoch')
        self._setup_one_metric(metrics_train, 'train')  # create self.metrics_train:
        self._setup_one_metric(metrics_valid if metrics_valid is not None else metrics_train, 'valid')
        self._setup_one_metric(metrics_test if metrics_test is not None else metrics_train, 'test')

    def _setup_one_metric(self, m, split, *args, **kwargs):
        if m is None:  # nothing specified, empty metrics
            metrics = MetricCollection([], *args, **kwargs)
            metrics.prefix = split
            setattr(self, f'metrics_{split}', metrics)
        else:
            if self.multi_output:
                # multiple outputs
                assert isinstance(m, (list, tuple)) and self.multi_output == len(m), \
                    f"{self.__class__.__name__}.setup_metrics(): metrics must be a list, tuple or dict, with one item per output head of the network"
                for i in range(self.multi_output):
                    # if isinstance(m, dict):
                    #     label = list(m.keys())[i]
                    #     # metrics_train_nhead[label] = list(metrics.values())[i]
                    # else:  # elif isinstance(m, (list, tuple)):
                    label = f'head{i}'
                    setattr(self, f'metrics_{split}_{label}', m[i].clone(prefix=f'{split}/{label}/'))
                setattr(self, f'metrics_{split}_collection', MetricCollection(m, prefix=f'{split}/'))
            else:
                setattr(self, f'metrics_{split}', m.clone(prefix=f'{split}/'))
