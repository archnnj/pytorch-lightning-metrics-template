from torch import nn
import pytorch_lightning as pl

class FC(pl.LightningModule):
    _activation_list = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }

    def __init__(self, in_dim, hidden_dims, out_dim, activation='relu'):
        super().__init__()
        self.save_hyperparameters('in_dim', 'hidden_dims', 'out_dim', 'activation')
        self.model = nn.Sequential()
        for i in range(len(self.hparams.hidden_dims) + 1):
            in_d = self.hparams.hidden_dims[i - 1] if i > 0 else self.hparams.in_dim
            out_d = self.hparams.hidden_dims[i] if i < len(self.hparams.hidden_dims) else self.hparams.out_dim
            self.model.add_module(f"fc{str(i)}", nn.Linear(in_d, out_d))
            if i < len(self.hparams.hidden_dims):
                self.model.add_module(f"{self.hparams.activation}{str(i)}",
                                      FC._activation_list[self.hparams.activation])


    def forward(self, x):
        return self.model(x)
