import torch
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/", batch_size: int = 32, split_train_valid: Optional[list] = None,
                 **dataloader_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_train_valid = split_train_valid if split_train_valid is not None else [55000, 5000]
        self.dataloader_kwargs = dataloader_kwargs
        self.transform = transforms.ToTensor()  # transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.save_hyperparameters('batch_size', 'split_train_valid')

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, self.split_train_valid)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, **self.dataloader_kwargs)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #
