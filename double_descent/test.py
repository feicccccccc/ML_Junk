import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy

MNIST.mirrors = [mirror for mirror in MNIST.mirrors
                 if not mirror.startswith("http://yann.lecun.com")]

CONFIG = {
    # Some Hyperparameter Defination
    "bs": 512,
    "lr": 1e-4,
    "epoch": 2000,

    # From Paper
    "noise": 0.2,  # Lable Noise
    "k": 60,  # Model Complexity. Layer specific as [28*28, 1k, 2k, 1k, 10]
}

# wandb run name
NAME = "Conv-CIFAR10-bs{}-lr{}-k{}-noise{}".format(CONFIG['bs'], CONFIG['lr'], CONFIG['k'], CONFIG['noise'])

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 512, noise: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noise = noise
        self.num_workers = 0

        self.save_hyperparameters()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        self.test_set = CIFAR10(self.data_dir, train=False, transform=transforms.ToTensor(), download=True)
        self.full_data_set = CIFAR10(self.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        self.train_set, self.val_set = random_split(self.full_data_set, [45000, 5000])

        # Add noise to training set label
        for i in range(len(self.train_set)):
            idx = self.train_set.indices[i]
            if torch.rand(1) < 0.2:
                self.train_set.dataset.targets[idx] = torch.randint(0, 10, (1,))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # For inference, used in Trainer.predcit()
    def predict_dataloader(self):
        # TODO: Check what to predict
        return DataLoader(self.test_set, batch_size=64)


dm = CIFAR10DataModule(batch_size=CONFIG['bs'], noise=CONFIG['noise'])
dm.prepare_data()
dm.setup()

# Display image and label.
train_features, train_labels = next(iter(dm.train_dataloader()))

print(train_features.size())
print(train_labels.size())
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

images=[]

for i in range(0, 5):
    images.append(train_features[i].squeeze())
    print(images[i].shape)

plt.figure(figsize=(20,10))
columns = 5

for i, image in enumerate(images):
    ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
    ax.imshow(image.permute(1, 2, 0))
    ax.title.set_text("Lable: {}".format(train_labels[i]))