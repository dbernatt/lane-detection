import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from torchvision.models import resnet18, resnet50
from torchsummary import summary
import os


class Core(pl.LightningModule):
    def __init__(self, config):
        self.config = config
        self.backbone = config
        self.fpn = None
        self.net = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self._forward_impl(x)
        train_loss = F.cross_entropy(pred, y)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self._forward_impl(x)
        test_loss = F.cross_entropy(pred, y)
        self.log('test_loss', test_loss, prog_bar=True)

    def _forward_impl(self, x):
        pass

    def forward(self, x):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
