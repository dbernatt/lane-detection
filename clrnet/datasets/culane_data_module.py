from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import pickle as pkl
import os
import numpy as np
import os.path as osp
from torchvision.transforms import Compose

from torchvision import transforms
from clrnet.datasets.process.transforms import ToTensor, Normalize
from clrnet.datasets import CULaneDataset
from clrnet.datasets.process import Process
from torch.utils.data.dataloader import default_collate

class CULaneDataModule(pl.LightningDataModule):
  def __init__(self, data_type,
                      data_root, 
                      batch_size,
                      img_w,
                      img_h,
                      cut_height,
                      img_norm,
                      work_dirs,
                      processes):
    super(CULaneDataModule, self).__init__()
    print('Init CULaneDataModule...')

    self.save_hyperparameters()
    self.cfg = {
      "data_root": data_root,
      "batch_size": batch_size,
      "img_w": img_w,
      "img_h": img_h,
      "cut_height": cut_height,
      "work_dirs": work_dirs,
      "img_norm": img_norm,
    }
    self.processes = processes

  def prepare_data(self):
    print('Preparing CULaneDataModule...')
    # download dataset

  def setup(self, stage = str):
    print('Setup CULaneDataModule...')
    # Assign train/val datasets for use in dataloaders

    if stage == "fit":
      print('fit: setup...')
      self.split = 'train'
      self.culane_train_set = CULaneDataset(self.cfg, self.split, self.processes['train'])

    # Assign test dataset for use in dataloader(s)
    if stage == "test":
      print('test: setup...')
      self.split = 'test'

    if stage == "predict":
      print('predict: setup...')
      self.split = 'val'

    print('Done.')
    return

  def train_dataloader(self):
    print('Create train dataloader...')
    return DataLoader(self.culane_train_set, batch_size=self.batch_size)
    # return DataLoader(self.mnist_train, batch_size=self.batch_size)

  def val_dataloader(self):
      print('Create val dataloader...')
      # return DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
      print('Create test dataloader...')
      # return DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
      print('Create predict dataloader...')
      # return DataLoader(self.mnist_predict, batch_size=self.batch_size)