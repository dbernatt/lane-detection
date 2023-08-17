from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import pickle as pkl
import os
import numpy as np
import os.path as osp
from torchvision.transforms import Compose
import random
from functools import partial
# from mmcv.parallel import collate

from torchvision import transforms
from clrnet.datasets import CULaneDataset
from clrnet.datasets.process import Process
from torch.utils.data.dataloader import default_collate
from clrnet.utils import Dict2Class

class CULaneDataModule(pl.LightningDataModule):
  def __init__(self, cfg,
                     processes,
                     *args,
                     **kwargs):
    super(CULaneDataModule, self).__init__()
    print('Init CULaneDataModule...')
    self.save_hyperparameters()

    self.cfg = Dict2Class(cfg)
    self.processes = processes
    print('data module cfg: ', self.cfg)
    print('data module processes: ', self.processes)

    self.train_set = None
    self.val_set = None
    self.test_set = None

  def worker_init_fn(worker_id, seed):
      worker_seed = worker_id + seed
      np.random.seed(worker_seed)
      random.seed(worker_seed)

  def prepare_data(self):
    print('Preparing CULaneDataModule...')
    # download dataset

  def setup(self, stage = str):
    print('Setup CULaneDataModule...')
    # Assign train/val datasets for use in dataloaders

    if stage == "fit":
      print('fit: setup...')
      print('fit: train_set setup...')
      self.train_set = CULaneDataset(self.cfg, 'train', self.processes['train'])
      print('fit: val_set setup...')
      self.val_set = CULaneDataset(self.cfg, 'val', self.processes['val'])

    # Assign test dataset for use in dataloader(s)
    if stage == "test":
      print('test: setup...')
      self.split = 'test'
      self.test_set = CULaneDataset(self.cfg, self.split)

    print('Done.')
    return

  def view(self):
    return "view"

  def train_dataloader(self):
    print('Create train dataloader...')
    # init_fn = partial(self.worker_init_fn, seed=self.cfg['seed'])
    data_loader = DataLoader(
          self.train_set,
          batch_size=self.cfg.batch_size,
          shuffle=True,
          num_workers=self.cfg.workers,
          pin_memory=False,
          drop_last=False,
          # collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
          # worker_init_fn=init_fn
          )
    item = next(iter(data_loader))
    # print("item seg: ", item['seg'].shape)
    # print('img shape: ', item['img'].shape) # torch.Size([24, 3, 160, 400])
    # print('seg shape: ', item['seg'].shape) # torch.Size([24, 160, 400]) (bg=0, fg=1)
    # print('lane_line shape: ', item['lane_line'].shape) # torch.Size([24, 4, 78])
    # print('meta shape: ', item['meta']) # torch.Size([24, 4, 78])
    """
      item = {
        'img': torch.Size([24, 3, 160, 400])
        'seg': torch.Size([24, 160, 400])
        'lane_line': torch.Size([24, 4, 78])
        'meta': { 'full_img_path': [img1path, img2path, ... img(24 -1)path]}
      }
    """
    # self.train_set.view(item[])
    print('data_loader len and seg shape:', len(data_loader), item['seg'].shape)
    return data_loader

  def val_dataloader(self):
    print('Create val dataloader...')
    data_loader = DataLoader(
          dataset=self.val_set,
          batch_size=self.cfg.batch_size,
          shuffle=False,
          num_workers=self.cfg.workers,
          pin_memory=False,
          drop_last=False)
    return data_loader

  def test_dataloader(self):
    print('Create test dataloader...')
    data_loader = DataLoader(
          self.test_set,
          batch_size=self.cfg.batch_size,
          shuffle=False,
          num_workers=self.cfg.workers,
          pin_memory=False,
          drop_last=False)
    return data_loader
