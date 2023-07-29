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

class CULaneDataModuleParams(object):
  def __init__(self, data_root, 
                      batch_size, 
                      img_w, 
                      img_h,
                      cut_height,
                      work_dirs,
                      img_norm,
                      num_points,
                      max_lanes,
                      workers,
                      ):
    self.data_root = data_root
    self.batch_size = batch_size
    self.img_w = img_w
    self.img_h = img_h
    self.cut_height = cut_height
    self.work_dirs = work_dirs
    self.img_norm = img_norm
    self.num_points = num_points
    self.max_lanes = max_lanes
    self.workers = workers

class CULaneDataModule(pl.LightningDataModule):
  def __init__(self, cfg: CULaneDataModuleParams, processes):
    super(CULaneDataModule, self).__init__()
    print('Init CULaneDataModule...')
    self.save_hyperparameters()

    # self.data_type = data_type
    # self.data_root = data_root
    # self.batch_size = batch_size
    # self.img_w = img_w
    # self.img_h = img_h
    # self.cut_height = cut_height
    # self.img_norm = img_norm
    # self.work_dirs = work_dirs
    
    print('data module cfg: ', cfg)
    print('data module processes: ', processes)
    self.cfg = cfg
    self.processes = processes
    self.culane_train_set = None
    self.culane_val_set = None
    self.culane_test_set = None

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
    # init_fn = partial(self.worker_init_fn, seed=self.cfg['seed'])
    data_loader =  DataLoader(
          self.culane_train_set,
          batch_size=self.cfg.batch_size,
          shuffle=True,
          num_workers=self.cfg.workers,
          # pin_memory=False,
          # drop_last=False,
          # collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
          # worker_init_fn=init_fn
          )
    print('data_loader len:', len(data_loader))
    print()
    return data_loader
    # return DataLoader(self.culane_train_set, batch_size=self.cfg['batch_size'])

  def val_dataloader(self):
      print('Create val dataloader...')
      # return DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
      print('Create test dataloader...')
      # return DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
      print('Create predict dataloader...')
      # return DataLoader(self.mnist_predict, batch_size=self.batch_size)