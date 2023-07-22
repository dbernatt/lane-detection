from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import pickle as pkl
import os
import numpy as np
import os.path as osp

from torchvision import transforms
from clrnet.datasets.process.transforms import ToTensor, Normalize
from clrnet.datasets import CULaneDataset

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}

class CULaneDataModule(pl.LightningDataModule):
  # *args, **kwargs
  def __init__(self, data_root, 
                      batch_size,
                      img_w,
                      img_h,
                      cut_height,
                      img_norm,
                      transforms,
                      work_dirs):
    super(CULaneDataModule, self).__init__()
    print('Init CULaneDataModule...')
    self.save_hyperparameters()
    self.data_root = data_root
    self.batch_size = batch_size
    # self.load_annotations()
    self.transform = transforms.Compose([ToTensor(), Normalize(img_norm)])

  
  def prepare_data(self):
    print('Preparing CULaneDataModule...')
    # download dataset

  def setup(self, stage = str):
    print('Setup CULaneDataModule...')
    # Assign train/val datasets for use in dataloaders
    
    if stage == "fit":
      print('fit: setup...')
      self.split = 'train'
      self.list_path = osp.join(self.data_root, LIST_FILE[self.split])
      self.load_annotations()


      # mnist_full = MNIST(self.data_root, train=True, transform=self.transform)
      # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
      

    # Assign test dataset for use in dataloader(s)
    if stage == "test":
      print('test: setup...')
      self.list_path = osp.join(self.data_root, LIST_FILE['test'])
      self.split = 'test'
      train_dataset = CULaneDataset()

    if stage == "predict":
      print('predict: setup...')
      self.list_path = osp.join(self.data_root, LIST_FILE['predict'])
      self.split = 'val'
    
    print('Done.')
    return

  def load_annotations(self):
      print('Loading CULane annotations...')
      # Waiting for the dataset to load is tedious, let's cache it
      os.makedirs('cache', exist_ok=True)
      cache_path = 'cache/culane_{}.pkl'.format(self.split)
      if os.path.exists(cache_path):
          with open(cache_path, 'rb') as cache_file:
              self.data_infos = pkl.load(cache_file)
              self.max_lanes = max(
                  len(anno['lanes']) for anno in self.data_infos)
              return

      self.data_infos = []
      with open(self.list_path) as list_file:
          for line in list_file:
              infos = self.load_annotation(line.split())
              self.data_infos.append(infos)
      
      # cache data infos to file
      with open(cache_path, 'wb') as cache_file:
          pkl.dump(self.data_infos, cache_file)

  def load_annotation(self, line):
      infos = {}
      img_line = line[0]
      img_line = img_line[1 if img_line[0] == '/' else 0::]
      img_path = os.path.join(self.data_root, img_line)
      infos['img_name'] = img_line
      infos['img_path'] = img_path
      if len(line) > 1:
          mask_line = line[1]
          mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
          mask_path = os.path.join(self.data_root, mask_line)
          infos['mask_path'] = mask_path

      if len(line) > 2:
          exist_list = [int(l) for l in line[2:]]
          infos['lane_exist'] = np.array(exist_list)

      anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
      with open(anno_path, 'r') as anno_file:
          data = [
              list(map(float, line.split()))
              for line in anno_file.readlines()
          ]
      lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
      lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
      lanes = [lane for lane in lanes
                if len(lane) > 2]  # remove lanes with less than 2 points

      lanes = [sorted(lane, key=lambda x: x[1])
                for lane in lanes]  # sort by y
      infos['lanes'] = lanes

      return infos

  def train_dataloader(self):
    print('Create train dataloader...')
    return DataLoader(self.mnist_train, batch_size=self.batch_size)
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