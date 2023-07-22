from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from torchvision.datasets import MNIST
from torchvision import transforms
import os.path as osp

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

class CULaneDataset(pl.LightningDataModule):

  def __init__(self, data_root, batch_size):
    super().__init__()
    print('Init CULaneDataModule...')
    self.save_hyperparameters()
    self.data_root = data_root
    self.batch_size = batch_size
    # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

  
  def prepare_data(self):
    print('Preparing CULaneDataModule...')
    # download dataset
    # MNIST(self.data_root, train=True, download=True)
    # MNIST(self.data_root, train=False, download=True)
  

  def prepare_data_per_node():
    pass

  def setup(self, stage = str):
    print('Setup CULaneDataModule...')
    # Assign train/val datasets for use in dataloaders

    if stage == "fit":
      print('fit: setup...')
      self.list_path = osp.join(self.data_root, LIST_FILE['train'])
      # mnist_full = MNIST(self.data_root, train=True, transform=self.transform)
      # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
      
      
      pass

    # Assign test dataset for use in dataloader(s)
    if stage == "test":
      print('test: setup...')
      self.list_path = osp.join(self.data_root, LIST_FILE['test'])

      pass

    if stage == "predict":
      print('predict: setup...')
      self.list_path = osp.join(self.data_root, LIST_FILE['predict'])
      
      pass
    
    print('Done.')
    return
  
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