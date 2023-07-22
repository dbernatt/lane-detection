from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from torchvision.datasets import MNIST
from torchvision import transforms

class CULaneDataModule(pl.LightningDataModule):

  def __init__(self, data_root, batch_size = 16):
    super().__init__()
    self._log_hyperparams = None
    self.data_root = data_root
    self.batch_size = batch_size
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

  
  def prepare_data(self):
    print('Preparing datamodule...')
    # download dataset
    MNIST(self.data_root, train=True, download=True)
    MNIST(self.data_root, train=False, download=True)
    pass
  

  def prepare_data_per_node():
    pass

  def setup(self, stage = str):
    print('Setup datamodule...')
    # Assign train/val datasets for use in dataloaders
    if stage == "fit":
      print('Fit setup...')
      mnist_full = MNIST(self.data_root, train=True, transform=self.transform)
      self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
      pass

    # Assign test dataset for use in dataloader(s)
    if stage == "test":
      print('Test setup...')
      pass

    if stage == "predict":
      print('Predict setup...')
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