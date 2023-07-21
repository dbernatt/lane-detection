from torch.utils.data import DataLoader
import pytorch_lightning as pl

class CULaneDataModule(pl.LightningDataModule):

  def __init__(self, data_root, batch_size = 16, train_transforms=None, val_transforms=None, test_transforms=None):
    super(CULaneDataModule).__init__()
    self.data_root = data_root
    self.batch_size = batch_size
    self.train_transforms = train_transforms
    self.val_transforms = val_transforms
    self.test_transforms = test_transforms
  
  def prepare_data(self):
    print('Preparing datamodule...')
    # download
    pass
  
  def setup(self, stage = str):
    print('Setup datamodule...')
    # Assign train/val datasets for use in dataloaders
    if stage == "fit":
      print('Fit setup...')
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

  def val_dataloader(self):
      print('Create val dataloader...')
      return DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
      print('Create test dataloader...')
      return DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
      print('Create predict dataloader...')
      return DataLoader(self.mnist_predict, batch_size=self.batch_size)