import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.functional as F

import torchvision.models as models
import torchvision as tv
from clrnet.models.heads import MyCLRHead
from clrnet.models.necks import FPN
from clrnet.models.backbones import ResNetWrapper

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sys import exit
from clrnet.utils.visualization import display_image_in_actual_size, show_img
from pytorch_lightning.loggers import TensorBoardLogger
from clrnet.models.nets import Detector

class Encoder:
  def __init__(self):
    super().__init__()

class Decoder:
  def __init__(self):
    super().__init__()

class Runner(pl.LightningModule):

  def __init__(self,
                     backbone: ResNetWrapper, 
                     neck: FPN | None, 
                     heads: MyCLRHead):
    print('Init Runner...')
    super().__init__()

    

    self.save_hyperparameters(ignore=['backbone', 'neck', 'heads'])
    self.automatic_optimization = False
    # self.backbone = backbone
    # self.neck = neck
    # self.heads = heads
    # self.aggregator = None
    self.net = Detector(backbone, neck, heads)
    self.view = False
    self.workdirs = 'work_dirs/clr/r18_culane'

    print('Init Runner Done.')

  def forward(self, batch):
    return self.net(batch)

  def training_step(self, batch, batch_idx):
    # print('CLRNet training step...')
    # print('batch_idx: ', batch_idx)
    # print('batch: ', batch)
    # print('batch key list: ', list(batch.keys())) # ['img', 'lane_line', 'seg', 'meta']
    imgs = batch['img'] # torch.Size([24, 3, 160, 400])
    lane_line = batch['lane_line']
    seg = batch['seg']
    meta = batch['meta'] # {'full_img_path': [...]}

    # print('img len: ', len(imgs)) # 24
    # self.showActivations(batch_idx, batch)
    output = self(batch)
    # output = self.net.module.heads.get_lanes(output)
    #             predictions.extend(output)
    # pred = output.get_lanes()
    # self.validation_step_outputs.append(output)

    # print("output: ", output)

    losses = output['loss']
    loss_stats = output['loss_stats']
    loss = output['loss'].sum()

    # self.logger.experiment.add_image(f"{group_path}/backbone_1", 
    #                                   numpy_image_1, 
    #                                   dataformats='NCHW') # group view: CHW
    
    self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def _shared_eval_step(self, batch, batch_idx):
    y = self(batch)
    loss = F.cross_entropy(y_hat, y)
    acc = accuracy(y_hat, y)
    return loss, acc

  def test_step(self, batch, batch_idx):
    loss, acc = self._shared_eval_step(batch, batch_idx)
    metrics = {"test_acc": acc, "test_loss": loss}
    self.log_dict(metrics)
    return metrics

  # def on_train_batch_end(self, outputs, batch, batch_idx: int):
  #   pass
    # print('on_train_batch_end...')
    # print('outputs: ', outputs)
    # output_lanes = self.net.heads.get_lanes(batch)
    # print('output_lanes: ', output_lanes)
  
  # def on_train_epoch_end(self):
  #   pass
    # all_preds = torch.stack(self.training_step_outputs)
    # do something with all preds
    # ...
    # self.training_step_outputs.clear()  # free memory
  
  # def on_train_start(self) -> None:
  #   print('on train start | ')
  #   return super().on_train_start()

  # def on_train_epoch_end(self):
  #   all_preds = torch.stack(self.training_step_outputs)


  # def test_step(self, batch, batch_idx):
  #   print("Test step")
  #   pass

  def validation_step(self, batch, batch_idx):
    pass

  # def validation_step(self, batch, batch_idx):
  #     print("Validation step...")

  #     output = self(batch)
  #     print("output: ", output.shape)
  #     # print(self.net.modules)
  #     # print(self.net.modules.heads)
  #     output_lanes = self.net.heads.get_lanes(output)
  #     print("output_lanes: ", output_lanes)
  #     print("output_lanes over.")

  #     self.predictions.extend(output_lanes)

  # def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
  #   print('on_validation_batch_end...')
  #   print("self.predictions : ", self.predictions)
  #   if self.view:
  #     # print(self.trainer.val_dataloaders == self.val_dataloader)
  #     # print(self.val_dataloader())
  #     # val_dataloader = self.val_dataloader()
  #     # val_dataset
  #     print("outputs: ", outputs)
  #     print("batch len: ", len(batch))
  #     print("batch img shape: ", batch['img'].shape)
  #     print("batch keys: ", batch.keys())
  #     print("batch meta: ", batch['meta'])
  #     self.trainer.val_dataloaders.dataset.view(outputs, batch['meta'])

  # def on_validation_start(self) -> None:
  #   print('\non_validation_start...')
  #   self.predictions = []

  # def on_validation_end(self):
  #   print('on_validation_end...')
  #   print("self.predictions : ", self.predictions)
  #   metric = self.trainer.val_dataloaders.dataset.evaluate(self.predictions, 
  #                                                          self.workdirs)
  #   self.log('metric', metric)

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=0.6e-3)
