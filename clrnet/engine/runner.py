import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.functional as F

import torchvision.models as models
import torchvision as tv
from clrnet.models.heads import CLRHead, MyCLRHead
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
                     heads: CLRHead | MyCLRHead):
    print('Init Runner...')
    super().__init__()

    self.save_hyperparameters(ignore=['backbone', 'neck', 'heads'])
    self.automatic_optimization = False
    # self.backbone = backbone
    # self.neck = neck
    # self.heads = heads
    # self.aggregator = None
    self.net = Detector(backbone, neck, heads)
    print('Init Runner Done.')

  def forward(self, batch):
    return self.net(batch)

  def training_step(self, batch, batch_idx):
    print('CLRNet training step...')
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
    loss = output['loss'].sum()
    print('CLRNet training step over.')
    # print("output['loss']: ", output['loss'])

    # opt = self.optimizers()
    # opt.zero_grad()
    # loss = output['loss'].sum()
    # self.manual_backward(loss)
    # opt.step()
    # loss = output['loss'].sum()
    # print('training_step: after y_hat = self(batch)')
    # print('loss: ', loss)
    # loss = nn.CrossEntropyLoss(y_hat, seg)
    # print('y_hat = ', y_hat)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  # def on_train_epoch_end(self):
  #   pass
    # all_preds = torch.stack(self.training_step_outputs)
    # do something with all preds
    # ...
    # self.training_step_outputs.clear()  # free memory
  
  # def on_train_start(self) -> None:
  #   print('on train start | ')
  #   return super().on_train_start()

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=0.6e-3)
