import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import _cufft_get_plan_cache_size
import torch.functional as F

import torchvision.models as models
from clrnet.models.heads import CLRHead, MyCLRHead
from clrnet.models.necks import FPN
from clrnet.models.backbones import ResNetWrapper

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Encoder:
  def __init__(self):
    super().__init__()

class Decoder:
  def __init__(self):
    super().__init__()

class CLRNet(pl.LightningModule):

  def __init__(self, backbone: ResNetWrapper, 
                     neck: FPN | None, 
                     heads: CLRHead | MyCLRHead):
    print('Init CLRNet...')
    super().__init__()
    # print('backbone = ', backbone)
    self.backbone = backbone
    self.automatic_optimization = False
    self.save_hyperparameters(ignore=['backbone', 'neck', 'heads'])

    # print('neck = ', neck)   
    self.neck = neck
    
    # print('head = ', heads)
    self.heads = heads
    self.aggregator = None

    print('Init CLRNet Done.')

  def view(self, img, img_full_path):
    print("view...")
    print("view img.shape: ", img.shape)
    plt.imshow(img.permute(1, 2, 0)) # [H, W, C]
    plt.show()

  def forward(self, batch):
    # print('CLRNet forward batch:', batch)
    print('CLRNet forward...')
    print("CLRNet batch.keys: ", batch.keys()) # dict_keys(['img', 'lane_line', 'seg', 'meta'])
    print("CLRNet batch.meta: ", batch['meta']) # {'full_img_path': ['data/CULane/driver_23_30frame/05161223_0545.MP4/04545.jpg', 'img2', ...]}
    out = {}
    out = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

    if self.aggregator:
      out[-1] = self.aggregator(out[-1])

    if self.neck:
      out = self.neck(out)

    img_idx = 0
    img = batch['img'][img_idx]
    img_full_path = batch['meta']['full_img_path'][img_idx]

    self.view(img, img_full_path)
    print("after view ")
    exit(0)

    # print('clrnet forward neck out: ', out)
    print('clrnet forward batch: ', batch.keys())
    if self.training:
      out = self.heads(out, batch=batch)
    else:
      out = self.heads(out)

    return out

  def training_step(self, batch, batch_idx):
    print('CLRNet training step...')
    # print('batch_idx: ', batch_idx)
    # print('batch: ', batch)
    # print('batch key list: ', list(batch.keys()))
    img = batch['img']
    lane_line = batch['lane_line']
    seg = batch['seg']
    meta = batch['meta']

    print('img len: ', len(img))
    output = self(batch)
    print("output['loss']: ", output['loss'])

    opt = self.optimizers()
    opt.zero_grad()
    loss = output['loss'].sum()
    self.manual_backward(loss)
    opt.step()
    return
    # loss = output['loss'].sum()
    # print('training_step: after y_hat = self(batch)')
    # print('loss: ', loss)
    # loss = nn.CrossEntropyLoss(y_hat, seg)
    # print('y_hat = ', y_hat)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  # def on_train_epoch_end(self):
    # all_preds = torch.stack(self.training_step_outputs)
    # do something with all preds
    # ...
    # self.training_step_outputs.clear()  # free memory
  
  # def on_train_start(self) -> None:
  #   print('on train start | ')
  #   return super().on_train_start()

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=0.6e-3)
