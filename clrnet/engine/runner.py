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
    # pred = output.get_lanes()
    # self.validation_step_outputs.append(output)

    print("output: ", output)

    losses = output['loss']
    loss_stats = output['loss_stats']
    loss = output['loss'].sum()

    stage_0_acc = loss_stats['stage_0_acc']
    stage_1_acc = loss_stats['stage_1_acc']
    stage_2_acc = loss_stats['stage_2_acc']
    cls_loss = loss_stats['cls_loss']
    reg_xytl_loss = loss_stats['reg_xytl_loss']
    iou_loss = loss_stats['iou_loss']

    # self.logger.experiment.add_image(f"{group_path}/backbone_1", 
    #                                   numpy_image_1, 
    #                                   dataformats='NCHW') # group view: CHW
    
    self.log("train_loss_sum", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True, logger=True)
    self.log("train_reg_xytl_loss", reg_xytl_loss, on_step=True, on_epoch=True, logger=True)
    self.log("train_iou_loss", iou_loss, on_step=True, on_epoch=True, logger=True)
    self.log("train_stage_0_acc", stage_0_acc, on_step=True, on_epoch=True, logger=True)
    self.log("train_stage_1_acc", stage_1_acc, on_step=True, on_epoch=True, logger=True)
    self.log("train_stage_2_acc", stage_2_acc, on_step=True, on_epoch=True, logger=True)

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

  # def on_train_epoch_end(self):
  #   all_preds = torch.stack(self.training_step_outputs)


  def test_step(self, batch, batch_idx):
    print("Test step")
    pass

  def validation_step(self, batch, batch_idx):
      print("Validation step...")

      output = self(batch)
      # print(self.net.modules)
      # print(self.net.modules.heads)
      output = self.net.heads.get_lanes(output)

      self.predictions.extend(output)

  def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
    print('on_validation_batch_end...')
    if self.cfg.view:
      self.val_loader.dataset.view(outputs, batch['meta'])

  def on_validation_start(self) -> None:
    print('\non_validation_start...')
    self.predictions = []

  def on_validation_end(self):
    print('on_validation_end...')
    metric = self.val_loader.dataset.evaluate(self.predictions,
                                              self.cfg.work_dir)
    self.log('metric', metric)

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=0.6e-3)
