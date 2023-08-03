import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.functional as F

import torchvision.models as models
from clrnet.models.heads import CLRHead, MyCLRHead
from clrnet.models.necks import FPN
from clrnet.models.backbones import ResNetWrapper

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sys import exit
from clrnet.utils.visualization import display_image_in_actual_size
from pytorch_lightning.loggers import TensorBoardLogger

class Encoder:
  def __init__(self):
    super().__init__()

class Decoder:
  def __init__(self):
    super().__init__()

class CLRNet(pl.LightningModule):

  def __init__(self,
                     backbone: ResNetWrapper, 
                     neck: FPN | None, 
                     heads: CLRHead | MyCLRHead):
    print('Init CLRNet...')
    super().__init__()
    # self.trainer = trainer
    print(self.logger)

    # self.trainer.logger = TensorBoardLogger(save_dir='lightning_logs/')
    # self.logger = TensorBoardLogger(save_dir='lightning_logs/')
    # print(self.logger)

    # self.logger.experiment.add_image(img_name, img, self.current_epoch, dataformats="HW")
    # print('trainer: ', self.trainer)
    # print('trainer: ', self.trainer.logger)
    # print('CLRNet log: ', self.log)
    # print('CLRNet trainer.logger: ', self.trainer.logger)
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

  def makegrid(output,numrows):
    outer=(torch.Tensor.cpu(output).detach())
    plt.figure(figsize=(20,5))
    b=np.array([]).reshape(0,outer.shape[2])
    c=np.array([]).reshape(numrows*outer.shape[2],0)
    i=0
    j=0
    while(i < outer.shape[1]):
      img=outer[0][i]
      b=np.concatenate((img,b),axis=0)
      j+=1
      if(j==numrows):
        c=np.concatenate((c,b),axis=1)
        b=np.array([]).reshape(0,outer.shape[2])
        j=0
            
      i+=1
    return c
  
  def showActivations(self,x):
    # logging reference image        
    self.logger.experiment.add_image("input",torch.Tensor.cpu(x[0][0]),self.current_epoch,dataformats="HW")

    # logging layer 1 activations        
    # out = self.layer1(x)
    # c=self.makegrid(out,4)
    # self.logger.experiment.add_image("layer 1",c,self.current_epoch,dataformats="HW")
      
    # # logging layer 1 activations        
    # out = self.layer2(out)
    # c=self.makegrid(out,8)
    # self.logger.experiment.add_image("layer 2",c,self.current_epoch,dataformats="HW")

    # # logging layer 1 activations        
    # out = self.layer3(out)
    # c=self.makegrid(out,8)
    # self.logger.experiment.add_image("layer 3",c,self.current_epoch,dataformats="HW")
      

  def log_tb_images(self, img, img_name) -> None:
    print('self.logger: ', self.logger)
        
    # Get tensorboard logger
    tb_logger = None

    if isinstance(self.logger, pl.loggers.TensorBoardLogger):
      tb_logger = self.logger.experiment

    if tb_logger is None:
      raise ValueError('TensorBoard Logger not found')

    # tb_logger = self.trainer.logger.experiment

    tb_logger.add_image(f"Image/img_path", img, 0)



  def forward(self, batch):
    # print('CLRNet forward batch:', batch)
    print('CLRNet forward...')
    print("CLRNet batch.keys: ", batch.keys()) # dict_keys(['img', 'lane_line', 'seg', 'meta'])
    print("CLRNet batch.meta: ", batch['meta']) # {'full_img_path': ['data/CULane/driver_23_30frame/05161223_0545.MP4/04545.jpg', 'img2', ...]}
    # batch['img'].shape = torch.Size([24, 3, 160, 400])
    out = {}
    img = batch['img'][0]
    img_name = '/'.join(batch['meta']['full_img_path'][0].split('/')[-3:])
    print("img_name: ", img_name)
    tensorboard = self.logger.experiment
    print("self.logger.experiment = ", self.logger.experiment)
    # print("batch['img'].shape: ", batch['img'].shape)

    tensorboard.add_image(img_name, img, self.current_epoch)
    # self.log_tb_images(img, img_name)
    # print('forward self logger: ', self.logger)
    # print('forward self logger: ', self.logger.experiment)
    # self.logger.experiment.add_image(img_name, img, self.current_epoch, dataformats="HW")
    out = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
    return out

    if self.aggregator:
      out[-1] = self.aggregator(out[-1])

    if self.neck:
      out = self.neck(out)

    img_idx = 0
    img = batch['img'][img_idx]
    img_full_path = batch['meta']['full_img_path'][img_idx]

    # self.view(img, img_full_path)
    for i in range(10):
      img = batch['img'][i]
      img_full_path = batch['meta']['full_img_path'][i]
      display_image_in_actual_size(img, img_full_path)

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
    # print('batch key list: ', list(batch.keys())) # ['img', 'lane_line', 'seg', 'meta']
    img = batch['img'] # torch.Size([24, 3, 160, 400])
    lane_line = batch['lane_line']
    seg = batch['seg']
    meta = batch['meta'] # {'full_img_path': [...]}

    print('img len: ', len(img)) # 24
    output = self(batch)
    # return 0.2
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
    # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    # return loss
  
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
