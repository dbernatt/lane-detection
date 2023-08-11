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

    # Configs
    self.save_hyperparameters(ignore=['backbone', 'neck', 'heads'])
    self.automatic_optimization = False

    # print("backbone = ": backbone)
    self.backbone = backbone

    # print('neck = ', neck)   
    self.neck = neck
    
    # print('head = ', heads)
    self.heads = heads
    self.aggregator = None

    print('Init CLRNet Done.')
  
  def showActivations(self, batch_idx, batch, img_idx = 0, num_fea_imgs = 1):
    print('showActivations batch.keys: ', batch.keys())
    # logging reference image        
    img = batch['img'][img_idx]
    img_lane_line = batch['lane_line'][img_idx]
    img_name = '/'.join(batch['meta']['full_img_path'][0].split('/')[-3:])
    group_path = f"batch_{batch_idx}_imgid_{img_idx}/{img_name}"

    self.logger.experiment.add_image(f"{group_path}/_input_", 
                                      img, 
                                      self.current_epoch)

    print("img_lane_line.shape: ", img_lane_line.shape)
    print("img_lane_line.type: ", type(img_lane_line))
    print("img_lane_line[0]: ", img_lane_line[0])

    print("input img: ", img.shape)
    print("input img: ", img.shape)
    print("img min and max: ", img.min(), img.max())
    
    # logging layer 1 activations - backbone  
    out = self.backbone(batch['img'] if isinstance(batch, dict) else batch) # []
    
    # out = self.layer1(img)
    # print('backbone out shape: ', len(out)) # 4
    # print('backbone out shape: ', out[0].shape) # torch.Size([24, 64, 22, 100])
    # print('backbone out shape: ', out[1].shape) # torch.Size([24, 128, 11, 50])
    # print('backbone out shape: ', out[2].shape) # torch.Size([24, 256, 6, 25])
    # print('backbone out shape: ', out[3].shape) # torch.Size([24, 512, 3, 13])
    # print('backbone out shape 0 0: ', out[0][0].shape) # torch.Size([24, 64, 22, 100])
    # c = self.makegrid(out[0][0], 4)

    layer1_activations = out[0]  # Get activations from layer 1
    layer2_activations = out[1]  # Get activations from layer 2
    layer3_activations = out[2]  # Get activations from layer 3
    layer4_activations = out[3]  # Get activations from layer 4

    # Get the features of the img_idx image in the batch
    img_fea_1 = layer1_activations[0]
    img_fea_2 = layer2_activations[0]
    img_fea_3 = layer3_activations[0]
    img_fea_4 = layer4_activations[0]

    print("img_fea_1.shape: ", img_fea_1.shape) # torch.Size([64, 22, 100])
    print("img_fea_1.unsqueeze(1).shape:  ", img_fea_1.unsqueeze(1).shape) # torch.Size([64, 1, 22, 100])

    # option 1: Grid view

    # grid_image_1 = tv.utils.make_grid(img_fea_1.unsqueeze(1), nrow=8, norm=True)
    # grid_image_2 = tv.utils.make_grid(img_fea_2.unsqueeze(1), nrow=8, norm=True)
    # grid_image_3 = tv.utils.make_grid(img_fea_3.unsqueeze(1), nrow=8, norm=True)
    # grid_image_4 = tv.utils.make_grid(img_fea_4.unsqueeze(1), nrow=8, norm=True)
    
    # Convert the tensor to a NumPy array and transpose the dimensions to (H, W, C)
    # numpy_image_1 = grid_image_1.permute(1, 2, 0).detach().cpu().numpy()
    # numpy_image_2 = grid_image_2.permute(1, 2, 0).detach().cpu().numpy()
    # numpy_image_3 = grid_image_3.permute(1, 2, 0).detach().cpu().numpy()
    # numpy_image_4 = grid_image_4.permute(1, 2, 0).detach().cpu().numpy()

    # option 2: Sequential view
    
    img_fea_1_norm = (img_fea_1 - img_fea_1.min()) / (img_fea_1.max() - img_fea_1.min())
    img_fea_2_norm = (img_fea_2 - img_fea_2.min()) / (img_fea_2.max() - img_fea_2.min())
    img_fea_3_norm = (img_fea_3 - img_fea_3.min()) / (img_fea_3.max() - img_fea_3.min())
    img_fea_4_norm = (img_fea_4 - img_fea_4.min()) / (img_fea_4.max() - img_fea_4.min())

    numpy_image_1 = img_fea_1_norm.unsqueeze(1).detach().cpu().numpy()[:num_fea_imgs]
    numpy_image_2 = img_fea_2_norm.unsqueeze(1).detach().cpu().numpy()[:num_fea_imgs]
    numpy_image_3 = img_fea_3_norm.unsqueeze(1).detach().cpu().numpy()[:num_fea_imgs]
    numpy_image_4 = img_fea_4_norm.unsqueeze(1).detach().cpu().numpy()[:num_fea_imgs]       

    print(" img_fea_1.min(): ",  img_fea_1.min()) # ? tensor(0., grad_fn=<MinBackward1>)
    print(" img_fea_1.max(): ",  img_fea_1.max()) # ? tensor(5.1708, grad_fn=<MaxBackward1>)

    # option 3: Grid view: input, b1, b2, b3, b4

    # numpy_img = img.detach().cpu() # C H W
    # numpy_image_1 = img_fea_1.unsqueeze(1).detach().cpu()[0] # 1 H W
    # numpy_image_2 = img_fea_2.unsqueeze(1).detach().cpu()[0]
    # numpy_image_3 = img_fea_3.unsqueeze(1).detach().cpu()[0]
    # numpy_image_4 = img_fea_4.unsqueeze(1).detach().cpu()[0] 

    # numpy_image_1 = numpy_image_1 * 255
    # numpy_image_2 = numpy_image_2 * 255
    # numpy_image_3 = numpy_image_3 * 255
    # numpy_image_4 = numpy_image_4 * 255

    # print("numpy_img: ", numpy_img.shape)
    # print("numpy_image_1: ", numpy_image_1.shape)

    # grid_one_img = tv.utils.make_grid([img, 
    #                                     numpy_image_1, 
    #                                     numpy_image_2, 
    #                                     numpy_image_3, 
    #                                     numpy_image_4],
    #                                   nrow=1, 
    #                                   norm=True)

    # self.logger.experiment.add_image(f"{group_path}", 
    #                               grid_one_img, 
    #                               dataformats='NCHW')
    
    # Scale the values to [0, 255] and cast the array to uint8 data type
    numpy_image_1 = (numpy_image_1 * 255).astype(np.uint8)
    numpy_image_2 = (numpy_image_2 * 255).astype(np.uint8)
    numpy_image_3 = (numpy_image_3 * 255).astype(np.uint8)
    numpy_image_4 = (numpy_image_4 * 255).astype(np.uint8)



    print('numpy_image_1: ', numpy_image_1.shape) # 64, 22, 100
    print('numpy_image_2: ', numpy_image_2.shape)
    print('numpy_image_3: ', numpy_image_3.shape)
    print('numpy_image_4: ', numpy_image_4.shape)

    self.logger.experiment.add_image(f"{group_path}/backbone_1", 
                                      numpy_image_1, 
                                      dataformats='NCHW') # group view: CHW
    self.logger.experiment.add_image(f"{group_path}/backbone_2", 
                                      numpy_image_2, 
                                      dataformats='NCHW')
    self.logger.experiment.add_image(f"{group_path}/backbone_3", 
                                      numpy_image_3, 
                                      dataformats='NCHW')
    self.logger.experiment.add_image(f"{group_path}/backbone_4", 
                                      numpy_image_4, 
                                      dataformats='NCHW')

    # print("CLRNet batch.keys: ", batch.keys()) # dict_keys(['img', 'lane_line', 'seg', 'meta'])
    # print("CLRNet batch.meta: ", batch['meta']) # {'full_img_path': ['data/CULane/driver_23_30frame/05161223_0545.MP4/04545.jpg', 'img2', ...]}
    # # batch['img'].shape = torch.Size([24, 3, 160, 400])

    # if self.aggregator:
    #   out[-1] = self.aggregator(out[-1])

    out = self.neck(out) # typle (l1, l2, l3)

    fpn_fea_1 = out[0]
    fpn_fea_2 = out[0]
    fpn_fea_3 = out[0]

    print("fpn_fea_1.shape: ", fpn_fea_1.shape)
    print("fpn_fea_2.shape: ", fpn_fea_2.shape)
    print("fpn_fea_3.shape: ", fpn_fea_3.shape)

    # img_idx = 0
    # img = batch['img'][img_idx]
    # img_full_path = batch['meta']['full_img_path'][img_idx]

    # # self.view(img, img_full_path)
    # for i in range(10):
    #   img = batch['img'][i]
    #   img_full_path = batch['meta']['full_img_path'][i]
    #   display_image_in_actual_size(img, img_full_path)

    # print("after view ")
    # exit(0)

    # # print('clrnet forward neck out: ', out)
    # print('clrnet forward batch: ', batch.keys())
    # if self.training:
    #   out = self.heads(out, batch=batch)
    # else:
    #   out = self.heads(out)

    # return out

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
    print("self.logger.experiment = ", self.logger.experiment)

    # print("batch['img'].shape: ", batch['img'].shape)

    # show_img(img, batch['meta']['full_img_path'][0])
    # self.logger.experiment.add_image(img_name, img, self.current_epoch)
    
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
    imgs = batch['img'] # torch.Size([24, 3, 160, 400])
    lane_line = batch['lane_line']
    seg = batch['seg']
    meta = batch['meta'] # {'full_img_path': [...]}

    # print('img len: ', len(imgs)) # 24
    # self.showActivations(batch_idx, batch)
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
