import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import DictType, _cufft_get_plan_cache_size
import torch.functional as F
import torchvision.models as models
from collections.abc import MutableMapping

from clrnet.models.heads import CLRHead
from clrnet.models.nets import Detector

# class RunnerParams(object):
#   def __init__(self, backbone, neck, heads = CLRHead, *args, **kwargs):
#     print('Init RunnerParams...')
#     print('backbone: ', backbone)
#     print('heads: ', heads)
#     print('neck: ', neck)

#     self.backbone = backbone
#     self.neck = neck
#     self.heads = heads

# class NetParams(object):
#   def __init__(self, cfg, *args, **kwargs):
#     print('Init DetectorParams...')
#     self.cfg = cfg

class Runner(pl.LightningModule):

  # def __init__(self, backbone, heads, neck = None, detector: object = Detector,  *args, **kwargs):
  # def __init__(self, backbone, neck, heads = CLRHead, *args, **kwargs):
  def __init__(self, cfg, *args, **kwargs):
    print('Init Runner...')
    super(Runner, self).__init__()
    # print('detector: ', detector)
    # self.backbone = backbone
    # self.neck = neck
    # self.heads = heads
    # self.detector = detector
    # self.detector = detector
    # print('detector.cfg: ', detector.cfg)
    
    self.net = cfg.net(cfg.net.cfg)

  # def _get_reconstruction_loss(self, batch):
  #   """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
  #   x, _ = batch  # We do not need the labels
  #   x_hat = self.forward(x)
  #   loss = F.mse_loss(x, x_hat, reduction="none")
  #   loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
  #   return loss

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    print('Runner training step...')
    print('batch_idx: ', batch_idx)
    print('batch: ', batch)
    print('batch key list: ', list(batch.keys()))
    # img, lane_line, seg, meta = batch
    img = batch['img']
    lane_line = batch['lane_line']
    seg = batch['seg']
    meta = batch['meta']
    print('img len: ', len(img))
    output = self.net(img)
    print('training step after net')
    print('output: ', output)
    loss = output['loss'].sum()
    # loss = nn.CrossEntropyLoss(y_hat, seg)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.6e-3)
