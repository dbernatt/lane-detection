import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import DictType, _cufft_get_plan_cache_size
import torch.functional as F
import torchvision.models as models
from collections.abc import MutableMapping

from clrnet.models.nets import Detector

class RunnerParams(object):
  def __init__(self, backbone, heads = None, neck = None, *args, **kwargs):
      print('Init RunnerParams...'
      )
      print('backbone: ', backbone)
      print('heads: ', heads)
      print('neck: ', neck)

      self.backbone = backbone
      self.neck = neck
      self.heads = heads

class Runner(pl.LightningModule):

  def __init__(self, cfg: RunnerParams, processes, batch_size = 16):
    super(Runner, self).__init__()
    print('Init Runner...')
    self.cfg = cfg
    self.processes = processes
    print('backbone = ', cfg.backbone)
    self.net = Detector(self.cfg)

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    print('CLRNet training step...')
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
