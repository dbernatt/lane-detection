import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import _cufft_get_plan_cache_size
import torch.functional as F

import torchvision.models as models
from clrnet.models.heads import CLRHead
from clrnet.models.necks import FPN
from clrnet.models.backbones import ResNetWrapper

class Encoder:
  def __init__(self):
    super().__init__()

class Decoder:
  def __init__(self):
    super().__init__()

def build_optimizer(cfg, net):
    params = []
    cfg_cp = cfg.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in dir(torch.optim):
        raise ValueError("{} is not defined.".format(cfg_type))

    _optim = getattr(torch.optim, cfg_type)
    return _optim(net.parameters(), **cfg_cp)

class CLRNet(pl.LightningModule):

  def __init__(self, backbone: ResNetWrapper, 
                     neck: FPN | None, 
                     heads: CLRHead):
    
    super().__init__()
    print('backbone = ', backbone)
    self.backbone = backbone
    self.save_hyperparameters(ignore=['backbone', 'neck', 'heads'])

    print()
    print('neck = ', neck)   
    self.neck = neck
    
    print('head = ', heads)
    self.heads = heads
    self.aggregator = None

    print('Init CLRNet Done.')

  def forward(self, batch):
    print('forward batch:', batch)
    out = {}
    out = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

    if self.aggregator:
      out[-1] = self.aggregator(out[-1])

    print('clrnet f out: ', out)

    if self.neck:
      out = self.neck(out)

    if self.training:
      out = self.heads(out, batch=batch)
    else:
      out = self.heads(out)

    return out

  def training_step(self, batch, batch_idx):
    print('CLRNet training step...')
    print('batch_idx: ', batch_idx)
    print('batch: ', batch)
    print('batch key list: ', list(batch.keys()))
    img = batch['img']
    lane_line = batch['lane_line']
    seg = batch['seg']
    meta = batch['meta']
    print('img len: ', len(img))
    output = self(batch)
    print("output['loss']: ", output['loss'])
    return
    loss = output['loss'].sum()
    print('training_step: after y_hat = self(batch)')
    print('loss: ', loss)
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
