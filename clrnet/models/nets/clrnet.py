import pytorch_lightning as pl
from torch import _cufft_get_plan_cache_size
import torch.functional as F

import torchvision.models as models

from ..registry import build_backbones, build_necks


class Encoder:
  def __init__(self):
    super().__init__()


class Decoder:
  def __init__(self):
    super().__init__()

class CLRNet(pl.LightningModule):

  def __init__(self, 
                backbone, 
                neck, 
                var = 10,
                batch_size = 16,
                ):
    super(CLRNet, self).__init__()
    print('Init CLRNet...')
    print(var)
    self.training = True
    self.batch_size = batch_size

    print('backbone = ', backbone)
    self.backbone = build_backbones(backbone)

    print('neck = ', neck)   
    self.neck = build_necks(neck)

  def forward(self, x):
    out = self.backbone(x['img'] if isinstance(x, dict) else x)

    if self.neck:
        features = self.neck(features)

    return out

  def training_step(self, batch_idx, batch):
    x, y = batch
    y_hat = self.encoder(x)
    loss = F.cross_entropy(y_hat, y)
    return loss

  def configure_optimizers(self):
    # TO DO
    pass
