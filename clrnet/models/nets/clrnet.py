import pytorch_lightning as pl
from torch import _cufft_get_plan_cache_size
import torch.functional as F

import torchvision.models as models

from ..registry import build_backbones


class CLRNet(pl.LightningModule):

  def __init__(self):
    super(CLRNet).__init__()
    self.batch_size = 16
    self.data_root = ''
    print('cfg: ', self.config)
    print(_cufft_get_plan_cache_size)
    self.backbone = models.resnet18(pretrained=True)
    # self.backbone = build_backbones(cfg)
    # self.neck = build_neck(cfg)
    # self.head = build_head(cfg)
    # self.loss = build_loss(cfg)

  def forward(self, x):
    out = self.backbone(x)
    return out

  def training_step(self, batch_idx, batch):
    x, y = batch
    y_hat = self.encoder(x)
    loss = F.cross_entropy(y_hat, y)
    return loss

  def configure_optimizers(self):
    # TO DO
    pass
