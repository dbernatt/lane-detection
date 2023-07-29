import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import _cufft_get_plan_cache_size
import torch.functional as F

import torchvision.models as models

from ..registry import build_backbones, build_necks, build_heads


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
                head,
                batch_size = 16
                ):
    super(CLRNet, self).__init__()
    print('Init CLRNet...')
    self.training = True
    self.batch_size = batch_size

    print('backbone = ', backbone)
    self.backbone = build_backbones(backbone)

    print('neck = ', neck)   
    self.neck = build_necks(neck)

    self.net = nn.Sequential(self.backbone, self.neck)

    # print('head = ', head)
    # self.heads = build_heads(head)

  # def resume(self):
  #   if not self.cfg.load_from and not self.cfg.finetune_from:
  #       return
  #   load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

  # def train_epoch(self, epoch, train_loader):
  #     max_iter = len(train_loader)
  #     for i, data in enumerate(train_loader):
  #         if self.recorder.step >= self.cfg.total_iter:
  #             break
  #         date_time = time.time() - end
  #         self.recorder.step += 1
  #         data = self.to_cuda(data)
  #         output = self.net(data)
  #         self.optimizer.zero_grad()
  #         loss = output['loss'].sum()
  #         loss.backward()
  #         self.optimizer.step()
  #         if not self.cfg.lr_update_by_epoch:
  #             self.scheduler.step()
  #         batch_time = time.time() - end
  #         end = time.time()
  #         self.recorder.update_loss_stats(output['loss_stats'])
  #         self.recorder.batch_time.update(batch_time)
  #         self.recorder.data_time.update(date_time)

  #         if i % self.cfg.log_interval == 0 or i == max_iter - 1:
  #             lr = self.optimizer.param_groups[0]['lr']
  #             self.recorder.lr = lr
  #             self.recorder.record('train')

  # def train(self):
  #     self.recorder.logger.info('Build train loader...')
  #     train_loader = build_dataloader(self.cfg.dataset.train,
  #                                     self.cfg,
  #                                     is_train=True)

  #     self.recorder.logger.info('Start training...')
  #     start_epoch = 0
  #     if self.cfg.resume_from:
  #         start_epoch = resume_network(self.cfg.resume_from, self.net,
  #                                       self.optimizer, self.scheduler,
  #                                       self.recorder)
  #     for epoch in range(start_epoch, self.cfg.epochs):
  #         self.recorder.epoch = epoch
  #         self.train_epoch(epoch, train_loader)
  #         if (epoch +
  #                 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
  #             self.save_ckpt()
  #         if (epoch +
  #                 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
  #             self.validate()

  def forward(self, x):
    print('forward x:', x)
    out = self.backbone(x['img'] if isinstance(x, dict) else x)
    out = self.neck(out)
    # out = self.heads(out, batch=batch)
    return out

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
    y_hat = self.net(img)
    print('training step after net')
    loss = nn.CrossEntropyLoss(y_hat, seg)
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
    return torch.optim.Adam(self.parameters(), lr=0.6e-3)
