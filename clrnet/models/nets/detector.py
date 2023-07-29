import torch.nn as nn
import torch

class Detector(nn.Module):
  def __init__(self, cfg):
    super(Detector, self).__init__()
    print('Init Detector')
    print('cfg = ', cfg)
    self.cfg = cfg
    self.backbone = cfg.backbone
    self.neck = cfg.neck
    self.heads = cfg.heads
    print('neck: ', self.neck)
    print('heads: ', self.heads)
    # self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
    
    # def get_lanes(self):
    #     return self.heads.get_lanes(output)

  def forward(self, batch):
    print('Detector forward')
    print('batch: ', batch)
    output = {}
    fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

    # if self.aggregator:
    #     fea[-1] = self.aggregator(fea[-1])

    if self.neck:
        print('Neck Detected')
        print('fea: ', fea)
        print('fea.shape: ', fea.shape)
        fea = self.neck(fea)

    print('heads: ', self.heads)
    output = self.heads(fea, batch=batch)
    if self.training:
        output = self.heads(fea, batch=batch)
    else:
        output = self.heads(fea)

    return fea