import torch.nn as nn
import torch

from clrnet.models.registry import NETS
from ..registry import build_backbones, build_necks, build_heads

class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        print('Init Detector')
        print('cfg = ', cfg)
        self.cfg = cfg
        self.backbone = build_backbones(cfg['backbone'])
        # self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg['neck']) if 'neck' in cfg else None
        self.heads = build_heads(cfg['head'])
    
    # def get_lanes(self):
    #     return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        # if self.aggregator:
        #     fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            print('Neck Detected')
            print('fea: ', fea)
            print('fea.shape: ', fea.shape)
            fea = self.neck(fea)

        output = self.heads(fea, batch=batch)
        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)

        return fea