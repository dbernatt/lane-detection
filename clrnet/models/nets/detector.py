import torch
import torch.nn as nn

from clrnet.models.heads import MyCLRHead
from clrnet.models.necks import FPN
from clrnet.models.backbones import ResNetWrapper


class Detector(nn.Module):
    def __init__(self,
                     backbone: ResNetWrapper,
                     neck: FPN | None,
                     heads: MyCLRHead):
        print("Init Detector...")
        super(Detector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads
        self.aggregator = None
        self.training = True

        print("Init Detector Done.")

    # def get_lanes(self):
    #   print("Detector get_lanes...")
    #   return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        print("batch seg: ", batch['seg'].shape)
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)

        return output