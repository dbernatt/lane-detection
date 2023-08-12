import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.functional as F

import torchvision.models as models
import torchvision as tv
from clrnet.models.heads import MyCLRHead
from clrnet.models.necks import FPN
from clrnet.models.backbones import ResNetWrapper

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sys import exit
from clrnet.utils.visualization import display_image_in_actual_size, show_img
from pytorch_lightning.loggers import TensorBoardLogger

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

        print("Init Detector Done.")

    # def get_lanes(self):
    #   print("Detector get_lanes...")
    #   return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
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