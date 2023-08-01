import torch.nn as nn
from torchvision.transforms import CenterCrop, Normalize

from ..registry import PROCESS

@PROCESS.register_module
class MyGenerateLaneLine(object):

  def __init__(self, transforms=None, cfg=None, training=True):
    self.cfg = cfg  
    self.transforms = []
    self.trainging = training

  def __call__(self, sample):
    pass