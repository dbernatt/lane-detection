import torch.nn as nn
import torch.nn.functional as F


class SegDecoder(nn.Module):
  '''
  Optionaly seg decoder
  '''
  def __init__(self,
                img_h,
                img_w,
                n_class,
                prior_fea_channels=64,
                refine_layers=3):
      super().__init__()
      # %10 input will be set to zero during forward propagation
      # prevents overfitting
      self.dropout = nn.Dropout2d(0.1) 
      self.conv = nn.Conv2d(prior_fea_channels * refine_layers, n_class, 1)
      self.img_h = img_h
      self.img_w = img_w

  def forward(self, x):
      x = self.dropout(x)
      x = self.conv(x)
      x = F.interpolate(x,
                        size=[self.img_h, self.img_w],
                        mode='bilinear',
                        align_corners=False)
      return x