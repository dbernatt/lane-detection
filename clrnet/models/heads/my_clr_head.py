
import torch
import torch.nn as nn
import os.path as osp


class MyCLRHeadParams(object):
  def __init__(self, 
               sample_y,
               log_interval,
               num_classes,
               ignore_label,
               bg_weight,
               lr_update_by_epoch,
               iou_loss_weight,
               cls_loss_weight,
               xyt_loss_weight,
               seg_loss_weight):
    self.sample_y = sample_y
    self.log_interval = log_interval
    self.num_classes = num_classes
    self.ignore_label = ignore_label
    self.bg_weight = bg_weight
    self.lr_update_by_epoch = lr_update_by_epoch
    self.iou_loss_weight = iou_loss_weight
    self.cls_loss_weight = cls_loss_weight
    self.xyt_loss_weight = xyt_loss_weight
    self.seg_loss_weight = seg_loss_weight


class MyCLRHead(nn.Module):

  def __init__(self, cfg: MyCLRHeadParams,
                      num_points=72,
                      prior_feat_channels=64,
                      fc_hidden_dim=64,
                      num_priors=192,
                      num_fc=2,
                      refine_layers=3,
                      sample_points=36,
                      img_w=400,
                      img_h=160):
    print("Init MyCLRHead...")
    super(MyCLRHead, self).__init__()
    
    self.cfg = cfg
    self.img_w, self.img_h = img_w, img_h 
    self.num_points = num_points
    self.refine_layers = refine_layers

    print("Init MyCLRHead done.")

  def lane_prior(self):
    pass

  def forward(self, x, **kwargs):
    # x - ([],[],[]) features from fpn
    print("MyCLRHead forward...")
    print("MyCLRHead forward len(x): ", len(x))
    print("MyCLRHead forward x[0].shape: ", x[0].shape)
    print("MyCLRHead forward x[1].shape: ", x[1].shape)
    print("MyCLRHead forward x[2].shape: ", x[2].shape)
    # print("MyCLRHead forward **kwargs: ", kwargs['batch'].keys())
    batch =  kwargs['batch'] #  dict_keys(['img', 'lane_line', 'seg', 'meta'])
    out = {}

    # for i in range(self.refine_layers):
      
    
    return out 
























    