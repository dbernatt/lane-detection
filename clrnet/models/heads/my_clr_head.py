import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import math
import torch.nn.functional as F

from clrnet.models.utils.seg_decoder import SegDecoder
from clrnet.models.utils.roi_gather import ROIGather
from clrnet.models.losses.focal_loss import FocalLoss, SoftmaxFocalLoss
from clrnet.models.losses.accuracy import accuracy
from clrnet.ops import nms
from clrnet.models.utils.dynamic_assign import assign
from clrnet.models.losses.lineiou_loss import liou_loss
from clrnet.utils.lane import Lane

class MyCLRHeadParams(object):
  def __init__(self, 
               sample_y,
               log_interval,
               n_classes,
               ignore_label,
               bg_weight,
               lr_update_by_epoch,
               iou_loss_weight,
               cls_loss_weight,
               xyt_loss_weight,
               seg_loss_weight,
               n_priors=192,
               refine_layers=3,
               prior_fea_channels=64,
               sample_points=36,
               fc_hidden_dim=64):
    self.sample_y = sample_y
    self.log_interval = log_interval # 1000
    self.n_classes = n_classes # 4 + 1
    self.ignore_label = ignore_label # 255
    self.bg_weight = bg_weight # 0.4
    self.lr_update_by_epoch = lr_update_by_epoch # false
    self.iou_loss_weight = iou_loss_weight # 2.0
    self.cls_loss_weight = cls_loss_weight # 2.0
    self.xyt_loss_weight = xyt_loss_weight # 0.2
    self.seg_loss_weight = seg_loss_weight # 1.0

    self.n_priors = n_priors # 192
    self.refine_layers = refine_layers # 3
    self.prior_fea_channels = prior_fea_channels # 64
    self.sample_points = sample_points # 36
    self.fc_hidden_dim = fc_hidden_dim # 64


class MyCLRHead(nn.Module):

  def __init__(self, cfg: MyCLRHeadParams,
                      n_points=72,
                      n_fc=2,
                      img_w=400,
                      img_h=160):
    print("Init MyCLRHead...")
    super(MyCLRHead, self).__init__()
    
    self.cfg = cfg
    self.img_w, self.img_h = img_w, img_h # 400, 160 
    self.n_points = n_points # 72
    # n_offsets: the horizontal distance between the prediction and its ground truth
    self.n_offsets = self.n_points # 72
    self.n_strips = self.n_points - 1 # 71
    self.sample_points = self.cfg.sample_points # 36
    self.refine_layers = self.cfg.refine_layers # 3
    self.prior_fea_channels = self.cfg.prior_fea_channels # 64
    self.n_priors = self.cfg.n_priors # 192
    self.fc_hidden_dim = self.cfg.fc_hidden_dim

    # sample_x_indexes is a constant, which stores the random sample points x
    self.register_buffer(name='sample_x_indexs', 
                         tensor=(
                            torch.linspace(0, 1, 
                                           steps=self.sample_points, # 36 
                                           dtype=torch.float32) * self.n_strips # 71
                         ).long()
                        )

    """
      print(self.sample_x_indexs) 
      tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
         36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 71])
    """

    # noramlized sample_x_indexs or prior feature y points
    self.register_buffer(name='prior_feat_ys', 
                         tensor=torch.flip(
                          (1 - self.sample_x_indexs.float() / self.n_strips), # 71
                          dims=[-1]
                         )
                        )
    """
      print(self.prior_feat_ys)
        tensor([0.0000, 0.0423, 0.0704, 0.0986, 0.1268, 0.1549, 0.1831, 0.2113, 0.2394,
          0.2676, 0.2958, 0.3239, 0.3521, 0.3803, 0.4085, 0.4366, 0.4648, 0.4930,
          0.5211, 0.5493, 0.5775, 0.6056, 0.6338, 0.6620, 0.6901, 0.7183, 0.7465,
          0.7746, 0.8028, 0.8310, 0.8592, 0.8873, 0.9155, 0.9437, 0.9718, 1.0000])
    """

    # sample prior y points from 1 to 0
    self.register_buffer(name='prior_ys', 
                         tensor=torch.linspace(1, 0,
                                               steps=self.n_offsets, # 72
                                               dtype=torch.float32
                                       )
                        )
    """
      print(self.prior_ys)
      tensor([1.0000, 0.9859, 0.9718, 0.9577, 0.9437, 0.9296, 0.9155, 0.9014, 0.8873,
              0.8732, 0.8592, 0.8451, 0.8310, 0.8169, 0.8028, 0.7887, 0.7746, 0.7606,
              0.7465, 0.7324, 0.7183, 0.7042, 0.6901, 0.6761, 0.6620, 0.6479, 0.6338,
              0.6197, 0.6056, 0.5915, 0.5775, 0.5634, 0.5493, 0.5352, 0.5211, 0.5070,
              0.4930, 0.4789, 0.4648, 0.4507, 0.4366, 0.4225, 0.4085, 0.3944, 0.3803,
              0.3662, 0.3521, 0.3380, 0.3239, 0.3099, 0.2958, 0.2817, 0.2676, 0.2535,
              0.2394, 0.2254, 0.2113, 0.1972, 0.1831, 0.1690, 0.1549, 0.1408, 0.1268,
              0.1127, 0.0986, 0.0845, 0.0704, 0.0563, 0.0423, 0.0282, 0.0141, 0.0000])
    """

    # To Do
    self._init_prior_embeddings()
    init_priors, priors_on_featmap = self.generate_priors_from_embeddings() #None, None

    print("init_priors: ", init_priors)
    print("priors_on_featmap: ", priors_on_featmap)

    self.register_buffer(name='priors', tensor=init_priors)
    self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

    # generate xys for feature map
    self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                  self.cfg.n_classes,
                                  self.prior_fea_channels,
                                  self.refine_layers)

    reg_modules = list()
    cls_modules = list()
    lin_module = nn.ModuleList([nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                                nn.ReLU(inplace=True)])
    for _ in range(n_fc):
      reg_modules += [*lin_module]
      cls_modules += [*lin_module]
    self.reg_modules = nn.ModuleList(cls_modules)
    self.cls_modules = nn.ModuleList(cls_modules)

    self.roi_gather = ROIGather(self.prior_fea_channels, self.n_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)


    self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
    self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)


    # weights for each lane 
    weights = torch.ones(self.cfg.n_classes)
    weights[0] = self.cfg.bg_weight

    self.criterion = nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                     weight=weights)

    # init the weights here
    self.init_weights()

    print("Init MyCLRHead done.")

  def init_weights(self):
    # initialize heads
    for m in self.cls_layers.parameters():
        nn.init.normal_(m, mean=0., std=1e-3)

    for m in self.reg_layers.parameters():
        nn.init.normal_(m, mean=0., std=1e-3)

  def lane_prior(self):
    pass

  def generate_priors_from_embeddings(self):
    predictions = self.prior_embeddings.weight  # (num_prop, 3)

    # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
    priors = predictions.new_zeros(
        (self.n_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)

    priors[:, 2:5] = predictions.clone()
    priors[:, 6:] = (
        priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
        (self.img_w - 1) +
        ((1 - self.prior_ys.repeat(self.n_priors, 1) -
          priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
          self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
              1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

    # init priors on feature map
    priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

    return priors, priors_on_featmap

  def _init_prior_embeddings(self):
    # n_priors: 192
    # [start_y, start_x, theta] -> all normalize
    self.prior_embeddings = nn.Embedding(self.n_priors, 3)

    bottom_priors_nums = self.n_priors * 3 // 4
    left_priors_nums = self.n_priors // 8

    strip_size = 0.5 / (left_priors_nums // 2 - 1)
    bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
    for i in range(left_priors_nums):
        nn.init.constant_(self.prior_embeddings.weight[i, 0],
                          (i // 2) * strip_size)
        nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
        nn.init.constant_(self.prior_embeddings.weight[i, 2],
                          0.16 if i % 2 == 0 else 0.32)

    for i in range(left_priors_nums,
                    left_priors_nums + bottom_priors_nums):
        nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
        nn.init.constant_(self.prior_embeddings.weight[i, 1],
                          ((i - left_priors_nums) // 4 + 1) *
                          bottom_strip_size)
        nn.init.constant_(self.prior_embeddings.weight[i, 2],
                          0.2 * (i % 4 + 1))

    for i in range(left_priors_nums + bottom_priors_nums, self.n_priors):
        nn.init.constant_(
            self.prior_embeddings.weight[i, 0],
            ((i - left_priors_nums - bottom_priors_nums) // 2) *
            strip_size)
        nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
        nn.init.constant_(self.prior_embeddings.weight[i, 2],
                          0.68 if i % 2 == 0 else 0.84)

  def pool_prior_features(self, batch_features, num_priors, prior_xs):
    '''
    pool prior feature from feature map.
    Args:
        batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
    '''

    batch_size = batch_features.shape[0]

    prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
    prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
        batch_size, num_priors, -1, 1)

    prior_xs = prior_xs * 2. - 1.
    prior_ys = prior_ys * 2. - 1.
    grid = torch.cat((prior_xs, prior_ys), dim=-1)
    feature = F.grid_sample(batch_features, grid,
                            align_corners=True).permute(0, 2, 1, 3)

    feature = feature.reshape(batch_size * num_priors,
                              self.prior_fea_channels, self.sample_points,
                              1)
    return feature

  def forward(self, x, **kwargs):
    '''
    Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
    Each feature is a 4D tensor.
    Args:
        x: input features (list[Tensor])
    Return:
        prediction_list: each layer's prediction result
        seg: segmentation result for auxiliary loss
    '''
    # x - ([],[],[]) features from fpn
    print("MyCLRHead forward...")
    print("MyCLRHead forward len(x): ", len(x))
    print("MyCLRHead forward x[0].shape: ", x[0].shape)
    print("MyCLRHead forward x[1].shape: ", x[1].shape)
    print("MyCLRHead forward x[2].shape: ", x[2].shape)
    # print("MyCLRHead forward **kwargs: ", kwargs['batch'].keys())
    out = {}

    batch_features = list(x[len(x) - self.refine_layers:])
    batch_features.reverse()
    batch_size = batch_features[-1].shape[0]

    print("training: ", self.training)
    if self.training:
      self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()
    
    priors, priors_on_featmap = self.priors.repeat(batch_size, 1,
                                                  1), self.priors_on_featmap.repeat(
                                                      batch_size, 1, 1)

    predictions_lists = []

    # iterative refine
    prior_features_stages = []
    for stage in range(self.refine_layers):
        n_priors = priors_on_featmap.shape[1]
        prior_xs = torch.flip(priors_on_featmap, dims=[2])

        batch_prior_features = self.pool_prior_features(
            batch_features[stage], n_priors, prior_xs)
        prior_features_stages.append(batch_prior_features)

        # No roi gather
        fc_features = self.roi_gather(prior_features_stages,
                                      batch_features[stage], stage)

        fc_features = fc_features.view(n_priors, batch_size,
                                        -1).reshape(batch_size * n_priors,
                                                    self.fc_hidden_dim)

        cls_features = fc_features.clone()
        reg_features = fc_features.clone()
        for cls_layer in self.cls_modules:
            cls_features = cls_layer(cls_features)
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)

        cls_logits = self.cls_layers(cls_features)
        reg = self.reg_layers(reg_features)

        cls_logits = cls_logits.reshape(
            batch_size, -1, cls_logits.shape[1])  # (B, n_priors, 2)
        reg = reg.reshape(batch_size, -1, reg.shape[1])

        predictions = priors.clone()
        predictions[:, :, :2] = cls_logits

        predictions[:, :,
                    2:5] += reg[:, :, :3]  # also reg theta angle here
        predictions[:, :, 5] = reg[:, :, 3]  # length

        def tran_tensor(t):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, n_priors, 1) -
              tran_tensor(predictions[..., 2])) * self.img_h /
              torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

        prediction_lines = predictions.clone()
        predictions[..., 6:] += reg[..., 4:]

        predictions_lists.append(predictions)

        if stage != self.refine_layers - 1:
            priors = prediction_lines.detach().clone()
            priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

    if self.training:
        seg = None
        seg_features = torch.cat([
            F.interpolate(feature,
                          size=[
                              batch_features[-1].shape[2],
                              batch_features[-1].shape[3]
                          ],
                          mode='bilinear',
                          align_corners=False)
            for feature in batch_features
        ],
                                  dim=1)
        seg = self.seg_decoder(seg_features)
        output = {'predictions_lists': predictions_lists, 'seg': seg}
        return self.loss(output, kwargs['batch'])

    return predictions_lists[-1]

  def predictions_to_pred(self, predictions):
      '''
      Convert predictions to internal Lane structure for evaluation.
      '''
      self.prior_ys = self.prior_ys.to(predictions.device)
      self.prior_ys = self.prior_ys.double()
      lanes = []
      for lane in predictions:
          lane_xs = lane[6:]  # normalized value
          start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                      self.n_strips)
          length = int(round(lane[5].item()))
          end = start + length - 1
          end = min(end, len(self.prior_ys) - 1)
          # end = label_end
          # if the prediction does not start at the bottom of the image,
          # extend its prediction until the x is outside the image
          mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                      ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
          lane_xs[end + 1:] = -2
          lane_xs[:start][mask] = -2
          lane_ys = self.prior_ys[lane_xs >= 0]
          lane_xs = lane_xs[lane_xs >= 0]
          lane_xs = lane_xs.flip(0).double()
          lane_ys = lane_ys.flip(0)

          lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                      self.cfg.cut_height) / self.cfg.ori_img_h
          if len(lane_xs) <= 1:
              continue
          points = torch.stack(
              (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
              dim=1).squeeze(2)
          lane = Lane(points=points.cpu().numpy(),
                      metadata={
                          'start_x': lane[3],
                          'start_y': lane[2],
                          'conf': lane[1]
                      })
          lanes.append(lane)
      return lanes

  def loss(self,
            output,
            batch,
            cls_loss_weight=2.,
            xyt_loss_weight=0.5,
            iou_loss_weight=2.,
            seg_loss_weight=1.):
      if self.cfg.cls_loss_weight:
          cls_loss_weight = self.cfg.cls_loss_weight
      if self.cfg.xyt_loss_weight:
          xyt_loss_weight = self.cfg.xyt_loss_weight
      if self.cfg.iou_loss_weight:
          iou_loss_weight = self.cfg.iou_loss_weight
      if self.cfg.seg_loss_weight:
          seg_loss_weight = self.cfg.seg_loss_weight

      predictions_lists = output['predictions_lists']
      targets = batch['lane_line'].clone()
      cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
      cls_loss = 0
      reg_xytl_loss = 0
      iou_loss = 0
      cls_acc = []

      cls_acc_stage = []
      print("output len predictions_lists: ", len(output['predictions_lists']))
      for stage in range(self.refine_layers):
          predictions_list = predictions_lists[stage]
          for predictions, target in zip(predictions_list, targets):
              target = target[target[:, 1] == 1]

              if len(target) == 0:
                  # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                  cls_target = predictions.new_zeros(predictions.shape[0]).long()
                  cls_pred = predictions[:, :2]
                  cls_loss = cls_loss + cls_criterion(
                      cls_pred, cls_target).sum()
                  continue

              with torch.no_grad():
                  matched_row_inds, matched_col_inds = assign(
                      predictions, target, self.img_w, self.img_h)

              # classification targets
              cls_target = predictions.new_zeros(predictions.shape[0]).long()
              cls_target[matched_row_inds] = 1
              cls_pred = predictions[:, :2]

              # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
              reg_yxtl = predictions[matched_row_inds, 2:6]
              reg_yxtl[:, 0] *= self.n_strips
              reg_yxtl[:, 1] *= (self.img_w - 1)
              reg_yxtl[:, 2] *= 180
              reg_yxtl[:, 3] *= self.n_strips

              target_yxtl = target[matched_col_inds, 2:6].clone()

              # regression targets -> S coordinates (all transformed to absolute values)
              reg_pred = predictions[matched_row_inds, 6:]
              reg_pred *= (self.img_w - 1)
              reg_targets = target[matched_col_inds, 6:].clone()

              with torch.no_grad():
                  predictions_starts = torch.clamp(
                      (predictions[matched_row_inds, 2] *
                        self.n_strips).round().long(), 0,
                      self.n_strips)  # ensure the predictions starts is valid
                  target_starts = (target[matched_col_inds, 2] *
                                    self.n_strips).round().long()
                  target_yxtl[:, -1] -= (predictions_starts - target_starts
                                          )  # reg length

              # Loss calculation
              cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
              ) / target.shape[0]

              target_yxtl[:, 0] *= self.n_strips
              target_yxtl[:, 2] *= 180
              reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                  reg_yxtl, target_yxtl,
                  reduction='none').mean()

              iou_loss = iou_loss + liou_loss(
                  reg_pred, reg_targets,
                  self.img_w, length=15)

              # calculate acc
              cls_accuracy = accuracy(cls_pred, cls_target)
              cls_acc_stage.append(cls_accuracy)

          cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))

      # extra segmentation loss
      # seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),
      #                       batch['seg'].long())

      cls_loss /= (len(targets) * self.refine_layers)
      reg_xytl_loss /= (len(targets) * self.refine_layers)
      iou_loss /= (len(targets) * self.refine_layers)

      # loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
      #     + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

      loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
          + iou_loss * iou_loss_weight

      return_value = {
          'loss': loss,
          'loss_stats': {
              'loss': loss,
              'cls_loss': cls_loss * cls_loss_weight,
              'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
              # 'seg_loss': seg_loss * seg_loss_weight,
              'iou_loss': iou_loss * iou_loss_weight
          }
      }

      for i in range(self.refine_layers):
        return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

      return return_value


  def get_lanes(self, output, as_lanes=True):
      '''
      Convert model output to lanes.
      '''
      softmax = nn.Softmax(dim=1)

      decoded = []
      for predictions in output:
          # filter out the conf lower than conf threshold
          threshold = self.cfg.test_parameters.conf_threshold
          scores = softmax(predictions[:, :2])[:, 1]
          keep_inds = scores >= threshold
          predictions = predictions[keep_inds]
          scores = scores[keep_inds]

          if predictions.shape[0] == 0:
              decoded.append([])
              continue
          nms_predictions = predictions.detach().clone()
          nms_predictions = torch.cat(
              [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
          nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
          nms_predictions[...,
                          5:] = nms_predictions[..., 5:] * (self.img_w - 1)

          keep, num_to_keep, _ = nms(
              nms_predictions,
              scores,
              overlap=self.cfg.test_parameters.nms_thres,
              top_k=self.cfg.max_lanes)
          keep = keep[:num_to_keep]
          predictions = predictions[keep]

          if predictions.shape[0] == 0:
              decoded.append([])
              continue

          predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
          if as_lanes:
              pred = self.predictions_to_pred(predictions)
          else:
              pred = predictions
          decoded.append(pred)

      return decoded
























    