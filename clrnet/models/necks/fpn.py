import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels # [128, 256, 512]
        self.out_channels = out_channels # 64
        self.num_ins = len(in_channels) # 3
        self.num_outs = num_outs # 3

        if end_level == -1:
            self.backbone_end_level = self.num_ins # 3
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level # 0
        self.end_level = end_level # 3

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i], # [128, 256, 512]
                out_channels, # 64
                kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, # 64
                                  out_channels, # 64
                                  kernel_size=3,
                                  padding=1)  

        self.lateral_convs.append(l_conv) # 3 Conv2ds
        self.fpn_convs.append(fpn_conv) # 3 Conv2ds


    def forward(self, inputs):
        """Forward function."""
        print('FPN forward')
        # in_channels = [128, 256, 512]
        assert len(inputs[1]) >= len(self.in_channels) 
        print('inputs: ', inputs)
        print('inputs shape: ', inputs.shape)
        print('inputs len 0: ',  len(inputs[0]))
        print('in_channels: ', self.in_channels)
        print('len inputs-in_channels: ', len(inputs), len(self.in_channels))

        # remove inputes from the beginning if inputs length > num of in_channels
        # if len(inputs[0]) > len(self.in_channels):
        #   for _ in range(len(inputs) - len(self.in_channels)):
        #       del inputs[0]

        # build laterals from start level (=0)
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i + self.start_level]))
        
        """
        Build top-down path (laterals)
        
          The top-down path combines higher-resolution features 
          from the i-th level with the corresponding lower-resolution features 
          from the (i-1)-th level.

          This process is repeated iteratively, gradually 
          increasing the resolution of feature maps as they move up the pyramid, 
          creating a multi-scale representation.
        """

        used_backbone_levels = len(laterals) # 3
        for i in range(used_backbone_levels - 1, 0, -1): # 2 1 0
            prev_lateral_shape = laterals[i - 1].shape[2:] # [N, C, H, W] -> [H, W]
            """
              F.interpolate(input, size=None, scale_factor=None, mode='nearest',...): 
              Down/up samples the input to either the given size 
              or the given scale_factor
              The input dimensions are interpreted in the form: 
                mini-batch x channels x [optional depth] x [optional height] x width.
            """
            laterals[i - 1] += F.interpolate(laterals[i],
                                              size=prev_lateral_shape,
                                              **self.upsample_cfg)

        
        # build outputs
        # part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
          outs.append(self.fpn_convs[i](laterals[i]))

        # part 2: add extra levels
        # if self.num_outs > len(outs): # 3 > 3
        #     # use max pool to get more levels on top of outputs
        #     # (e.g., Faster R-CNN, Mask R-CNN)
        #     if not self.add_extra_convs:
        #       for i in range(self.num_outs - used_backbone_levels):
        #           outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        
        return tuple(outs)
           

 

        


