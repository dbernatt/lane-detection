from clrnet.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')


def build(registry, cfg):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(registry, cfg_) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(registry, cfg)


def build_backbones(cfg):
    return build(BACKBONES, cfg)


def build_necks(cfg):
    return build(NECKS, cfg)


def build_heads(cfg):
    return build(cfg, HEADS)


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS)


def build_net(cfg):
    return build(NETS, cfg)