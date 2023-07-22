from clrnet.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')


def build(cfg, registry):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry)


def build_backbones(cfg):
    return build(cfg, BACKBONES)


def build_necks(cfg):
    return build(cfg, NECKS)


def build_heads(cfg):
    return build(cfg, HEADS)


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS)


def build_net(cfg):
    return build(cfg, NETS)