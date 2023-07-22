from clrnet.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg, BACKBONES, default_args=dict(cfg=cfg))


def build_necks(cfg):
    return build(cfg, NECKS, default_args=dict(cfg=cfg))


def build_heads(cfg):
    return build(cfg, HEADS, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg, NETS, default_args=dict(cfg=cfg))