import torch.nn as nn
from lanedet.utils import Registry, build_from_cfg


def build(cfg, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, default_args=dict(cfg=cfg))


def build_necks(cfg):
    return build(cfg.necks, default_args=dict(cfg=cfg))


def build_aggregator(cfg):
    return build(cfg.aggregator, default_args=dict(cfg=cfg))


def build_heads(cfg):
    return build(cfg.heads, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, default_args=dict(cfg=cfg))


def build_neck(cfg):
    return build(cfg.neck, default_args=dict(cfg=cfg))
