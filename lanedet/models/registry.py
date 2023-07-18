from lanedet.utils import Registry, build_from_cfg
import torch.nn as nn
# from lanedet.models import Detector

BACKBONES = Registry('backbones')
AGGREGATORS = Registry('aggregators')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')

# # Register the Detector class in the NETS registry
# NETS.register_module(Detector)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:

        b = build_from_cfg(cfg, registry, default_args)

        return b


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))


def build_necks(cfg):
    return build(cfg.necks, NECKS, default_args=dict(cfg=cfg))


def build_aggregator(cfg):
    return build(cfg.aggregator, AGGREGATORS, default_args=dict(cfg=cfg))


def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))


def build_necks(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))
