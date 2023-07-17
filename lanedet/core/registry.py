from torch import nn 
from lanedet.utils import Registry, build_from_cfg

def build(cfg, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, default_args)


def build_trainer(cfg):
    return build(cfg.trainer, default_args=dict(cfg=cfg))


def build_evaluator(cfg):
    return build(cfg.evaluator, default_args=dict(cfg=cfg))