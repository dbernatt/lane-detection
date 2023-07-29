from torch import nn
from clrnet.utils import Registry, build_from_cfg

TRAINER = Registry('trainer')
EVALUATOR = Registry('evaluator')


def build(registry, cfg, default_cfg=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(registry, cfg_, default_cfg) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(registry, cfg, default_cfg)


def build_trainer(cfg):
    return build(TRAINER, cfg.trainer, default_cfg=dict(cfg=cfg))


def build_evaluator(cfg):
    return build(EVALUATOR, cfg.evaluator, default_cfg=dict(cfg=cfg))