from clrnet.utils import Registry, build_from_cfg

import torch as nn
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import random
# from mmcv.parallel import collate

DATASETS = Registry('datasets')
PROCESS = Registry('process')


def build(registry, cfg, default_cfg):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(registry, cfg_, default_cfg) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(registry, cfg, default_cfg)


def build_dataset(split_cfg, cfg):
    return build(DATASETS, split_cfg, default_cfg=cfg)


def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(worker_init_fn, seed=cfg.seed)

    # samples_per_gpu = cfg.batch_size // cfg.gpus
    data_loader = nn.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=False,
        # collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn)

    return data_loader