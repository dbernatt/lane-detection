import os
import argparse
import pytorch_lightning as pl
from lanedet.utils import Config
from lanedet.core.runner import Runner
from lanedet.datasets.registry import build_dataloader


def main():
    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
    # str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)
    cfg.seed = args.seed


    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    runner = Runner(cfg)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=cfg.epochs)

    if hasattr(args, 'validate'):
        raise NotImplementedError('Validation is not implemented yet.')
    elif hasattr(args, 'test'):
        raise NotImplementedError('Test is not implemented yet.')
    else:
        runner.train()
        # trainer.fit(runner)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train arguments for detection')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
