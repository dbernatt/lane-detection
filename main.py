import argparse
from lanedet.utils import Config
from lanedet.core.runner import Runner
from lanedet.datasets.registry import build_dataloader
import pytorch_lightning as pl


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    runner = Runner(cfg)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=cfg.epochs)

    if args.validate:
        raise NotImplementedError('Validation is not implemented yet.')
    elif args.test:
        raise NotImplementedError('Test is not implemented yet.')
    else:
        trainer.fit(runner)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train arguments for detection')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
