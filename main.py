import argparse
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

from clrnet.utils import Config, CustomCLI
from clrnet.models import CLRNet
from clrnet.datamodules import CULaneDataModule

def cli_main(args = None):
  print('be')
  cli = CustomCLI(CLRNet, 
                    CULaneDataModule)
  config = cli.config
  print('config = ', config)
  batch_size = config['trainer']['num_nodes']
  print('num_nodes = ', batch_size)

  # cfg = Config.load_from_yaml(args.config)
  # print('cfg=',cfg)
  
  # data_module = CULaneDataModule(args.config)
  # trainer = pl.Trainer(fast_dev_run=True, max_epochs=cfg.epochs)

  # if hasattr(args, 'validate'):
  #   trainer.validate(model, data_module=data_module)
  # elif hasattr(args, 'test'):
  #   trainer.test(model, data_module=data_module)
  # else:
  #   trainer.train(model, data_module=data_module)


def parse_args():
  parser = argparse.ArgumentParser(description='Train arguments for detection')
  
  # parser.add_argument("--config", 
  #                     default='./clrnet/configs/clr_culane_resnet18.yaml', 
  #                     help="Config file path")
  # parser.add_argument("--stage", choices=['fit', 'validate', 'test'], default='fit', required=True, help="Choose mode for the model")
  
  args = parser.parse_args()
  
  return args

if __name__ == '__main__':
  # config = parse_args()
  cli_main()