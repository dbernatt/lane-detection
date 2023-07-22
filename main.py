import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.cli import ArgsType, LightningCLI
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

from clrnet.models import CLRNet
from clrnet.datamodules import CULaneDataModule

def cli_main(args: ArgsType = None):
  print('running cli_main()...')
  cli = LightningCLI(CLRNet, 
                    CULaneDataModule,
                    args=args,
                    run=False)

def parse_args():
  parser = argparse.ArgumentParser(description='Train arguments for detection')
  parser.add_argument("--config", 
                      default='./clrnet/configs/clr_culane_resnet18.yaml', 
                      help="Config file path")
  args = parser.parse_args()
  print('parse args=', args)
  return args

if __name__ == '__main__':
  # args = parse_args()
  cli_main()