import argparse
import yaml
import errno
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, ArgsType
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

from clrnet.models import CLRNet
from clrnet.datasets import CULaneDataset
from clrnet.cli import CLRNetCLI

data_types = [
  'CULane'
]

def main():
  # args = parse_args()
  # config = load_cfg(config)
  # assert isinstance(config['data']['type'], (str))
  # data_type = config['data']['type']

  cli = CLRNetCLI(run = False)

  # if data_type == data_types[0]:
  #   cli = CLRNetCLI(CLRNet, CULaneDataModule)
  # else:
  #   raise ModuleNotFoundError('This data type does not exists!')

def load_cfg(filename):
  try:
    with open(filename, 'r') as f:
      config = yaml.safe_load(f)
    return config
  except IOError as e:
    if e.errno == errno.ENOENT:
        raise e('File not found')
    elif e.errno == errno.EACCES:
        raise e('Permission denied')
    else:
        raise e
  return None

def parse_args():
  parser = argparse.ArgumentParser(description='Train arguments for detection')
  print('aaa')
  parser.add_argument("--config", 
                      default='./clrnet/configs/clr_culane_resnet18.yaml', 
                      help="Config file path")

  args = parser.parse_args()
  print('parse args=', args)
  return args

if __name__ == '__main__':
  main()