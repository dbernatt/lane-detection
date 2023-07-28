import argparse
import yaml
import errno
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, ArgsType
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

# from clrnet.models import CLRNet
from clrnet.datasets import CULaneDataModule
from clrnet.cli import CLRNetCLI, Runner

data_types = {
  "CULane": "clrnet.datasets.CULaneDataModule"
}

def main():
  parser = argparse.ArgumentParser()
  args = parse_args()
  config = load_cfg(args.config)
  print('config = ', config)
  data = config.get('data')
  
  if data == None:
    raise ValueError('Missing data from config!')

  data_type = data.get('class_path')
  
  if data_type == None:
    raise ValueError('Missing class_path from data!')
  
  assert isinstance(data_type, (str))

  if data_type == data_types['CULane']:
    cli = CLRNetCLI(Runner, 
                    CULaneDataModule, 
                    run=True,
                    subclass_mode_model=True, 
                    subclass_mode_data=True,
                    parser_kwargs={
                      "default_config_files": ["/configs/clr_culane_resnet18.yaml"],
                      "parser_mode": "omegaconf"
                    })
    
  else:
    raise ValueError("'{}' data_type not found!".format(data_type))

def load_cfg(filename):
  try:
    with open(filename, 'r') as f:
      config = yaml.safe_load(f)
    return config
  except IOError as e:
    if e.errno == errno.ENOENT:
        raise e('Config file not found')
    elif e.errno == errno.EACCES:
        raise e('Permission denied to open config file')
    else:
        raise e
  return None

def parse_args():
  parser = argparse.ArgumentParser(description='Train arguments for detection')
  
  parser.add_argument("--config", 
                      default='./clrnet/configs/clr_culane_resnet18.yaml', 
                      help="Config file path")
  parser.add_argument("fit", 
                    nargs='?',
                    help="Set cli stage [fit, predict, test]",
                    )
  parser.add_argument("test", 
                    nargs='?',
                    help="Set cli stage [fit, predict, test]",
                    )
                    
  parser.add_argument("predict", 
                    nargs='?',
                    help="Set cli stage [fit, predict, test]",
                    )
  args = parser.parse_args()
  print('parse args=', args)
  return args

if __name__ == '__main__':
  main()