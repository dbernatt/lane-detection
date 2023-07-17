from os import sys, path
import argparse
from lanedet.core import Core

def main():
  args = parse_args()
  # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
  #   str(gpu) for gpu in args.gpus)
  cfg = utils.Config.fromfile(args.config)
  
  core = Core(cfg)

def parse_args():
  parser = argparse.ArgumentParser(description='Train arguments for detection')
  parser.add_argument('config', help='train config file path')
  # parser.add_argument('--gpus', nargs='+', type=int, default='0')

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  main()
