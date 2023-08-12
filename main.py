import argparse
import yaml
import errno
import pytorch_lightning as pl
from pytorch_lightning.cli import ArgsType
import torch
import torchvision
# from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

from clrnet.cli import CLRNetCLI
from clrnet.engine import Runner
  
def main(args: ArgsType = None):
  cli = CLRNetCLI(Runner, 
                    run=True,
                    # args=args,
                    # trainer_class=TensorBoardLogger,
                    subclass_mode_model=True, 
                    subclass_mode_data=True,
                    parser_kwargs={
                      "default_config_files": ["/configs/clr_culane_resnet18.yaml"],
                      "parser_mode": "omegaconf",
                      "default_env": True},
                    save_config_kwargs={"config_filename": "config.yaml"},
                  )

if __name__ == '__main__':
  print(torch.__version__) # 2.0.1+cu118
  print(torchvision.__version__) # 0.15.2+cu118
  print(torch.__file__) # /usr/local/lib/python3.10/site-packages/torch/__init__.py
  print(torch.cuda.is_available()) # True
  print(torch.version.cuda) # 11.8
  main()