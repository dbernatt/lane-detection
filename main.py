import argparse
import yaml
import errno
import pytorch_lightning as pl
from pytorch_lightning.cli import ArgsType
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
  main()