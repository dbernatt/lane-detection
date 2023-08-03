import torch
from pytorch_lightning.cli import LightningCLI, ArgsType
from jsonargparse import lazy_instance
from typing import Dict

from clrnet.models.nets import CLRNet
from clrnet.datasets import CULaneDataModule
from pytorch_lightning.loggers import TensorBoardLogger




class CLRNetCLI(LightningCLI):
    # def __init__(self, *args, **kwargs):
    #   super(CLRNetCLI, self).__init__(*args, **kwargs)
    #   print('Init CLRNetCLI...')
    #   # print('kwargs = ', **kwargs)
      
    #   super(CLRNetCLI, self).__init__(model_class=model_class,
    #                     datamodule_class=datamodule_class,
    #                     run=False)

    # def add_arguments_to_parser(self, parser):
    #   parser.link_arguments('trainer', 'model.init_args.trainer')
      # parser.set_defaults({"trainer.logger": lazy_instance(TensorBoardLogger, save_dir="/logs")})
      
    def before_instantiate_classes(self):
      print(self.config)
      print('Before init')
      config = self.config
      print('config = ', config)
      # data = config.get('data')

      # if data == None:
      #   raise ValueError('Missing data from config!')

      # data_type = data.get('data_type')
      
      # if data_type == None:
      #   raise ValueError('Missing data_type from data!')
      
      # assert isinstance(data_type, (str))

      # if data_type == data_types['CULane']:
      #   cli = CLRNetCLI(CLRNet, CULaneDataModule, args=None, run=False)
      # else:
      #   raise ValueError("'{}' data_type not found!".format(data_type))    

      # Override the run() method to handle your custom commands
    # def run(self):
    #     if self.config.subcommand is None:
    #         # Handle the case when no subcommand is provided
    #         self.print_help()
    #         return

    #     if self.config.subcommand == 'my_custom_command':
    #         self.my_custom_command_handler()

    #     # You can still use the base class's run() method to handle built-in commands
    #     super().run()

    def add_arguments_to_parser(self, parser):
      print('parser = ', parser)
      # parser.add_optimizer_args(torch.optim.Adam)
      # parser.link_arguments('trainer', 'model.init_args.trainer')
      parser.link_arguments("data.init_args.cfg.init_args.img_w", "model.init_args.heads.init_args.img_w")
      parser.link_arguments("data.init_args.cfg.init_args.img_h", "model.init_args.heads.init_args.img_h")
      # parser.link_arguments("data.init_args.cfg.init_args.num_classes", "model.init_args.heads.init_args.num_classes")
      # parser.link_arguments("data.init_args.cfg.init_args.batch_size", "model.init_args.batch_size")
      # parser.link_arguments("data.init_args.cfg", "model.init_args.cfg")

      # parser.link_arguments("data.batch_size", "model.batch_size", apply_on='instantiate')

    # def add_arguments_to_parser(self, parser):
      # parser.add_subcommands(required=False, dest='fit')
      # parser.set_defaults({"data.data_type": lazy_instance(CULaneDataset, type='CULane')})
