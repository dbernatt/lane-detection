from pytorch_lightning.cli import LightningCLI, ArgsType
from jsonargparse import lazy_instance

from clrnet.models.nets import CLRNet
from clrnet.datasets import CULaneDataset

class CLRNetCLI(LightningCLI):
    def __init__(self, *kwargs):
      print('Init CLRNetCLI...')
      print(*kwargs)
      super().__init__(*kwargs)

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")

    # def add_arguments_to_parser(self, parser):
    #   parser.set_defaults({"data.data_type": lazy_instance(CULaneDataset, type='CULane')})
