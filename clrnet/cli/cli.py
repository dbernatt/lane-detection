from pytorch_lightning.cli import LightningCLI, ArgsType
from jsonargparse import lazy_instance

from clrnet.models.nets import CLRNet
from clrnet.datasets import CULaneDataModule

class CLRNetCLI(LightningCLI):
    def __init__(self, run: bool = True):
      print('Init CLRNetCLI...')
      super(CLRNetCLI, self).__init__(model_class=CLRNet,
                        datamodule_class=CULaneDataModule,
                        run=run)

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")

    # def add_arguments_to_parser(self, parser):
    #   parser.set_defaults({"data.data_type": lazy_instance(CULaneDataset, type='CULane')})
