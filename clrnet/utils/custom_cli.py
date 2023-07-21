import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

class CustomCLI(LightningCLI):
    def __init__(self, model, data_module):
        super().__init__(model, data_module)
        print('Init CustomCLI...')
        print('cfg = ', self.config)

    # Override the run() method to handle your custom commands
    def run(self):
        if self.config.subcommand is None:
            # Handle the case when no subcommand is provided
            self.print_help()
            return

        if self.config.subcommand == 'my_custom_command':
            self.my_custom_command_handler()

        # You can still use the base class's run() method to handle built-in commands
        super().run()

    # Implement your custom command handler
    def my_custom_command_handler(self):
        # Your custom logic here
        print("Running my custom command")

        # Access the configuration values
        config = self.config

        # You can also access your model and data module instances here if needed
        model = self.get_model()
        data_module = self.get_datamodule()

        # You have full control over how to handle your custom command and configuration
        # You can start training, run custom validation, or perform any other tasks
