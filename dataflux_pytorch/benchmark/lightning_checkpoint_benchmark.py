import os
import time
from typing import Tuple

import torch
from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import Transformer, WikiText2
from lightning.pytorch.plugins.io import TorchCheckpointIO
from torch import Tensor
from torch.utils.data import DataLoader

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint


class LightningTransformer(LightningModule):

    def __init__(self, vocab_size: int = 33278, nlayers: int = 100) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def prepare_data(self) -> None:
        WikiText2(download=True)

    def train_dataloader(self) -> DataLoader:
        dataset = WikiText2()
        return DataLoader(dataset)

def main(project: str, bucket: str, ckpt_dir_path: str, save_only_latest: bool, dataflux_ckpt: bool, layers: int = 100, steps: int = 5):
    """Checkpoints a PyTorch Ligthning demo model to GCS using gcsfs or DatafluxLightningCheckpoint.

    This function utilizes PyTorch Lightning to checkpoint the WikiText2 dataset. It
    takes in information regarding the gcs location to save the checkpoints, the type of
    checkpoint, and other configuration variables. Default this function runs on
    gcsfs to write PyTorch Ligthtning checkpoints, TorchCheckpointIO. If dataflux_ckpt
    is enabled the Trainer will be passed a DatafluxLightningCheckpoint, which is an
    implementation of the CheckpointIO interface, as a plugin.

    Typical usage example:

      Run DatafluxLightningCheckpoint over 10 steps:

      project = 'test-project'
      bucket = 'test-bucket'
      ckpt_dir_path = 'gs://path/to/dir/'
      save_only_latest = False
      dataflux_ckpt = True
      layers = 1000
      steps = 10

      main(project=project, bucket=bucket, save_only_latest=save_onlylatest,
      dataflux_ckpt=dataflux_ckpt, layers=layers, steps=steps)

      Run gcsfs over 10 steps:

      ckpt_dir_path = 'gs://path/to/dir/'
      save_only_latest = False
      dataflux_ckpt = False
      layers = 1000
      steps = 10

      main(project=project, bucket=bucket, save_only_latest=save_onlylatest,
      dataflux_ckpt=dataflux_ckpt, layers=layers, steps=steps)
    """
    if steps < 1:
        raise ValueError("Steps need to greater than 0.")

    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)
    model = LightningTransformer(vocab_size=dataset.vocab_size, nlayers=layers)
    ckpt = TorchCheckpointIO()
    if dataflux_ckpt:
        ckpt = DatafluxLightningCheckpoint(project_name=project,
                                           bucket_name=bucket)
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        plugins=[ckpt],
        callbacks=[checkpoint_callback],
        min_epochs=4,
        max_epochs=5,
        max_steps=1,
        accelerator="cpu",
    )
    trainer.fit(model, dataloader)

    start = time.time()
    for i in range(steps):
        trainer.save_checkpoint(os.path.join(ckpt_dir_path,f'ckpt_{i}.ckpt'))
    end = time.time()
    print("Average time to save one checkpoint: " + str((end-start)/steps) + " seconds")

if __name__ == "__main__":

    DEFAULT_LAYERS = 100
    DEFAULT_STEPS = 5
    layers = int(os.getenv("LAYERS", DEFAULT_LAYERS))
    steps = int(os.getenv("STEPS", DEFAULT_STEPS))

    main(
        os.getenv("PROJECT"),
        os.getenv("BUCKET"),
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("SAVE_ONLY_LATEST") == "1",
        os.getenv("DATAFLUX_CKPT") == "1",
        layers,
        steps,
    )
