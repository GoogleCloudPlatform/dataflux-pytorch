import os
import time
import torch

from typing import Tuple
from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2, Transformer
from lightning.pytorch import LightningModule

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint

class LightningTransformer(LightningModule):
    def __init__(self, vocab_size: int = 33278, nlayers: int = 100) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
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


def main(project: str, bucket: str, ckpt_dir_path: str, save_only_latest: bool, model_size: int = 100, steps: int = 5):
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)
    model = LightningTransformer(vocab_size=dataset.vocab_size, nlayers=model_size)
    dataflux_ckpt = DatafluxLightningCheckpoint(project_name=project,bucket_name=bucket)
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
        plugins=[dataflux_ckpt],
        callbacks=[checkpoint_callback],
        min_epochs=4,
        max_epochs=5,
        max_steps=5,
        accelerator="cpu",
    )
    start = time.time()
    trainer.fit(model, dataloader)
    end = time.time()
    print(end-start)

if __name__ == "__main__":

    DEFAULT_SIZE = 100
    DEFAULT_STEPS = 5
    size = int(os.getenv("CHECKPOINT_SIZE'",DEFAULT_SIZE))
    steps = int(os.getenv("STEPS'",DEFAULT_STEPS))

    main(
        os.getenv("PROJECT"),
        os.getenv("BUCKET"),
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("SAVE_ONLY_LATEST") == "1",
        size,
        steps,
    )
