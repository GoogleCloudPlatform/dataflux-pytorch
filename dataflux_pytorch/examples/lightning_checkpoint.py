from lightning import Trainer
from lightning.pytorch.demos import WikiText2, LightningTransformer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint

def main(project: str, bucket: str, ckpt_dir_path: str, save_only_latest: bool):
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    dataflux_ckpt = DatafluxLightningCheckpoint(project_name=project, bucket_name=bucket)
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1 if save_only_latest else -1,
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
        max_steps=3,
        accelerator="cpu",
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    import os

    main(
        os.getenv("PROJECT"),
        os.getenv("BUCKET"),
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("SAVE_ONLY_LATEST") == "1",
    )
