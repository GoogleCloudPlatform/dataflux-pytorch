import torch
from torch import optim, nn, utils, Tensor
from torch.utils import data
from dataflux_pytorch import dataflux_mapstyle_dataset
import lightning.pytorch as pl

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--project", type=str)
#     parser.add_argument("--bucket", type=str)
#     parser.add_argument("--prefix", type=str)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--num-workers", type=int, default=0)
#     parser.add_argument("--no-dataflux", type=bool, default=False)
#     parser.add_argument("--batch-size", type=int, default=100)
#     parser.add_argument("--sleep-per-step", type=float, default=1.3604)
#     parser.add_argument("--prefetch-factor", type=int, default=2)
#     parser.add_argument("--threads-per-worker", type=int, default=2)
#     parser.add_argument("--max-composite-object-size",
#                         type=int,
#                         default=100000000)
#     parser.add_argument("--log-level", type=str, default="ERROR")
#     parser.add_argument("--retry-timeout", type=float, default=300.0)
#     parser.add_argument("--retry-initial", type=float, default=1.0)
#     parser.add_argument("--retry-multiplier", type=float, default=1.2)
#     parser.add_argument("--retry-maximum", type=float, default=45.0)
#     return parser.parse_args()

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(10, 6), nn.ReLU(), nn.Linear(6, 3))
decoder = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 10))


class TextDemoModel(pl.LightningModule):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss


def format_data(content_in_bytes):
    return content_in_bytes


def main():
    config = dataflux_mapstyle_dataset.Config(prefix="fineweb/sample/10BT/01")
    my_model = TextDemoModel(encoder, decoder)

    dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
        project_name="gcs-tess",
        bucket_name="mirvine-benchmark-central1",
        config=config,
        data_format_fn=format_data,
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainer = pl.Trainer(limit_train_batches=2, max_epochs=1)
    trainer.fit(model=my_model, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
