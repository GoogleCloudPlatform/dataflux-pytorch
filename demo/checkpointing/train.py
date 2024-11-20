import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import torch
import torch.nn as nn
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

from dataflux_pytorch.dataflux_checkpoint import DatafluxCheckpoint


class TextClassifier(nn.Module):
    """Basic implementation of a Text Classifier model."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)
        return logits.type(torch.FloatTensor)


class CheckpointHelper:
    """A wrapper class around DatafluxCheckpoint that allows the checkpoint
    save to be called asynchronously."""

    def __init__(self, project_name: str, bucket_name: str):
        self._ckpt = DatafluxCheckpoint(project_name, bucket_name)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def save_checkpoint(self,
                        path: str,
                        state_dict: Dict[str, torch.Tensor],
                        use_async: bool = False):

        def _save():
            # TODO: error handling
            with self._ckpt.writer(path) as writer:
                torch.save(state_dict, writer)

        if use_async:
            self._executor.submit(_save)
        else:
            _save()

    def teardown(self):
        self._executor.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configuration for single-node async checkpoint demo.")
    # User config
    parser.add_argument("--project",
                        type=str,
                        required=True,
                        help="GCS project ID.")
    parser.add_argument("--bucket",
                        type=str,
                        required=True,
                        help="GCS bucket for saving checkpoints.")
    parser.add_argument("--use-async",
                        action="store_true",
                        default=False,
                        help="If checkpoint save should use async.")

    # Hyper parameters
    parser.add_argument("--embedding-dim",
                        type=int,
                        default=100,
                        help="The size of each embedding vector.")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=128,
                        help="The number of features in the hidden state.")
    parser.add_argument("--output-dim",
                        type=int,
                        default=35,
                        help="The size of each output sample.")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="How many samples per batch to load.")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=10,
                        help="The number of full passes through training.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help=
        "The size of the optimizer steps taken during the training process.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    Typical usage example:

      python3 -u demo/checkpointing/train.py \
      --project=<gcs_project_id> \
      --bucket=<bucket_name> \
      --embedding-dim=100 \
      --use-async
    """

    # Load dataset and create a Dataloader
    dataset = WikiText2()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # Create Model
    model = TextClassifier(dataset.vocab_size, args.embedding_dim,
                           args.hidden_dim, args.output_dim)
    ckpt = CheckpointHelper(args.project, args.bucket)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    losses = []
    save_checkpoint_times = []
    total_time = time.time()

    # Training workload
    for epoch in range(args.num_epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().numpy())
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {loss:.4f}")

        # Save checkpoint
        save_time = time.time()
        ckpt_path = f'single-node/async/checkpoint_{epoch}.pth'
        ckpt.save_checkpoint(ckpt_path, model.state_dict(), args.use_async)
        save_checkpoint_times.append(time.time() - save_time)

    # Clean up and report measurements.
    ckpt.teardown()
    print(f'Average checkpoint save time: '
          f'{statistics.mean(save_checkpoint_times):.4f} seconds '
          f'(stdev {statistics.stdev(save_checkpoint_times):.4f})')
    duration = int(time.time() - total_time)
    print(f'Total run time: {duration//60}m{duration%60}s')


if __name__ == '__main__':
    args = parse_args()
    main(args)
