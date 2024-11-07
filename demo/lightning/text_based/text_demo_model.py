"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import torch

import lightning.pytorch as pl
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from torch import optim, nn

# This is a pre-trained tokenizer designed for text-based use cases. It is configured to expect a maximum
# input size of 512. A user can speciy any tokenizer they wish, or design their own as long as the output
# parameter count matches the input of the encoder (in this case 512).
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
INPUT_SIZE = 512


class TextDemoModel(pl.LightningModule):
    """
    This model subclasses the base LightningModule and configures itself as an autoencoder.
    """

    def __init__(self, encoder, decoder):
        """
        The encoder and decoder sequences to use are passed directly into this model.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def configure_optimizers(self):
        """
        This function sets the loss parameter (lr) which determines the rate at which the model "learns".
        The parameters pass is a direct-passthrough of any existing parameters from the LightingModule.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        The training step defines the train loop. It performs the basic functions of an autoencoder, where
        loss is calculated by taking the Mean Squared Error of input against output. This call is independent
        from forward.
        """
        batch_size = len(train_batch)
        # Data must be converted into a tensor format for input into the training sequence.
        x = torch.zeros(batch_size, INPUT_SIZE)
        for index, row in enumerate(train_batch):
            tokenized = tokenize(row)
            x[index, :len(tokenized)] = tokenized
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default.
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss


def tokenize(sequence):
    """
    This helper function translates the string input into a sequence of tokens with a max value of 512.
    Values must be converted into tensor floats in preparation for encoding. The tokenizer model
    in use is the `Bert Base Cased` pre-trained model.
    """
    # This tokenizer auto-truncates to a value of 512.
    tokens = tokenizer.tokenize(sequence, truncation=True)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor(ids).to(torch.float)


def format_data(raw_bytes):
    """
    The data comes in as a large parquet_file that must be read into a buffer and parsed
    into the correct parquet format.
    """
    reader = pa.BufferReader(raw_bytes)
    table = pq.ParquetFile(reader)
    return table
