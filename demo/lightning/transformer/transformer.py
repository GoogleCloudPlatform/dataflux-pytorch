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

import lightning
import torch
from lightning.pytorch import demos
from torch.nn import functional
from torch.utils import data

dataset = demos.WikiText2()


class DatafluxTransformer(lightning.LightningModule):

    def __init__(self, vocab_size):
        super().__init__()
        self.model = demos.Transformer(
            vocab_size=vocab_size,
            nhead=6,
            nlayers=6,
            nhid=192,
            ninp=192,
        )

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))

        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)


def main():
    dataloader = data.DataLoader(dataset)
    model = DatafluxTransformer(vocab_size=dataset.vocab_size)
    trainer = lightning.Trainer(fast_dev_run=100)
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == '__main__':
    main()
