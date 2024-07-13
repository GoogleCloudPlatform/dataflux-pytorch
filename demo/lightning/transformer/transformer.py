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

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        output_idx = torch.multinomial(output.exp(), 1)
        print(to_string(inputs[0]), to_string(output_idx), sep="\n\n")


def to_string(input: torch.Tensor) -> str:
    s = ""
    for idx in input:
        idx = idx.item()
        s += dataset.dictionary.idx2word[idx] + " "
    return s


class ValidationDataset(data.Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise Exception
        words = [
            dataset.dictionary.word2idx[word]
            for word in "I am a language model , and I".split(" ")
        ]
        l = [torch.tensor(words).type(torch.int64)]
        return torch.cat(l), torch.zeros(8).type(torch.int64)


def main():
    dataloader = data.DataLoader(dataset)
    model = DatafluxTransformer(vocab_size=dataset.vocab_size)
    trainer = lightning.Trainer(fast_dev_run=1000)
    trainer.fit(model=model, train_dataloaders=dataloader)
    trainer.validate(model=model,
                     dataloaders=data.DataLoader(ValidationDataset()))


if __name__ == '__main__':
    main()
