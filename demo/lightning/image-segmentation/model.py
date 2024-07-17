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
from unet3d import Unet3D
from losses import DiceCELoss


class Unet3DLightning(pl.LightningModule):

    def __init__(self, flags):
        self.flags = flags
        # Init instance of Unet3D
        self.model = Unet3D(1,
                            3,
                            normalization=flags.normalization,
                            activation=flags.activation)
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            use_softmax=True,
            layout=flags.layout,
            include_background=flags.include_background,
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters,
            lr=self.flags.learning_rate,
            momentum=self.flags.momentum,
            nesterov=True,
            weight_decay=self.flags.weight_decay,
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        predictions = self.model(images)
        loss = self.loss_fn(predictions, labels)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
