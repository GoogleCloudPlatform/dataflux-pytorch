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

import torch.nn as nn

from model.layers import DownsampleBlock, InputBlock, OutputLayer, UpsampleBlock


class Unet3D(nn.Module):
    def __init__(
        self, in_channels, n_class, normalization, activation, weights_init_scale=1.0
    ):
        super(Unet3D, self).__init__()

        filters = [32, 64, 128, 256, 320]
        self.filters = filters

        self.inp = filters[:-1]
        self.out = filters[1:]
        input_dim = filters[0]

        self.input_block = InputBlock(in_channels, input_dim, normalization, activation)

        self.downsample = nn.ModuleList(
            [
                DownsampleBlock(i, o, normalization, activation, index=idx)
                for idx, (i, o) in enumerate(zip(self.inp, self.out))
            ]
        )
        self.bottleneck = DownsampleBlock(
            filters[-1], filters[-1], normalization, activation, index=4
        )
        upsample = [
            UpsampleBlock(filters[-1], filters[-1], normalization, activation, index=0)
        ]
        upsample.extend(
            [
                UpsampleBlock(i, o, normalization, activation, index=idx + 1)
                for idx, (i, o) in enumerate(
                    zip(reversed(self.out), reversed(self.inp))
                )
            ]
        )
        self.upsample = nn.ModuleList(upsample)
        self.output = OutputLayer(input_dim, n_class)

        for name, v in self.named_parameters():
            if "weight" in name or "bias" in name:
                v.data *= float(weights_init_scale)

    def forward(self, x):
        x = self.input_block(x)
        outputs = [x]

        for downsample in self.downsample:
            x = downsample(x)
            outputs.append(x)

        x = self.bottleneck(x)

        for upsample, skip in zip(self.upsample, reversed(outputs)):
            x = upsample(x, skip)

        x = self.output(x)

        return x
