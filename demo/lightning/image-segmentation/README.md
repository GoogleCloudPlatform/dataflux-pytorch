<!-- Talk about code organization - what classes

Shift focus to how dataflux fits in the pytorch lightning way of things

How to run the code locally on a GPU/CPU machine
-->

# Image Segmentation Model with Pytorch Lightning and Dataflux

This demo walks you through the various pieces of implementing an image segmentation model using the 
Pytorch Lightning framework with Dataflux supporting the data loading phase with the goal of showing
how easily Dataflux can be plugged into the Pytorch Lightning framework.

## Code Organization

This demo implements a variant of the Unet3D model which is a Convolutional Nerual Net (CNN). The implementation of this model is spread across two files -- `layers.py` and `unet3d.py`.
The `Unet3D` class, which inherits pytorch's `nn.Module` is then wrapped in `Unet3DLightning` which inherits Pytorch Lightning's `LightningModule`.


```python
class Unet3DLightning(pl.LightningModule):

    def __init__(self, flags):
        super().__init__()
        self.flags = flags
        self.model = Unet3D(1,
                            3,
                            normalization=flags.normalization,
                            activation=flags.activation)
        ...

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        ...

    def training_step(self, train_batch, batch_idx):
        ...
```
