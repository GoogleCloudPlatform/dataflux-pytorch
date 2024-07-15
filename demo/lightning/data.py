import torch
import lightning.pytorch as pl

from dataflux_pytorch import dataflux_mapstyle_dataset


class Unet3DDataModule(pl.LighitningDataModule):
    def __init__(self, gcs_bucket):
        self.data_dir = gcs_bucket + "/images"
        self.transform = 1
    
    def prepare_data(self):
        
        pass

    def setup(self, state="train"):
        # Init DatafluxPytTrain object 
        # Pass transform function as argument - modify DatafluxPyTrain constructor
        pass

    def train_dataloader(self):
        # Init dataflux dataloader
        # Wrap it in torch.DataLoader
        pass
