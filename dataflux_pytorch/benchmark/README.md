# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save/load function. This script does not rely on GPUs, TPUs or CPU Clusters and can be run directly on your machine. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

## Getting started

### Installation

```shell
pip install gcsfs
pip install .
```

### Configuration

First ensure you are running within a virtual python enviroment, make sure gcloud config project is set to correct value. Otherwise use the following command to set it 

```shell
gcloud config set project {PROJECT_ID}
```

Then set the enviroment variables.

`CKPT_DIR_PATH` is the location of where to save the checkpoints. `STEPS` is the number of steps the model will take (the number of checkpoints created will be the same). The default value for `STEPS` is 5. The benchmark will run `save_checkpoint` and `load_checkpoint` repeatedly and produce the average at the end for each. 

```shell
export CKPT_DIR_PATH=`gs://path/to/directory/`
export STEPS=5
```

You can also optionally change the size of the model. The `LAYERS` variable will be passed into `nn.Transformer` for `num_encoder_layers` and `num_decoder_layers`. The default value for `LAYERS` is 100.

```shell
export LAYERS=1000
```

### Dataflux Lightning Checkpoint

If you are benchmarking Dataflux Lightning Checkpoint, save information regarding your project and make sure to enable the flag by setting it to `1`.

You can also control whether Transfer Manager is used to do multipart-upload and multipart-download of the checkpoints by setting the `USE_TRANSFER_MANAGER` flag. Note that the default is true for the checkpointer, but will only be used here if you set it:

```shell
export PROJECT=`YOUR_PROJECT_NAME`
export DATAFLUX_CKPT=1
export USE_TRANSFER_MANAGER=1
```

### Running

Run the script.

```shell
python dataflux_pytorch/benchmark/lightning_checkpoint_benchmark.py
```

The time will print out and the checkpoints can be viewed in GCS at the location passed in. A sample output is shown below.

```shell
$ python dataflux_pytorch/benchmark/lightning_checkpoint_benchmark.py
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default

  | Name  | Type        | Params | Mode
----------------------------------------------
0 | model | Transformer | 658 M  | train
----------------------------------------------
658 M     Trainable params
0         Non-trainable params
658 M     Total params
2,634.181 Total estimated model params size (MB)
.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=47` in the `DataLoader` to improve performance.
Epoch 0:   0%|                                                                                                            | 10/59674 [12:17<1221:29:06,  0.01it/s, v_num=5]`Trainer.fit` stopped: `max_steps=10` reached.
Epoch 0:   0%|                                                                                                            | 10/59674 [12:17<1221:29:09,  0.01it/s, v_num=5]
Average time to save one checkpoint: 58.68560411930084 seconds
Average time to load one checkpoint: 63.66342296364783 seconds
```

## Results

The table below contains benchmarking times on saving checkpoints to GCS. For 10/100-layer checkpoints the average save/load times is taken over 20 calls to save_checkpoint or load_checkpoint, and 10 calls for 1000 layers. The tests were done from a VM with 48vCPU, 192 GB RAM, 512 GB SSD located in `us-west1-a` zone. The GCS bucket was located in the same region, `us-west1`.


Dataflux's implementation of CheckpointIO for PyTorch Lightning is undergoing active development. The numbers below will be continuously updated to reflect the current state and performance of Dataflux's PyTorch Lightning checkpoint utility. These values are compared to `TorchCheckpointIO`, which refers to PyTorch Lightning's default [TorchCheckpointIO](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.io.TorchCheckpointIO.html) implementation. A value is provided for direct writes and reads with GCS using gcsfs via the default implementation.

### Saving Checkpoints

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Type</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Layers</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Size (MB) per step</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Average Checkpoint Save Time</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Write Throughput (MB/s)</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">TorchCheckpointIO (gcsfs)
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">0.89
   </td>
   <td style="background-color: #f3f3f3">84.9
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Disabled)
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">0.80
   </td>
   <td style="background-color: #f3f3f3">94.5
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Enabled)
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">0.55
   </td>
   <td style="background-color: #f3f3f3">137.5
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">TorchCheckpointIO (gcsfs)
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">3.38
   </td>
   <td style="background-color: #d9d9d9">88.2
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">Dataflux (Transfer Manager Disabled)
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">3.16
   </td>
   <td style="background-color: #d9d9d9">94.3
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">Dataflux (Transfer Manager Enabled)
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">1.19
   </td>
   <td style="background-color: #d9d9d9">250.4
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">TorchCheckpointIO (gcsfs)
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">32.44
   </td>
   <td style="background-color: #f3f3f3">77.1
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Disabled)
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">27.48
   </td>
   <td style="background-color: #f3f3f3">91.0
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Enabled)
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">7.74
   </td>
   <td style="background-color: #f3f3f3">323.0
   </td>
  </tr>
</table>

### Loading Checkpoints

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Type</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Layers</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Size (MB) per step</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Average Checkpoint Load Time</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Read Throughput (MB/s)</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">TorchCheckpointIO (gcsfs)
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">2.38
   </td>
   <td style="background-color: #f3f3f3">31.8
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Disabled)
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">2.37
   </td>
   <td style="background-color: #f3f3f3">31.9
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Enabled)
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">0.38
   </td>
   <td style="background-color: #f3f3f3">198.9
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">TorchCheckpointIO (gcsfs)
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">12.00
   </td>
   <td style="background-color: #d9d9d9">24.8
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">Dataflux (Transfer Manager Disabled)
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">8.44
   </td>
   <td style="background-color: #d9d9d9">35.3
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">Dataflux (Transfer Manager Enabled)
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">1.72
   </td>
   <td style="background-color: #d9d9d9">173.3
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">TorchCheckpointIO (gcsfs)
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">174.40
   </td>
   <td style="background-color: #f3f3f3">14.33
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Disabled)
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">77.92
   </td>
   <td style="background-color: #f3f3f3">32.1
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux (Transfer Manager Enabled)
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">12.87
   </td>
   <td style="background-color: #f3f3f3">194.3
   </td>
  </tr>
</table>
