# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save function. This script does not rely on GPUs, TPUs or CPU Clusters and can be run directly on your machine. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

## Getting started

### Installation

```shell
pip install gcs-torch-dataflux gcsfs
```

### Configuration

First ensure you are running within a virtual python enviroment, then set the enviroment variables.

`CKPT_DIR_PATH` is the location of where to save the checkpoints. `STEPS` is the number of steps the model will take (the number of checkpoints created will be the same). The default value for `STEPS` is 5.

```shell
export CKPT_DIR_PATH=`gs://path/to/directory/`
export STEPS=5
```

You can also optionally change the size of the model. The `LAYERS` variable will be passed into `nn.Transformer` for `num_encoder_layers` and `num_decoder_layers`. The default value for `LAYERS` is 100.

```shell
export LAYERS=1000
```

### Dataflux Lightning Checkpoint

If you are benchmarking Dataflux Lightning Checkpoint, save information regarding your project and bucket, and make sure to enable the flag by setting it to `1`.

```shell
export PROJECT=`YOUR_PROJECT_NAME`
export BUCKET=`YOUR_BUCKET_NAME`
export DATAFLUX_CKPT=1
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
/usr/local/google/home/divyarawal/dataflux_dev/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default

  | Name  | Type        | Params | Mode
----------------------------------------------
0 | model | Transformer | 19.8 M | train
----------------------------------------------
19.8 M    Trainable params
0         Non-trainable params
19.8 M    Total params
79.189    Total estimated model params size (MB)
/usr/local/google/home/divyarawal/dataflux_dev/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=47` in the `DataLoader` to improve performance.
Epoch 0:   0%|                                                                                                               | 5/59674 [00:11<39:37:59,  0.42it/s, v_num=2]`Trainer.fit` stopped: `max_steps=5` reached.
Epoch 0:   0%|                                                                                                               | 5/59674 [00:11<39:38:05,  0.42it/s, v_num=2]
Time to train over 5 steps: 14.197517395019531 seconds
Time to save one checkpoint: 2.075364589691162 seconds
```

## Results

The table below contains benchmarking times writing checkpoints to GCS.

Dataflux's implementation of CheckpointIO for PyTorch Lightning is undergoing active development. The numbers below will be continuously updated to reflect the current state and performance of Dataflux's PyTorch Lightning checkpoint utility. These values are compared to `Default`, which refers to fsspec.

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Type</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Layers</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Size (MB) per step</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Steps</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Train Time (s)</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Single Checkpoint Save Time (s)</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Write Throughput (MB/s)</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"> Default
   </td>
   <td style="background-color: #d9d9d9">10
   </td>
   <td style="background-color: #d9d9d9">75.6
   </td>
   <td style="background-color: #d9d9d9">5
   </td>
   <td style="background-color: #d9d9d9">13.25
   </td>
   <td style="background-color: #d9d9d9">1.64
   </td>
   <td style="background-color: #d9d9d9">46.09
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   <td style="background-color: #f3f3f3">5
   </td>
   <td style="background-color: #f3f3f3">14.08
   </td>
   <td style="background-color: #f3f3f3">2.07
   </td>
   <td style="background-color: #f3f3f3">36.52
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9">Default
   </td>
   <td style="background-color: #d9d9d9">100
   </td>
   <td style="background-color: #d9d9d9">298
   </td>
   <td style="background-color: #d9d9d9">5
   </td>
   <td style="background-color: #d9d9d9">36.55
   </td>
   <td style="background-color: #d9d9d9">5.21
   </td>
   <td style="background-color: #d9d9d9">57.20
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux
   </td>
   <td style="background-color: #f3f3f3">100
   </td>
   <td style="background-color: #f3f3f3">298
   </td>
   <td style="background-color: #f3f3f3">5
   </td>
   <td style="background-color: #f3f3f3">44.07
   </td>
   <td style="background-color: #f3f3f3">7.04
   </td>
   <td style="background-color: #f3f3f3">42.32
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"> Default
   </td>
   <td style="background-color: #d9d9d9">1000
   </td>
   <td style="background-color: #d9d9d9">2500
   </td>
   <td style="background-color: #d9d9d9">5
   </td>
   <td style="background-color: #d9d9d9">266.16
   </td>
   <td style="background-color: #d9d9d9">39.14
   </td>
   <td style="background-color: #d9d9d9">63.87
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux
   </td>
   <td style="background-color: #f3f3f3">1000
   </td>
   <td style="background-color: #f3f3f3">2500
   </td>
   <td style="background-color: #f3f3f3">5
   </td>
   <td style="background-color: #f3f3f3">349.19
   </td>
   <td style="background-color: #f3f3f3">53.71
   </td>
   <td style="background-color: #f3f3f3">46.55
   </td>
  </tr>
</table>
