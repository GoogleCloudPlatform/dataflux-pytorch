# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save/load function. This script does not rely on GPUs, TPUs or CPU Clusters and can be run directly on your machine. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

## Getting started

### Installation

```shell
pip install gcs-torch-dataflux gcsfs
```

### Configuration

First ensure you are running within a virtual python enviroment, make sure gcloud config project is set to correct value. Otherwise use the following command to set it 

```shell
gcloud config set project {PROJECT_ID}
```

Then set the command line variables.

`--ckpt_dir_path`: the location of where to save the checkpoints. 
`--steps`: the number of steps the model will take (the number of checkpoints created will be the same). The default value for `--steps` is 5. The benchmark will run `save_checkpoint` repeatedly and produce the average at the end, then run `load_checkpoint` on all the saved checkpoints and produce the average.

`--layers`: you can also optionally change the size of the model. The `--layers` argument will be passed into `nn.Transformer` for `num_encoder_layers` and `num_decoder_layers`. The default value for `--layers` is 100.

`--clear-kernel-cache`: this is an optional argument which when set clears kernel cache after saving the checkpoints. If benchmarking local filesystems or storage solutions that expose a filesystem interface (such as gcsfuse), this argument _must be set_ in order to get more accurate performance measurements. 

> [!NOTE]  
> The benchmarking script might have to be run as sudo if `--clear-kernel-cache` is set. It has no effect on machines that are not running on Linux or MacOS. 

### Dataflux Lightning Checkpoint

`--checkpoint=df_lightning`: This flag will measure timings for Dataflux Pytorch Checkpointing with lightning. This is the default execution mode.

`--checkpoint=no_df`: If you are not benchmarking Dataflux Lightning Checkpoint, this will disable dataflux checkpointing and use the default lightning checkpointing instead. 

`--checkpoint=df_async_lightning`: This flag will enable asynchronous calls to `save_checkpoint` which will optimize CPU/GPU utilization by making save calls non-blocking. 

`--checkpoint=no_lightning`: This flag will measure timings for Dataflux Pytorch Checkpointing without lightning.

`--disable-multipart`: This flag will disable multipart upload performance improvements. In most cases this will dramatically reduce the upload speed of checkpoint saves and is not recommended.

### Running

To execute this demo, run a command like the following:

```shell
python dataflux_pytorch/benchmark/checkpointing/singlenode/train.py --project=my-project --ckpt-dir-path=gs://my-bucket/path/to/dir/ --layers=10 --steps=5
```

The time will print out and the checkpoints can be viewed in GCS at the location passed in. A sample output is shown below.

```shell
$ python dataflux_pytorch/benchmark/checkpointing/singlenode/train.py
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
Average time to load one checkpoint: 62.54739844375839 seconds
```

## Results

The table below contains benchmarking times on saving checkpoints to GCS, the average save/load time is taken over 10 calls to save_checkpoint and load_checkpoint. The tests were done from a VM with 48vCPU, 192 GB RAM, 512 GB SSD located in `us-west1-a` zone. The GCS bucket was located in the same region, `us-west1`.


Dataflux's implementation of CheckpointIO for PyTorch Lightning is undergoing active development. The numbers below will be continuously updated to reflect the current state and performance of Dataflux's PyTorch Lightning checkpoint utility. These values are compared to `Default`, which refers to the default `TorchCheckpointIO` with fsspec/gcsfs.

### Checkpoint Save

| Checkpoint Type | Layers | Checkpoint File Size (MB) | Avg Checkpoint Save Time | Write Throughput (MB/s) |
| --- | --- | --- | --- | --- |
| Default   | 10      | 75.6    | 0.81    | 93.33   |
| Dataflux  | 10      | 75.6    | 0.56    | 135.00  |
| Default   | 100     | 298     | 2.87    | 103.98  |
| Dataflux  | 100     | 298     | 1.03    | 289.32  |
| Default   | 1,000   | 2,500   | 25.61   | 97.61   |
| Dataflux  | 1,000   | 2,500   | 6.25    | 400.00  |
| Default   | 10,000  | 24,200  | 757.10  | 31.96   |
| Dataflux  | 10,000  | 24,200  | 64.50   | 375.19  |


### Checkpoint Load

| Checkpoint Type | Layers | Checkpoint File Size (MB) | Avg Checkpoint Restore Time | Read Throughput (MB/s) |
| --- | --- | --- | --- | --- |
| Default   | 10      | 75.6    | 2.38      | 31.76   |
| Dataflux  | 10      | 75.6    | 0.51      | 148.24  |
| Default   | 100     | 298     | 1.69      | 176.33  |
| Dataflux  | 100     | 298     | 1.03      | 289.32  |
| Default   | 1,000   | 2,500   | 186.57    | 13.40   |
| Dataflux  | 1,000   | 2,500   | 14.77     | 169.26  |
| Default   | 10,000  | 24,200  | 2,093.52  | 11.56   |
| Dataflux  | 10,000  | 24,200  | 113.14    | 213.89  |
