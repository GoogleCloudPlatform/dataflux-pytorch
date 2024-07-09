# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save function. This script does not rely on GPUs or CPU Clusters and can be run directly on your machine. The script runs the `WikiText2` PyTorch Lightining demo code with some modifications.

First ensure you are running within a virtual python enviroment, then set the enviroment variables.

`CKPT_DIR_PATH` is the location of where to save the checkpoints. `STEPS` is the number of steps the model will take (the number of checkpoints created will be the same). The default value for `STEPS` is 5.

```shell
export CKPT_DIR_PATH=`gs://path/to/directory/`
export STEPS=5
```

You can also optionally change the size of the model. The size variable will be passed into `nn.Transformer` for `num_encoder_layers` and `num_decoder_layers`. The default value for size is 100.

```shell
export CHECKPOINT_SIZE=`1000`
```

If you are benchmarking Dataflux Lightning Checkpoint, save information regarding your project and bucket, and make sure to enable the flag by setting it to `1`.

```shell
export PROJECT=`YOUR_PROJECT_NAME`
export BUCKET=`YOUR_BUCKET_NAME`
export DATAFLUX_CKPT=1
```

Run the script.

```shell
python dataflux_pytorch/benchmark/lightning_checkpoint_benchmark.py
```

The time will print out and the checkpoints can be viewed in GCS at the location passed in.
