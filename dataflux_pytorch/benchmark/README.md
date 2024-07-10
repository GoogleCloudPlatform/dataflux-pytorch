# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save function. This script does not rely on GPUs or CPU Clusters and can be run directly on your machine. The script runs the `WikiText2` PyTorch Lightining demo code with some modifications.

### Getting started

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

The time will print out and the checkpoints can be viewed in GCS at the location passed in.

### Results
<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Type</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Size</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Checkpoint Size (MB) per step</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Steps</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Time (s)</strong>
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
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux
   </td>
   <td style="background-color: #f3f3f3">10
   </td>
   <td style="background-color: #f3f3f3">75.6
   </td>
   </td>
   <td style="background-color: #f3f3f3">5
   </td>
   </td>
   <td style="background-color: #f3f3f3">14.08
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
  </tr>
  <tr>
   <td style="background-color: #f3f3f3">Dataflux
   </td>
   <td style="background-color: #f3f3f3">100
   </td>
   <td style="background-color: #f3f3f3">298
   </td>
   </td>
   <td style="background-color: #f3f3f3">5
   </td>
   </td>
   <td style="background-color: #f3f3f3">44.07
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
  </tr>
</table>
