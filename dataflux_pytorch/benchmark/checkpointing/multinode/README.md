# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save/load function. The multinode benchmarking script does require a GPU. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

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


### Running

To run the script use the following command. 

```shell
python dataflux_pytorch/benchmark/checkpointing/multinode/train.py --project=my-project --ckpt-dir-path=gs://my-bucket/path/to/dir/ --layers=10 --steps=5
```

Optionally you can pass, min_epochs, max_epochs and max_steps for save & restore.
