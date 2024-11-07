# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save/load function. The multinode benchmarking script does require a GPU. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

## Getting started

### Setup
 
#### Install requirements
Create a Python virtual environment and activate it
```
python3 -m venv .venv
source .venv/bin/activate
```
Run the following commands from the root of the repo to install the packages needed by the benchmarking code

```
pip install -r dataflux_pytorch/benchmark/requirements.txt`
pip install .
```

#### GCP Config


```shell
gcloud config set project {PROJECT_ID}
```

## Run


### Environment variables

Set the following environment variables by updating the deployment file if deploying on a GKE cluster, or by running `set ENV_VAR=value` if running locally on your machine.

1. The following environment variables must be set
  
  * `PROJECT`: The GCP project you are using
  
  * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`

  * `CKPT_RESTORE_PATH`: The path to restore checkpoints from. Typically the `CKPT_DIR_PATH` + `/checkpoints/`

2. The following environment variables 
  
  * `NUM_LAYERS`: The number of layers in the model, which affects the size of the model and therefore the size of the checkpoints. Defaults to `10`.
  
  * `ACCELERATOR`: Set to `gpu` if running on a GPU, or `cpu` if running on a CPU. Defaults to `cpu`.
    * If running on GPU(s) `PJRT_DEVICE` must be set to `CPU`.
  
  * `NUM_SAVE_CALLS`: The number of times `trainer.save_checkpoint` is called. Defaults to `3`.
  
  * `NUM_LOAD_CALLS`: The number of times `trainer.strategy.load_checkpoint` is called. Defaults to `3`. 

  * `NUM_NODES`: The number of nodes you wish to deploy the workload on. Defaults to `1`.

  * `NUM_DEVICES`: The number of devices per node, or `auto`. Defaults to `auto`. 




### Checkpointing Strategy 

* Custom Strategy needs to be implemented and passed to the trainer. The strategy needs to extend [Strategy class](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.Strategy.html) or one of the class which extend Strategy class. When using Strategy which extends [FSDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html), GPU is required.

* According to Pytorch lightning documentation for FSDPStrategy the model is divided accross all the available GPU's accross all the nodes, so num_nodes can be >=1. The only caveat is to make sure that num_nodes * devices should be equal total available number of GPU. For example if you have 1 VM with 4 GPU you can or 4 VM's with 1 GPU each. num_nodes and devices can be either 1 and 4 respectively or 4 and 1 respectively, in either case FSDP will write 1 shard per GPU.

### Run

### 

```shell
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train
```
