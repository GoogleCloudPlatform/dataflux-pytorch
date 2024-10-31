# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save/load function. The multinode benchmarking script does require a GPU. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

## Getting started

### Configuration

First ensure you are running within a virtual python enviroment, make sure gcloud config project is set to correct value. Otherwise use the following command to set it 

```shell
gcloud config set project {PROJECT_ID}
```

### Environment variables:

You will need to set the following environment variables in order for the benchmarking code to run properly. 
1. Set the environment variables required to run the demo. These include:
  
  * `PROJECT`: The GCP project you are using
  
  * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`

  * `CKPT_RESTORE_PATH`: The path to restore checkpoints from. Typically the `CKPT_DIR_PATH` + `/checkpoints/`

2. Set the optional environment variables, if desired:
  
  * `NUM_LAYERS`: The number of layers in the model, which affects the size of the model and therefore the size of the checkpoints. Defaults to 10.
  
  * `ACCELERATOR`: Set to `gpu` if running on a GPU, or `cpu` if running on a CPU (default)
    * If running on a GPU, you also must set `PJRT_DEVICE` to `CPU`.
  
  * `NUM_SAVE_CALLS`: The number of times `trainer.save_checkpoint` is called. Defaults to 3.
  
  * `NUM_LOAD_CALLS`: The number of times `trainer.strategy.load_checkpoint` is called. Defaults to 3. 

  * `NUM_NODES`: The number of nodes (machines). Defaults to 1.

  * `NUM_DEVICES`: The number of devices per node, or `auto`. Defaults to `auto`. 


### Installing Requirements:
 
* Install the requirements using following command `pip install -r dataflux_pytorch/benchmark/requirements.txt`; `pip install .`


### Checkpointing Strategy: 

* Custom Strategy needs to be implemented and passed to the trainer. The strategy needs to extend [Strategy class](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.Strategy.html) or one of the class which extend Strategy class. When using Strategy which extends [FSDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html), GPU is required.

* According to Pytorch lightning documentation for FSDPStrategy the model is divided accross all the available GPU's accross all the nodes, so num_nodes can be >=1. The only caveat is to make sure that num_nodes * devices should be equal total available number of GPU. For example if you have 1 VM with 4 GPU you can or 4 VM's with 1 GPU each. num_nodes and devices can be either 1 and 4 respectively or 4 and 1 respectively, in either case FSDP will write 1 shard per GPU.

### Running

To run the script use the following command. 

```shell
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train
```

### Distributed Async Checkpoint Save

Additionally, you can make your distributed checkpointing save calls asynchronous by using the `AsyncDatafluxFSDPStrategy`. Under the hood, this will leverage the [torch.distributed.checkpoitn.async_save](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save) to launch the checkpoint save in a separate thread.

This separate demo introduces a new env var to increase the number of training steps between checkpoint saves, which results in more training cycles between checkpoint saves, as well as a var to select using distributed `save` or `async_save` behavior.

  * `STEPS_PER_SAVE`: The number of training steps to complete before a new checkpoint save occurs.

  * `STRATEGY`: The training strategy to use, providing the ability to compare the difference between DatafluxFSDPStrategy (`dataflux_fsdp`) and AsyncDatafluxFSDPStrategy (`async_dataflux_fsdp`).


To run the script use the following command. 

```shell
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train_async_save
```
