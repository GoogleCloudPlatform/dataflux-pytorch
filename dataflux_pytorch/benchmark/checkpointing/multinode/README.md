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

### Environment variables:

You will need to set the following environment variables in order for the benchmarking code to run properly. 
1. Set the environment variables required to run the demo. These include:
  
  * `PROJECT`: The GCP project you are using
  
  * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`

2. Set the optional environment variables, if desired:
  
  * `NUM_LAYERS`: The number of layers in the model, which affects the size of the model and therefore the size of the checkpoints. Defaults to 10.
  
  * `ACCELERATOR`: Set to `gpu` if running on a GPU, or `cpu` if running on a CPU (default)
    * If running on a GPU, you also must set `PJRT_DEVICE` to `CPU`.
  
  * `MIN_EPOCHS_SAVE`: Minimum epochs for which training should run during first training loop which saves checkpoint. Defaults to 4. For detailed explaination of min_epochs see [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#min-epochs).
  
  * `MIN_EPOCHS_RESTORE`: Minimum epochs for which training should run during second training loop which restores checkpoint. Defaults to 4.
  
  * `MAX_EPOCHS_SAVE` : Maximum epochs for which training should run during first training loop, which saves checkpoint. Defaults to 5. For detailed explanation of max_epochs see [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#max-epochs).
  
  * `MAX_EPOCHS_RESTORE` : Maximum epochs for which training should run during second training loop, which restores checkpoint. Defaults to 5.
  
  * `MAX_STEPS_SAVE`: Maximum number of steps for which training can run during first trainig loop. Defaults to 5. For more infomration on max_steps see [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#max-steps).
  
  * `MAX_STEPS_RESTORE`: Maximum number of steps for which training can run during second trainig loop. Defaults to 5. 


### Installing Requirements:
 
* Install the requirements using following command `pip install -r dataflux_pytorch/benchmark/checkpointing/requirements.txt`; `pip install .`


### Checkpointing Strategy: 

* Custom Strategy needs to be implemented and passed to the trainer. The strategy needs to extend [Strategy class](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.Strategy.html) or one of the class which extend Strategy class. When using Strategy which extends [FSDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html), GPU is required.

* According to Pytorch lightning documentation for FSDPStrategy the model is divided accross all the available GPU's accross all the nodes, so num_nodes can be >=1. The only caveat is to make sure that num_nodes * devices should be equal total available number of GPU. For example if you have 1 VM with 4 GPU you can or 4 VM's with 1 GPU each. num_nodes and devices can be either 1 and 4 respectively or 4 and 1 respectively, in either case FSDP will write 1 shard per GPU.

### Running

To run the script use the following command. 

```shell
python dataflux_pytorch/benchmark/checkpointing/multinode/train.py
python dataflux_pytorch/benchmark/checkpointing/multinode/train.py
```
