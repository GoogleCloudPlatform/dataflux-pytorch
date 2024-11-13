# Multi-node Checkpoint Save/Restore Benchmarks

Follow the instructions here to benchmark the speeds of saving and loading checkpoints to and from GCS bucket. Access to a machine with GPUs is needed to be able to run the benchmarks. 

The benchmark code utilizes the Fully Sharded Data Parallel (FSDP) Strategy to perform multi-node checkpointing where each node writes a "shard" of the checkpoint to a shared location. This code currently supports benchmarking checkpoint saves/restores to/from GCS buckets using Dataflux, FSSpec, and GCSFuse. Additionally, there is an option to benchmark saves/restores to/from boot disk.

As the default [FSDPStrategy](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/strategies/fsdp.html#FSDPStrategy) does not support any of these options out of the box, custom strategies that inherit `lightning.pytorch.strategies.fsdp.FSDPStrategy` were implemented. The definitions of all the custom strategies referenced by the benchmark code can be found [here](https://github.com/GoogleCloudPlatform/dataflux-pytorch/blob/main/demo/lightning/checkpoint/multinode/strategies.py).


## Setup

### Install requirements
Create a Python virtual environment and activate it
```shell
python3 -m venv .venv
source .venv/bin/activate
```

Run the following commands from the root of the repo to install the packages needed by the benchmarking code

```shell
pip install -r dataflux_pytorch/benchmark/requirements.txt
pip install .
```

### Auth

Set up credentials for the benchmark code to be able to access your GCS bucket

```shell
gcloud config set project PROJECT_ID
gcloud auth application-default login
```

## Run

### Environment variables

Set the following environment variables by updating the deployment file if deploying on a GKE cluster, or by running `export ENV_VAR=value` if running locally on your machine.

1. The following environment variables must be set:
  
    * `PROJECT`: Your GCP project.
    
    * `CKPT_DIR_PATH`: The path to the directory the checkpoint files will be written to.

    * `CKPT_RESTORE_PATH`: The path to the directory the checkpoints will be loaded from. It must be `"${CKPT_DIR_PATH}/checkpoints"`. For example, if `CKPT_DIR_PATH` is `gs://my-bucket`, `CKPT_RESTORE_PATH` must be `gs://my-bucket/checkpoints`.

1. The following environment variables are optional. Default values will be used if not set:
  
    * `NUM_LAYERS`: The number of layers in the model, which affects the size of the model and therefore the size of the checkpoints. Defaults to `10`.
    
    * `ACCELERATOR`: Set to `gpu` if running on a GPU, or `cpu` if running on a CPU. Defaults to `cpu`.
      * If running on GPU(s) `PJRT_DEVICE` must be set to `CPU`.
    
    * `NUM_SAVE_CALLS`: The number of times `trainer.save_checkpoint` is called. Defaults to `3`.
    
    * `NUM_LOAD_CALLS`: The number of times `trainer.strategy.load_checkpoint` is called. Defaults to `3`. 

    * `NUM_NODES`: The number of nodes you wish to deploy the workload on. Defaults to `1`.

    * `NUM_DEVICES`: The number of devices per node. Defaults to `auto`. 

### Local Execution 

#### Dataflux
```shell
export CKPT_DIR_PATH="gs://<your-bucket-name>"
export CKPT_RESTORE_PATH="${CKPT_DIR_PATH}/checkpoints"
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train --strategy=dataflux_fsdp
```

#### FSSpec
```shell
export CKPT_DIR_PATH="gs://<your-bucket-name>"
export CKPT_RESTORE_PATH="${CKPT_DIR_PATH}/checkpoints"
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train --strategy=fsspec_fsdp
```

#### Distributed Filesystem
> [!NOTE]
> Currently only GCSFuse is supported by the benchmark code. Follow the instructions [here](https://cloud.google.com/storage/docs/cloud-storage-fuse/quickstart-mount-bucket) to mount your GCS bucket to a local directory.

```shell
export CKPT_DIR_PATH=<path-to-local-directory> 
export CKPT_RESTORE_PATH="${CKPT_DIR_PATH}/checkpoints"
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train --strategy=fsdp --distributed_filesystem
```

### Multi-node Execution

_Note: the following instructions assume that you have Jobset and Kueue enabled on your GKE cluster. For easy compatability we recommend creating a cluster with [XPK](https://github.com/google/xpk) which will configure these features automatically._

> [!IMPORTANT]  
> If benchmarking GCSFuse, make sure to uncomment the `template.metadata`, `spec.volumes`, and `spec.container.volumeMounts` blocks in the `benchmark-deploy.yaml` file.

1. Connect to your GKE cluster from your workstation
    ```shell
    # Connect to your cluster
    gcloud container clusters get-credentials {YOUR-GKE-CLUSTER-NAME} --zone {ZONE} --project {YOUR-GCP-PROJECT}
    ```

1. Build the demo container

    Make sure your working directory is `dataflux-pytorch`
    ```shell
    docker build -t dataflux-demo .
    ```

1. Upload your container to container registry

    ```shell
    docker tag dataflux-demo gcr.io/{YOUR-GCP-PROJECT}/dataflux-demo
    docker push gcr.io/{YOUR-GCP-PROJECT}/dataflux-demo
    ```

1. Update the environment variables `CKPT_DIR_PATH` and `CKPT_RESTORE_PATH` by editing the deployment file. In the cases of Dataflux
and FSSpec these variables must be set to the name of the bucket with the `gs://` prefix. In all other cases, they must be set to a POSIX path that points to some local directory.

1. Apply deployment  

   ```shell
   kubectl apply -f dataflux_pytorch/benchmark/checkpointing/multinode/benchmark-deploy.yaml
   ```


#### Benchmarking Checkpoint Saves/Loads to/from Boot Disk
>[!NOTE]
> This option is only intended for multi-node executions. 

It is not possible to create a peristent volume backed by boot disk and make it accessible to all the nodes in a cluster. The benchmark code works around this limitation by letting us benchmark saves and loads separately. 

When `--save_only` is set, only the save calls are timed and executed. All nodes write their checkpoint shards to directories local to them, saved on their respective boot disks. 

When `--load_only` is set, all nodes write the checkpoint to a GCS bucket using Dataflux. All nodes then copy the contents of this bucket to directories local to them, saved on their respective boot disks. Checkpoint load operations proceed as usual, where each node loads its own shard from the local directory.  

Update the values of the environment variables in the deployment file

```
    - name: CKPT_DIR_PATH
        value: "<path-to-local-directory>"
    - name: CKPT_RESTORE_PATH
        value: "<path-to-local-directory>/checkpoints"
```

Bemchmark saves first, by updating the command int he deployment file to 
```
python3 -u /app/dataflux_pytorch/benchmark/checkpointing/multinode/train.py --strategy=fsdp --save_only;
```

Benchmark loads next
```
python3 -u /app/dataflux_pytorch/benchmark/checkpointing/multinode/train.py --strategy=fsdp --load_only;
```


### Distributed Async Checkpoint Save

Additionally, you can make your distributed checkpointing save calls asynchronous by initializing `DatafluxFSDPStrategy` with the kwarg `use_async=True`. Under the hood, this will leverage the [torch.distributed.checkpoint.async_save](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save) to launch the checkpoint save in a separate thread.

This separate demo introduces a new env var to increase the number of training steps between checkpoint saves, as well as an env var to select distributed save or async_save behavior:

  * `STEPS_PER_SAVE`: The number of training steps to complete before a new checkpoint save occurs.

  * `USE_ASYNC`: Enable async checkpoint saves by setting this to `USE_ASYNC=1`.


To run the script use the following command. 

```shell
python3 -m dataflux_pytorch.benchmark.checkpointing.multinode.train_async_save
```

### GPU Benchmark Results

All Benchmarks were run with VMs co-located in the same region as the bucket under test. Note that performance will vary heavily for different workloads, and different GPU configurations.

We have tested our multi-node benchmark against single VMs on models of up to 13B parameters in size, sharded across 4 GPUs. The unit under test was a single [a3-highgpu-4g](https://cloud.google.com/compute/docs/gpus#a3-high) spot node. Our results were the following:

#### A3 Single Node Benchmark Results
| Model Parameters | Optimizer | GPUs | Shard Size (GB) | Total size (GB) | Save time Dataflux (s) | Save time fsspec (s) | Load time Dataflux (s) | Load time fsspec (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5B | SGD | 1 | 24.7 | 24.7 | 52.3 | 260.3 | 289.5 | 1192 |
| 6.5B | SGD | 2 | 12.4 | 24.7 | 31 | 178.9 | 101.4 | 835.9 |
| 6.5B | SGD | 4 | 6.2 | 24.7 | 22.7 | 110.9 | 68.7 | 418.5 |
| 6.5B | AdamW | 4 | 18.6 | 74.4 | 65.3 | 300.7 | 421.2 | 1225.9 |
| 13B | SGD | 2 | 24.7 | 49.5 | 66.6 | 376.1 | 218.3 | 1929.6 |
| 13B | SGD | 4 | 12.4 | 49.5 | 43 | 216.9 | 138.4 | 738.6 |

Additionally, we tested our benchmark using a GKE cluster of 16 nodes each with a single [T4 VM](https://cloud.google.com/compute/docs/gpus#t4-gpus). This benchmark iterations exhibited similar trends to the first:

#### T4 Cluster Benchmark Results

| Model Parameters | Optimizer | GPUs | Size of each shard | Total size  |Save time Dataflux (s) | Save time fsspec (s) | Load time Dataflux (s) | Load time fsspec (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1B | AdamW | 4 (1 per node) | 2.8 GB | 11.2 GB | 28.1 | 61.0 | 116.4 | 235.3 |
| 1B | AdamW | 16 (1 per node) | 724.9 MB | 11.2 GB | 31.2 | 53.2 | 119.0 | 161.9 |
| 1B | SGD | 2 (1 per node) | 1.9 GB | 3.87 GB | 12.2 | 29.9 | 32.8 | 154.5 |
| 1B | SGD | 16 (1 per node) | 241. 6 MB | 3.87 GB | 9.0 | 29.9 | 27.5 | 61.5 |
