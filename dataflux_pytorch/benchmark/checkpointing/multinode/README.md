# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

This benchmarking script will allow you to run and benchmark the performance of the PyTorch Lightning Checkpoint save/load function. The multinode benchmarking script does require a GPU. The script runs the `WikiText2` PyTorch Lightning demo code with some modifications.

## Getting started

### Setup
 
#### Install requirements
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

#### Auth

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

1. Update the environment variables and the arguments to `train.py` (`spec.command`) as needed. See the previous section for details.
   >[!NOTE]
   >Update the environment variables `CKPT_DIR_PATH` and `CKPT_RESTORE_PATH` by editing the deployment file. In the cases of Dataflux and FSSpec these variables must be set to the name of the bucket with the `gs://` prefix. In all other cases, they must be set to a POSIX path that points to some local directory.

1. Apply deployment  

   ```shell
   kubectl apply -f dataflux_pytorch/benchmark/checkpointing/multinode/benchmark-deploy.yaml
   ```


#### Benchmarking Checkpoint Saves/Loads to/from Boot Disk
>[!NOTE]
> This option is only intended for mulit-node executions. 

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