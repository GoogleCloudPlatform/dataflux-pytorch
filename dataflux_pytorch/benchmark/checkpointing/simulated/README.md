# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

*   This benchmarking script allows you to run and benchmark the performance of the [Fully Sharded Data Parallel (FSDP)](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html) save/load function in order to save and restore checkpoints from GCS. 

*   This is a simulated version of the code that runs on CPU to test the performance of GCS as a backing store for AI/ML workloads without GPU.

*   The script generates [state_dict](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#what-is-a-state-dict-in-pytorch) that represents the state of the model. Each node then saves a part of the state_dict to GCS (the state_dict is divided among the nodes, such that the size of each shard is the same). During restore, each node reads its shard from GCS in parallel. The `.metadata` file, which is written and read by the Rank_0 (coordinator) node, contains information about which shard each node should read.

*   The official implementation of FSDP in PyTorch Lightning for save_checkpoint can be found [here](https://github.com/Lightning-AI/pytorch-lightning/blob/3627c5bfac704d44c0d055a2cdf6f3f9e3f9e8c1/src/lightning/fabric/strategies/fsdp.py#L419)
*   The official implementation of FSDP in PyTorch Lightning for load_checkpoint can be found [here](https://github.com/Lightning-AI/pytorch-lightning/blob/3627c5bfac704d44c0d055a2cdf6f3f9e3f9e8c1/src/lightning/fabric/strategies/fsdp.py#L519)

*   While the official implementation of both `save_checkpoint` and `load_checkpoint` does many things beyond just saving/loading the state_dict, these additional steps have been omitted for the following reasons: 
    *   We wanted to demonstrate how GCS will scale with increasing model/checkpoint sizes without needing an actual GPU.
    *   These additional steps do not interact with GCS.

*   If a user wishes to run an end to end FSDP implementation on an actual GPU, then please look at the example under the [multinode](https://github.com/GoogleCloudPlatform/dataflux-pytorch/tree/main/demo/lightning/checkpoint/multinode) directory.

## Getting started

### Configuration

Ensure you are running within a python virtual environment, and ensure gcloud config project is set to your project. You can use the following command to set it:

```shell
gcloud config set project {PROJECT_ID}
```

### Environment variables:

Set the following environment variables by updating the deployment file to run on a GKE cluster.
1. Set the environment variables required to run the demo. These include:
  
    * `PROJECT`: The GCP project you're using.
  
    * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`

    * `WORLD_SIZE`: The number of nodes.

    * `LAYER_SIZE`: The size of each layer. Each layer will have layer_size number of neurons and each neuron will receive input of 1000 features.

2. Set the optional environment variables, if desired:
  
    * `PADDING_SIZE`: The number of dummy tensors to add to the state_dict in order to produce checkpoints of desired size.
        *   Both padding_size and layer_size will impact the size of the checkpoint. Therefore, set layer_size to an appropriate value first, and then adjust padding_size until the checkpoint of the desired size is generated. The default value for padding_size is 4000.
        *   Increasing the padding_size by 2x while keeping the layer_size same will increase the checkpoint size by 2x.
        *   Increasing the layer_size by 2x while keeping the padding_size same will increase the checkpoint size by 2x.
  
    * `SAMPLE_COUNT`: The number of times save_checkpoint/load_checkpoint is called. The default is 8.

    * `USE_FSSPEC`: If set to true, the code will use [gcsfs/fsspec](https://github.com/fsspec/gcsfs) in order to save_checkpoint/restore_checkpoint from GCS.
        *   If not set, it will use GCS Connector for Pytorch by default.

### Installing Requirements:
 
Install the required dependencies using the following commands:

```shell
 pip install -r dataflux_pytorch/benchmark/requirements.txt
 pip install .
```
### GCS Auth

Run the following commands to authenticate your application so it can access GCS buckets.
```shell
gcloud auth application-default login
```

### Running on GKE

> [!NOTE]  
> The following instructions assume that you have Jobset and Kueue enabled on your GKE cluster. For easier compatibility, we recommend creating a cluster using [XPK](https://github.com/google/xpk) which will configure these features automatically.

1.  To connect to GKE, you need to generate a kubeconfig file. This can be done using the following command:

```shell
gcloud container clusters get-credentials {YOUR-GKE-CLUSTER-NAME} --zone {ZONE} --project {YOUR-GCP-PROJECT}
```

2.  Build the container.

Ensure that your working directory is `dataflux-pytorch`
```shell
docker build -t {YOUR_CONTAINER_NAME} .
```

3.  Upload your container to the container registry:

```shell
docker tag {YOUR_CONTAINER_NAME} gcr.io/{YOUR-GCP-PROJECT}/{YOUR_CONTAINER_NAME}
docker push gcr.io/{YOUR-GCP-PROJECT}/{YOUR_CONTAINER_NAME}
```

4.  Update the values in `benchmark-deployment.yaml`.

5.  Deploy the configuration using the following command:

```shell
kubectl apply -f /app/dataflux_pytorch/benchmark/checkpointing/simulated/benchmark-deploy.yaml
```


### Benchmarking Results.

The table below contains benchmarking times for saving checkpoints to GCS. The average save/load time is taken over 8 calls to  save_checkpoint and load_checkpoint. The tests were conducted on a single GKE cluster containing several nodes, with each VM having the `n2-standard-32` configuration. The GKE cluster, and the GCS bucket were located in `us-central1`.

### Checkpoint Save:


| Num_nodes | Parameters | Checkpoint Type | Num_layers + padding_size | Size of each shard (in GB) | Total size (in GB) | Save time (in seconds) | Ingress Throughput (in GBps) |
|-----------|------------|-----------------|----------------------------|----------------------------|---------------------|------------------------|------------------------------|
| 4         | 6B         | Dataflux        | 100K + 196                 | 18.25 GB                   | 73.02 GB            | 49.5160                | 1.474                        |
| 4         | 6B         | Fsspec          | 100K + 196                 | 18.25 GB                   | 73.02 GB            | 217.1122               | 0.337                        |
| 35        | 60B        | Dataflux        | 100K + 2K                  | 21.23 GB                   | 743.20 GB           | 60.8552                | 12.21                        |
| 35        | 60B        | Fsspec          | 100K + 2K                  | 21.23 GB                   | 743.20 GB           | 270.6083               | 2.75                         |
| 548       | 1T         | Dataflux        | 100K + 30K                 | 20.12 GB                   | 11023.88 GB         | 106.2221               | 103.61                       |
| 548       | 1T         | Fsspec          | 100K + 30K                 | 20.12 GB                   | 11023.88 GB         | 290.2007               | 38.03                        |



### Checkpoint Load:

| Num_nodes | Parameters | Checkpoint Type | Num_layers + padding_size | Size of each shard (in GB) | Total size (in GB) | Load time (in seconds) | Egress Throughput (in GBps) |
|-----------|------------|-----------------|----------------------------|----------------------------|---------------------|------------------------|------------------------------|
| 4         | 6B         | Dataflux        | 100K + 196                 | 18.25 GB                   | 73.02 GB            | 116.3968               | 0.627                        |
| 4         | 6B         | Fsspec          | 100K + 196                 | 18.25 GB                   | 73.02 GB            | 182.5110               | 0.400                        |
| 35        | 60B        | Dataflux        | 100K + 2K                  | 21.23 GB                   | 743.20 GB           | 240.0361               | 3.10                         |
| 35        | 60B        | Fsspec          | 100K + 2K                  | 21.23 GB                   | 743.20 GB           | 258.3709               | 2.87                         |
| 548       | 1T         | Dataflux        | 100K + 30K                 | 20.12 GB                   | 11023.88 GB         | 253.1604               | 43.59                        |
| 548       | 1T         | Fsspec          | 100K + 30K                 | 20.12 GB                   | 11023.88 GB         | 495.8329               | 22.23                        |
