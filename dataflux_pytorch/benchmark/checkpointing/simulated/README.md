# Benchmarking PyTorch Lightning Checkpoints with Google Cloud Storage

*   This benchmarking script will allow you to run and benchmark the performance of the [Fully Sharded Data Parallel (FSDP)](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html) save/load function in order to save and restore checkpoints from GCS. 

*   This is simulated version of code to run on CPU so that performance of GCS as backing store can be tested. 

*   The script generates [state_dict](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#what-is-a-state-dict-in-pytorch) which represents the state of the model and then everynode saves a part of the state_dict to GCS (state_dict is divided among the nodes, such that size of each shard is same). During restore, each node reads each shard from the GCS parallely. The information which tells the node, which shard it should read is stored in `.metadata` file which is writeen and read by Rank_0 node which is also called co-ordinator node. 

*   Official implementation of FSDP pytorch lightning for save_checkpoint can be found [here]
(https://github.com/Lightning-AI/pytorch-lightning/blob/3627c5bfac704d44c0d055a2cdf6f3f9e3f9e8c1/src/lightning/fabric/strategies/fsdp.py#L419)
*   Official implementation of FSDP pytorch lightning for load_checkpoint can be found [here](https://github.com/Lightning-AI/pytorch-lightning/blob/3627c5bfac704d44c0d055a2cdf6f3f9e3f9e8c1/src/lightning/fabric/strategies/fsdp.py#L519)

*   While the official implementation for both save_checkpoint and load_checkpoint does many things other than just saving/loading state_dict, those steps have been omited for couple of reasons 
*   *   We wanted to provide customer a way to see how GCS will scale with increasing model/ checkpoint sizes without needing actual GPU    
*   *   Those steps dont interact with GCS.

*   However if, for some reason, customer wants to run end to end FSDP implementation on an actual GPU, then please look at the example under multinode directory.
## Getting started

### Configuration

First ensure you are running within a virtual python enviroment, make sure gcloud config project is set to correct value. Otherwise use the following command to set it 

```shell
gcloud config set project {PROJECT_ID}
```

### Environment variables:

Set the following environment variables by updating the deployment file in order to run this on GKE cluster.
1. Set the environment variables required to run the demo. These include:
  
  * `PROJECT`: The GCP project you are using
  
  * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`

  * `WORLD_SIZE`: Number of nodes.

  * `LAYER_SIZE`: Size of each layer.

2. Set the optional environment variables, if desired:
  
  * `PADDING_SIZE`: Number of dummy tensors to add to state_dict in order to produce checkpoint of desired size. 
    *   Both padding_size and layer_size will impact the sie of the checkpoint, hence in order to reduce the variables Set layer_size to appropriate value and then keep on changing padding_size until checkpoint of desired size is generated. Default value for padding_size has been set to 4000.
  
  * `SAMPLE_COUNT`: The number of times save_checkpoint/load_checkpoint is called. Defaults to 8.

  * `USE_FSSPEC`: If set to true the code will use [gcsfs/fsspec](https://github.com/fsspec/gcsfs) in order to save_checkpoint/restore_checkpoint from GCS.
    *   If not set it will use dataflux. Defaults to dataflux.


### Installing Requirements:
 
* Install the requirements using following command `pip install -r dataflux_pytorch/benchmark/requirements.txt`; `pip install .`

### GCS Auth

Run following commands in order to authenticate your application so that it can access GCS buckets properly.
```shell
gcloud config set project PROJECT_ID
gcloud auth application-default login
```

### Running

_Note: the following instructions assume that you have Jobset and Kueue enabled on your GKE cluster. For easy compatability we recommend creating a cluster with [XPK](https://github.com/google/xpk) which will configure these features automatically._

1.  In order to connect to GKE, kubeconfig needs to be generated, this can be done using following command

```shell
gcloud container clusters get-credentials {YOUR-GKE-CLUSTER-NAME} --zone {ZONE} --project {YOUR-GCP-PROJECT}
```

2.  Build the container.

Make sure your working directory is `dataflux-pytorch`
```shell
docker build -t {YOUR_CONTAINER_NAME} .
```

3.  Upload your container to container registry

```shell
docker tag {YOUR_CONTAINER_NAME} gcr.io/{YOUR-GCP-PROJECT}/{YOUR_CONTAINER_NAME}
docker push gcr.io/{YOUR-GCP-PROJECT}/{YOUR_CONTAINER_NAME}
```

4.  Update values in benchmark-deployment.yaml.

5.  Deploy the config using following command. 

```shell
kubectl apply -f /app/dataflux_pytorch/benchmark/checkpointing/simulated/benchmark-deploy.yaml
```


### Benchmarking Results.

The table below contains benchmarking times on saving checkpoints to GCS, the average save/load time is taken over 10 calls to save_checkpoint and load_checkpoint.  The tests were done from a single GKE cluster containing several nodes with each VM having `n2-standard-32` configuration. The GKE cluster was based in `us-central1` and the GCS bucket was located in the same region. 

### Checkpoint Save & Load times.


| Num_nodes | Parameters | Checkpoint Type | Num_layers + padding_size | Size of each shard (in GB) | Total size (in GB) | Save time (in seconds) | Load time (in seconds) | Average Throughput (in GBps) |
|-----------|------------|-----------------|----------------------------|----------------------------|---------------------|------------------------|------------------------|------------------------------|
| 4         | 6B         | Dataflux        | 100K + 196                 | 18.25 GB                   | 73.02 GB            | 49.5160                | 116.3968               | 0.440                        |
| 4         | 6B         | Fsspec          | 100K + 196                 | 18.25 GB                   | 73.02 GB            | 217.1122               | 182.5110               | 0.246                        |
| 35        | 60B        | Dataflux        | 100K + 2K                  | 21.23 GB                   | 743.20 GB           | 60.8552                | 240.0361               | 2.47                         |
| 35        | 60B        | Fsspec          | 100K + 2K                  | 21.23 GB                   | 743.20 GB           | 270.6083               | 258.3709               | 1.93                         |
| 548       | 1T         | Dataflux        | 100K + 30K                 | 20.12 GB                   | 11023.88 GB         | 106.2221               | 253.1604               | 30.68                        |
| 548       | 1T         | Fsspec          | 100K + 30K                 | 20.12 GB                   | 11023.88 GB         | 290.2007               | 495.8329               | 6.58                         |
