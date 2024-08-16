# Parquet Demo Code
The code examples in this directory demonstrate how dataflux can be used for text-based training alongside Pytorch-Lightning. To support this training, this example subclasses DataFluxIterableDataset for compatibility with parquet files from the HuggingFace FineWeb dataset.

Please note that these demos are for educational and example purposes only, and have not been optimized for performance.

## Single Node
The single node exmaple code here can be run on your local workstation with a command such as the following:

```
# python3 ./single-node/model.py --project=<MY-PROJECT> --bucket=<MY-BUCKET> --batch-size=100 --prefix="fineweb/sample/10BT"
```

This will perform a simple autoencoder training pass that leverages lightning to skip writing a large volume of boilerplate code.

## Distributed
The distributed demo provides a simple example of how the DatafluxIterableDataset can be extended to function seamlessly with distributed GKE runs facilitated by Lightning.

To execute the demo-code locally and simulate the use of mulltiple nodes, use the following commands. 

*Note: The number of batches processed by each process must be identical across all processes to avoid freezing/deadlock during DDP executions. In the demo this can be guaranteed by providing the `--limit-train-batches` argument with a value less than the minimum batch count for a given process.*

### Single process local execution:
The following command will execute a single-process DDP strategy execution, where the number of processes is dictated by the `--devices` parameter. For local execution, ensure that the parameter `--local` is specified.
```
python3 ./distributed/model.py --rank=0 --project=<MY-PROJECT> --bucket=<MY-BUCKET> --prefix=fineweb/sample/10BT/ --num-workers=1 --num-nodes=1 --batch-size=128 --epochs=2 --devices=2 --log-level=INFO --local --limit-train-batches=10
```

### Multiprocess local execution:
To test this code locally with multiple processes (representing multiple nodes), use a setup similar to the following.

Execution 1:
```
python3 ./distributed/model.py --rank=0 --project=<MY-PROJECT> --bucket=<MY-BUCKET> --prefix=fineweb/sample/10BT/ --num-workers=1 --num-nodes=2 --batch-size=128 --epochs=2 --devices=2 --log-level=INFO --local=True --limit-train-batches=10
```

Execution 2 (in a separate shell from execution 1):
```
python3 ./distributed/model.py --rank=1 --project=<MY-PROJECT> --bucket=<MY-BUCKET> --prefix=fineweb/sample/10BT/ --num-workers=1 --num-nodes=2 --batch-size=128 --epochs=2 --devices=2 --log-level=INFO --local=True --limit-train-batches=10
```

This will set the demo to communicate across two processes, treating them as individual nodes. They will default to communicating via localhost:1234. Both the `--num-nodes` and `--devices` parameters can be configured together, as Lightning will correctly identify the division of labor across both nodes and devices. This can apply in distributed use-cases where you have multiple GPUs per node across a cluster.

### Multiprocess GKE cluster execution:
A docker container can be constructed using the `Dockerfile` in the root of the pytorch directory. To build a container and push the artifact to your Google Cloud Project you can use the following commands:

```
docker build -t my-container .
docker tag my-container gcr.io/<PROJECT_NAME>/my-container
docker push gcr.io/<PROJECT_NAME>/my-container
```

To create an execution, please take a look at [example-deploy.yaml](demo/lightning/text-based/distributed/example-deploy.yaml). This can be run against your own cluster once configured via the `kubectl` command:
```
kubectl apply -f deploy.yaml
```

*Note: the example YAML file  and model code assumes that you have jobset and Kueue enabled on your GKE cluster. For easy compatability we recommend creating a cluster with [XPK](https://github.com/google/xpk) which will configure these features automatically.*
