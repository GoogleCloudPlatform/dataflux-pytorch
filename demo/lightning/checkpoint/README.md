# Checkpoint Demo for PyTorch Lightning

The code in this folder provides a training demo for checkpointing with PyTorch Lightning. This demo is under development.

## Limitations

* The demo currently only runs with [`state_dict_type="full"`](https://lightning.ai/docs/pytorch/stable/common/checkpointing_expert.html#save-a-distributed-checkpoint) when using FSDP.
* `requirements.txt` includes gcsfs because even though it is not used for checkpointing, PyTorch Lightning's default logger also writes to the root directory where checkpoints are saved.

## Running locally

1. Set the environment variables required to run the demo. These include:
  * `PROJECT`: The GCP project you are using
  * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`
2. Set the optional environment variables, if desired:
  * `NUM_LAYERS`: The number of layers in the model, which affects the size of the model and therefore the size of the checkpoints
  * `ACCELERATOR`: Set to `gpu` if running on a GPU, or `cpu` if running on a CPU (default)
    * If running on a GPU, you also must set `PJRT_DEVICE` to `CUDA`. 
  * `TRAIN_STRATEGY`: Set to `fsdp` to use the FSDP strategy. The default is `ddp`. If using FSDP, you must use GPUs
4. Install requirements: `pip install -r demo/lightning/checkpoint/requirements.txt`; `pip install .`
3. Run the binary: `python3 -m demo.lightning.checkpoint.train`

## Running on GKE

These instructions assume you have an existing GKE cluster with Kueue and Jobset installed. These are installed by default if you create the cluster using [xpk](https://github.com/google/xpk).

### Build and push the Docker container

```
docker build -t my-container .
docker tag my-container gcr.io/<PROJECT_NAME>/my-container
docker push gcr.io/<PROJECT_NAME>/my-container
```

Make sure to update the container name in the yaml config file to match the one you're using.

### Run the workload on GKE

1. Connect to your GKE cluster: `gcloud container clusters get-credentials <CLUSTER_NAME> --region=<COMPUTE_REGION>`
2. Make a copy of `demo/lightning/checkpoint/example-deploy.yaml` and update the placeholders and environment variables as needed
3. Run `kubectl -f apply <path-to-your-yaml-file>`
