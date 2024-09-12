# Checkpoint Demo for PyTorch Lightning

The code in this folder provides a training demo for multi node checkpointing with PyTorch Lightning. This demo is under development.

## Limitations
* Custom Strategy needs to be implemented and passed to the trainer. The strategy needs to extend [Strategy class](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.Strategy.html) or one of the class which extend Strategy class. When using Strategy which extends [FSDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html), GPU is required.
* `requirements.txt` includes gcsfs because even though it is not used for checkpointing, PyTorch Lightning's default logger also writes to the root directory where checkpoints are saved.

## Running locally

1. Set the environment variables required to run the demo. These include:
  * `PROJECT`: The GCP project you are using
  * `CKPT_DIR_PATH`: The full path of the directory in which to save checkpoints, in the format `gs://<bucket>/<directory>/`
2. Set the optional environment variables, if desired:
  * `NUM_LAYERS`: The number of layers in the model, which affects the size of the model and therefore the size of the checkpoints
  * `ACCELERATOR`: Set to `gpu` if running on a GPU, or `cpu` if running on a CPU (default)
    * If running on a GPU, you also must set `PJRT_DEVICE` to `CPU`.
  * `MIN_EPOCHS_SAVE`: Minimum epochs for which training should run during first training loop which saves checkpoint. Defaults to 4. For detailed explaination of min_epochs see [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#min-epochs).
  * `MIN_EPOCHS_RESTORE`: Minimum epochs for which training should run during second training loop which restores checkpoint. Defaults to 4.
  * `MAX_EPOCHS_SAVE` : Maximum epochs for which training should run during first training loop, which saves checkpoint. Defaults to 5. For detailed explanation of max_epochs see [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#max-epochs).
  * `MAX_EPOCHS_RESTORE` : Maximum epochs for which training should run during second training loop, which restores checkpoint. Defaults to 5.
  * `MAX_STEPS_SAVE`: Maximum number of steps for which training can run during first trainig loop. Defaults to 3. For more infomration on max_steps see [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#max-steps).
  * `MAX_STEPS_RESTORE`: Maximum number of steps for which training can run during second trainig loop. Defaults to 3. 
3. Install requirements: `pip install -r demo/lightning/checkpoint/requirements.txt`; `pip install .`
4. Run the binary: `python3 -m demo.lightning.checkpoint.multinode.train`

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
2. Make a copy of `demo/lightning/checkpoint/multinode/example-deploy.yaml` and update the placeholders and environment variables as needed
3. Run `kubectl apply -f <path-to-your-yaml-file>`
