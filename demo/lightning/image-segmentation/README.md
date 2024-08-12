# Image Segmentation Demo Code
The code examples in this directory demonstrate how Dataflux can be used for image segmentation training alongside PyTorchLightning. The image segmentation workload implemented here works on the KiTS 19 dataset which has `210` images and their corresponding labels. The images and their labels are stored in separate directories.


Please note that these demos are for educational and example purposes only, and have not been optimized for performance.


## Setup
1. Create a copy of the dataset in your GCP project
    ```sh
    gcloud storage cp -r gs://dataflux-demo-public/image-seg-dataset gs://{YOUR-GCS-BUCKET}
    ```
1. Clone `dataflux-pytorch` repo to your workstation
    ```sh
    # Create a new directory on your workstation
    mkdir dataflux-demo && cd dataflux-demo

    # Clone dataflux-pytorch
    git clone --recurse-submodules https://github.com/GoogleCloudPlatform/dataflux-pytorch
    ```
1. Install dependencies
    ```sh
    # Create a python virutal environment and activate it
    python3 -m venv .venv && source .venv/bin/activate

    cd dataflux-pytorch
    pip install .
    ```

## Single Node Local Execution

```sh
    python3 demo/lightning/image-segmentation/train.py \
    --gcp_project={YOUR-GCP-PROJECT} \
    --gcs_bucket={YOUR-GCS-BUCKET} \
    --images_prefix={YOUR-GCS-BUCKET}\images \
    --labels_prefix={YOUR-GCS-BUCKET}\labels \
    --num_dataloader_threads=10 \
    --prefetch_factor=5 \
    --num_devices=1 \
    --num_nodes=1 \
    --local=True 
```

Be sure to set `--local` to `True`.

## Multi-node GKE Cluster Execution
_Note: the following instructions assume that you have Jobset and Kueue enabled on your GKE cluster. For easy compatability we recommend creating a cluster with XPK which will configure these features automatically._

1. Connect to your GKE cluster from your workstation
    ```sh
    # If needed, run
    gcloud auth login && gcloud auth application-default login
    # Connect to the cluster
    gcloud container clusters get-credentials {YOUR-GKE-CLUSTER-NAME} --zone {ZONE} --project {YOUR-GCP-PROJECT}
    ```

1. Build the demo container

    Make sure your working directory is `dataflux-pytorch`
    ```sh
    docker build -t dataflux-demo .
    ```

1. Upload your container to container registry
    ```sh
    docker tag dataflux-demo gcr.io/{YOUR-GCP-PROJECT}/dataflux-demo
    docker push gcr.io/{YOUR-GCP-PROJECT}/dataflux-demo
    ```
1. Apply deployment  

   Update `demo/lightning/image-segmentation/deployment.yaml` at appropriate places. Specifically the arguments to `spec.containers.command` and `spec.containers.image`. The deployment has been tested on a cluster with `4` nodes. If you wish to run the workload on different number of nodes, make sure to set `spec.parallelism`, `spec.completions`, the environment variable `WORLD_SIZE`, and the argument `--num_nodes` to `spec.containers.command` are all set to the _same_ value, which is the number of nodes.

   ```sh
   kubectl apply -f deploy.yaml
   ``` 