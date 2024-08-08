## Deployment

### Cluster

We use [Ray.IO](Ray.IO) to deploy and scale the training workload with [PyTorch](https://pytorch.org/) as the underlying machine learning framework.

We will spin up a [Ray cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html#ray-cluster) for our workload. The [cluster.yaml](cluster.yaml) file contains the specification and here's the overview:



*   Ray head node
    *   There’s only one head node in the cluster that is responsible for cluster management. See more about the Ray head node [here](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node).
    *   Key Specification:
        *   **Machine type:** n1-standard-32
        *   **Source Image:** projects/deeplearning-platform-release/global/images/family/common-cpu
*   Ray worker node
    *   We have two worker nodes in the cluster and they are responsible for running the workload. See more about the Ray worker node [here](https://docs.ray.io/en/latest/cluster/key-concepts.html#worker-node).
    *   Key Specification:
        *   **Machine type**: n1-standard-32
        *   **Source Image**: projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11-py310
            *   This image comes pre-installed with NVIDIA drivers and a lot more machine learning applications.
        *   **GPUs**: 4 \* NVIDIA Tesla V100 GPUs


> **_NOTE:_**  It would be best to run the following steps on your workstation with a display device so you can see the Ray dashboard using a browser.

1. Clone the `dataflux-pytorch` repository, install the packages and go to the working directory.
   ```shell
   git clone --recurse-submodules https://github.com/GoogleCloudPlatform/dataflux-pytorch
   cd dataflux-pytorch
   pip install gcs-torch-dataflux
   cd demo/image-segmentation
   ```
2. [Install Ray](https://docs.ray.io/en/latest/ray-overview/installation.html).
   ```shell
   pip install -U "ray[default]"
   ```
3. Modify the [cluster.yaml](cluster.yaml) file by replacing all the parameters highlighted in the curly brackets. Specifically, they are `{YOUR_CLUSTER_NAME}
`, `{YOUR_REGION}`, `{{YOUR_AVAILABILITY_ZONE}}` and `{YOUR_PROJECT_ID}`. Note that there's an additional `{YOUR_PROJECT_ID}` field in the `serviceAccounts` field that you'll need to replace.
1. Run the following command to start the Ray cluster. You may follow the logs to check the startup progress.
    ```shell
    ray up cluster.yaml -y --no-config-cache
    ```
2. Once the cluster has launched, run the following command to connect to the Ray dashboard.
   ```shell
   ray dashboard cluster.yaml --no-config-cache
   ```
3. On your workstation, you can visit http://localhost:8265/ to see the status of the created Ray cluster.

    * Note that it might take a while to set up the worker nodes. Once they are completed, you can verify the cluster status looks like:

    ![sample cluster image](imgs/cluster.png "sample cluster image")

### Workload


3. Finally, we support these additional flags in [`arguments.py`](arguments.py) and pass those values in from [`run_and_time.sh`](run_and_time.sh).

> **_IMPORTANT:_**  Make sure to modify the [`run.sh`](run.sh) file by specifying the values for ``, ``, `GCP_PROJECT` and `GCS_BUCKET`.

4. Submit the workload job by running

```shell
export RAY_ADDRESS=http://localhost:8265
python3 submit.py
```

## Results
The job should finish within a couple minutes and you can follow the job log by clicking into the job detail from the Ray dashboard.
