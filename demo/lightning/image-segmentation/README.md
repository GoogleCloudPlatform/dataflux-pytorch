# Image Segmentation Demo Code
The code examples in this directory demonstrate how GCS Connector for Pytorch can be used for image segmentation training alongside PyTorch Lightning. The image segmentation workload implemented here works on the [KiTS19](https://github.com/neheller/kits19) dataset which contains `210` images and their corresponding labels. The images and their labels are stored in separate directories.

If you are looking to benchmark data loading performance, pass `--benchmark` to `train.py`. Note that this will run the training loop but does not run the actual training logic.

If you are looking to benchmark the listing performance only, pass `--listing_only` to `train.py`. Note that this will skip training altogether. Passing `--benchmark` along with `--listing_only` would have the same effect as passing just the latter.


1. `dataset.py`

    Defines the `DatafluxPytTrain` class which wraps two `DataFluxMapStyleDataset`s corresponding to the `images` and the `labels` in the raw dataset.
    ```py
    def __init__(
            self,
            project_name,
            bucket_name,
            **kwargs,
        ):

            self.train_transforms = kwargs["transforms"]
            patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
            self.patch_size = patch_size
            self.rand_crop = RandBalancedCrop(patch_size=patch_size,
                                            oversampling=oversampling)

            # GCS Connector for Pytorch specific setup.
            self.project_name = project_name
            self.bucket_name = bucket_name

            self.images_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
                project_name=self.project_name,
                bucket_name=self.bucket_name,
                config=dataflux_mapstyle_dataset.Config(
                    # This needs to be True to map images with labels
                    sort_listing_results=True,
                    prefix=kwargs["images_prefix"],
                ),
            )

            self.labels_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
                project_name=self.project_name,
                bucket_name=self.bucket_name,
                config=dataflux_mapstyle_dataset.Config(
                    # This needs to be True to map images with labels
                    sort_listing_results=True,
                    prefix=kwargs["labels_prefix"],
                ),
            )
    ```

1. `data.py`

    Implements the `Unet3DDataModule` class which inherits from PyTorch Lightning's `LightningDataModule`. An instance of the `DatafluxPytTrain` class is created in its `setup` function.

    ```py
    def setup(self, stage="fit"):
        if stage == "fit":
            train_data_kwargs = {
                "patch_size": self.args.input_shape,
                "oversampling": self.args.oversampling,
                "seed": self.args.seed,
                "images_prefix": self.args.images_prefix,
                "labels_prefix": self.args.labels_prefix,
                "transforms": get_train_transforms(),
            }
            self.train_dataset = DatafluxPytTrain(
                project_name=self.args.gcp_project,
                bucket_name=self.args.gcs_bucket,
                **train_data_kwargs,
            )
            self.train_sampler = None
            if self.args.num_workers > 1:
                self.train_sampler = DistributedSampler(
                    self.train_dataset,
                    seed=self.args.seed,
                    drop_last=True
                )
    
    ```

1. `train.py`

    An instance of the `Unet3DDataModule` class is created and passed to the Pytorch Lightning `Trainer` instance's `fit` function in the `train_dataloaders` keyword argument.
    ```py
    model = Unet3DLightning(flags)
    train_data_loader = Unet3DDataModule(flags)
    trainer = pl.Trainer(
        accelerator=flags.accelerator,
        max_epochs=flags.epochs,
        devices=flags.num_devices,
        num_nodes=flags.num_nodes,
        strategy="ddp",
    )
    trainer.fit(model=model, train_dataloaders=train_data_loader)

    ```


Please note that these demos are for educational and example purposes only, and have not been optimized for performance.


## Setup
1. Create a copy of the dataset in your GCP project
    ```sh
    gcloud storage cp -r gs://dataflux-demo-public/image-segmentation-dataset gs://{YOUR-GCS-BUCKET}
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
    # Create a python virtual environment and activate it
    python3 -m venv .venv && source .venv/bin/activate

    cd dataflux-pytorch
    pip install .
    ```

## Single Node Local Execution

```sh
    python3 demo/lightning/image-segmentation/train.py \
    --gcp_project={YOUR-GCP-PROJECT} \
    --gcs_bucket={YOUR-GCS-BUCKET} \
    --images_prefix={YOUR-GCS-BUCKET}/images \
    --labels_prefix={YOUR-GCS-BUCKET}/labels \
    --num_dataloader_threads=10 \
    --prefetch_factor=5 \
    --num_devices=1 \
    --num_nodes=1 \
    --local 
```

Be sure to specify `--local`.

## Multi-node GKE Cluster Execution
_Note: the following instructions assume that you have Jobset and Kueue enabled on your GKE cluster. For easy compatability we recommend creating a cluster with [XPK](https://github.com/google/xpk) which will configure these features automatically._

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

   Update `demo/lightning/image-segmentation/deployment.yaml` at appropriate places. Specifically, `spec.containers.image` and the arguments to `spec.containers.command`. 
   
   This deployment has been tested on a cluster with `4` nodes. If you wish to run the workload on a cluster with different number of nodes, make sure to set `spec.parallelism`, `spec.completions`, the environment variable `WORLD_SIZE`, and the argument `--num_nodes` to `spec.containers.command` are all set to the _same_ value, which is the number of nodes in your cluster.

   ```sh
   kubectl apply -f deploy.yaml
   ``` 

## Benchmark Results

The table below summarizes the benchmark results from our internal testing. Specifically, the the maximum egress throughput and the maximum QPS observed on the bucket that held the dataset are recorded. The executions in the benchmarking effort features the following shared parameters:

- num_devices: `1`
- batch_size: `32`
- prefetch_factor: `2`
- num_epochs: `2`
- num_workers (`torch.utils.data.DataLoader`): `16`


These benchmarks are simulated, meaning no actual training is performed during each step. Instead, the node sleeps for a set number of seconds. All benchmarks were run against a dataset with `105,000` images and their corresponding labels, each of which are about `150 MB` in size. These images and labels were derived from the [KiTS19](https://github.com/neheller/kits19) dataset. All testing was performed on a multi-node CPU only cluster, backed by a bucket co-located in the same region (this is critical for maximizing dataloading performance). The test cluster was comprised of [n2-standard-32](https://cloud.google.com/compute/docs/general-purpose-machines#n2_machine_types) nodes with `128 GiB` of allocatable memory each. 

For all runs here, the observed step time was approximately `1 second`. 

This code is still being updated and improved, so all benchmark results are subject to change.

<table>
    <tr>
        <th rowspan=2> <pre>num_nodes</pre> </th>
        <th rowspan=2> Steps per node </th>
        <th colspan=2> Max Egress Throughput </th>
        <th colspan=2> Max QPS </th>
    </tr>
    <tr>
        <th> Epoch 1</th>
        <th> Epoch 2</th>
        <th> Epoch 1</th>
        <th> Epoch 2</th>
    </tr>
    <tr>
        <td> 32 </td>
        <td> 102</td>
        <td> 454.2 Gbps</td>
        <td> 503.8 Gbps</td>
        <td> 619</td>
        <td> 692</td>
    </tr>
    <tr>
        <td> 64</td>
        <td> 51</td>
        <td> 1.032 Tbps</td>
        <td> 1.067 Tbps</td>
        <td> 1358</td>
        <td> 1458</td>
    </tr>
    <tr>
        <td> 128</td>
        <td> 25</td>
        <td> 1.844 Tbps</td>
        <td> 1.261 Tbps</td>
        <td> 2439</td>
        <td> 1879</td>
    </tr>
    <tr>
        <td> 256</td>
        <td> 12</td>
        <td> 1.95 Tbps</td>
        <td> 1.832 Tbps</td>
        <td> 2648</td>
        <td> 2220</td>
    </tr>
    <tr>
        <td>512</td>
        <td> 6</td>
        <td> 1.907 Tbps</td>
        <td> 1.82 Tbps</td>
        <td> 2855</td>
        <td> 2133</td>
    </tr>
    <tr>
        <td>1024</td>
        <td> 3</td>
        <td> 1.823 Tbps</td>
        <td> 2.122 Tbps</td>
        <td> 3286</td>
        <td> 2710</td>
    </tr>
</table>

