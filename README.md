# Accelerated Dataloader for PyTorch with Google Cloud Storage

The Accelerated Dataloader for PyTorch with Google Cloud Storage lets you connect directly to a GCS bucket as a PyTorch dataset, without pre-loading the data to local disk, and with optimizations to make training up to **3X faster** when the dataset consists of many small files (e.g., 100 - 500 KB).

The Accelerated Dataloader for PyTorch implements PyTorch’s [dataset primitive](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that can be used to efficiently load training data from GCS. The library currently supports [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random data access patterns and [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for streaming data access patterns.

Furthermore, the Accelerated Dataloader for PyTorch provides a checkpointing interface to conveniently save and load checkpoints directly to and from a Google Cloud Storage (GCS) bucket.

## Getting started

### Prerequisites
- Python 3.8 or greater is installed (Note: Using 3.12+ is not recommended as PyTorch does not support).

### Installation

```shell
pip install gcs-torch-dataflux
```

### Configuration
Authentication must be provided to use the Accelerated Dataloader for PyTorch via [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) through one of the following methods:
1. While running this library on a GCE VM, Application Default Credentials will automatically use the VM’s attached service account by default. More details can be found [here](https://cloud.google.com/compute/docs/access/app-authentication-methods).
2. Application Default Credentials can also be configured manually as described [here](https://cloud.google.com/docs/authentication/application-default-credentials). The quickest way is to log in directly using the gcloud CLI:
```shell
gcloud auth application-default login
```

### Examples
Please checkout the `demo` directory for a complete set of examples, which includes a [simple starter Jupyter Notebook (hosted by Google Colab)](demo/simple-walkthrough/Getting%20Started%20with%20Dataflux%20Dataset%20for%20PyTorch%20with%20Google%20Cloud%20Storage.ipynb) and an [end-to-end image segmentation training workload walkthrough](demo/image-segmentation/README.md). Those examples will help you understand how the Accelerated Dataloader for PyTorch works and how you can integrate it into your own workload.

#### Sample Examples
Before getting started, please make sure you have installed the library and configured authentication following the instructions above.

##### Data Loading
Both map-style and iterable-style datasets for PyTorch can be constructed by specifying the project name, bucket name and an optional prefix.
```python
from dataflux_pytorch import dataflux_mapstyle_dataset

# Please update these fields.
PROJECT_NAME = "<PROJECT_NAME>"
BUCKET_NAME = "<BUCKET_NAME>"
PREFIX = "<PREFIX>"

# Map-style dataset.
map_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
)

# Random access to an object.
sample_object = map_dataset.objects[0]

# Learn about the name and the size (in bytes) of the object.
name = sample_object[0]
size = sample_object[1]

# Iterate over the datasets.
for each_object in map_dataset:
  # Raw bytes of the object.
  print(each_object)
```
Similarly, for `DataFluxIterableDataset`:
```python
from dataflux_pytorch import dataflux_iterable_dataset

# Iterable-style dataset.
iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
  project_name=PROJECT_ID,
  bucket_name=BUCKET_NAME,
  config=dataflux_iterable_dataset.Config(prefix=PREFIX),
)

for each_object in iterable_dataset:
  # Raw bytes of the object.
  print(each_object)
```

The Accelerated Dataloader for PyTorch offers the flexibility to transform the downloaded raw bytes of data into any format of choice. This is particularly useful since the `PyTorch DataLoader` works well with Numpy arrays or PyTorch tensors.

```python
# Assume that you have a bucket with image files and you want
# to load them into Numpy arrays for training.
import io
import numpy
from PIL import Image

transform = lambda img_in_bytes : numpy.asarray(Image.open(io.BytesIO(img_in_bytes)))

map_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
  data_format_fn=transform,
)

for each_object in map_dataset:
  # each_object is now a Numpy array.
  print(each_object)
```

Similarly, for `DataFluxIterableDataset`:
```python
iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
  project_name=PROJECT_ID,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
  data_format_fn=transform,
)
for each_object in dataset:
  # each_object is now a Numpy array.
  print(each_object)
```

##### Checkpointing

The Accelerated Dataloader for PyTorch supports fast data loading and allows the user to save and load model checkpoints directly to/from a Google Cloud Storage (GCS) bucket.

```python
import torch
import torchvision

from dataflux_pytorch import dataflux_checkpoint

ckpt = dataflux_checkpoint.DatafluxCheckpoint(
  project_name=PROJECT_NAME, bucket_name=BUCKET_NAME
)
CKPT_PATH = "checkpoints/epoch0.ckpt"

model = torchvision.models.resnet50()

with ckpt.writer(CKPT_PATH) as writer:
  torch.save(model.state_dict(), writer)

with ckpt.reader(CKPT_PATH) as reader:
  read_state_dict = torch.load(reader)

model.load_state_dict(read_state_dict)
```

##### Lightning Checkpointing

The Accelerated Dataloader for PyTorch also provides an integration for PyTorch Lightning, featuring a DatafluxLightningCheckpoint, an implementation of PyTorch Lightning's CheckpointIO.

End to end example and the notebook for the PyTorch Lightning integration can be found in the [demo/lightning](https://github.com/GoogleCloudPlatform/dataflux-pytorch/tree/main/demo/lightning) directory.

```python
from lightning import Trainer
from lightning.pytorch.demos import WikiText2, LightningTransformer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint

CKPT = "gcs://BUCKET_NAME/checkpoints/ckpt.ckpt"
dataflux_ckpt = DatafluxLightningCheckpoint(project_name=PROJECT_NAME, bucket_name=BUCKET_NAME)

dataset = WikiText2()
dataloader = DataLoader(dataset, num_workers=1)
model = LightningTransformer(vocab_size=dataset.vocab_size)
trainer = Trainer(
    plugins=[dataflux_ckpt],
    min_epochs=4,
    max_epochs=5,
    max_steps=10,
    accelerator="cpu",
)

trainer.fit(model, dataloader)
trainer.save_checkpoint(CKPT)
```

## Performance

### Map-style Dataset
We tested the Map-style Dataset's early performance using [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) simulations with standard mean file-sizes and dataset sizes. A total of 5 training epochs were simulated. For small files (100KB, 500KB), the Accelerated Dataloader for PyTorch can be **2-3x** faster than using GCS native APIs.

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>File size / count</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Tool</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Training time (s)</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>100 KiB / 500000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">1,299
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Accelerated Dataloader Map-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>515</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>500 KiB / 2.2m files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">6,499
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Accelerated Dataloader Map-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>2,058</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>3 MiB / 50000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">399
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Accelerated Dataloader Map-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>277</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>150 MiB / 5000 files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">1,396
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Accelerated Dataloader Map-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>1,173</strong>
   </td>
  </tr>
</table>

### Iterable-style Dataset
Since the [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) doesn’t easily support an implementation of a PyTorch iterable dataset, we implemented a [simple training loop](demo/simple_iterable_dataset.py) that has similar IO behaviors as the DLIO benchmark and used that loop to benchmark the Iterable Datasets.

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>File size / count</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Tool</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Training time (s)</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>100 KiB / 500000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">1,145
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Accelerated Dataloader Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>611</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>500 KiB / 2.2m files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">5,174
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Accelerated Dataloader Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>2,503</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>3 MiB / 50000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">413
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Accelerated Dataloader Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>384</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>150 MiB / 5000 files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">1,225
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Accelerated Dataloader Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>1,143</strong>
   </td>
  </tr>
</table>

*Note: Within each experiment, all training parameters such as batch size and parallelism are consistent. The team is working on publishing a detailed analysis soon.*

## Limitations

### Billing
To optimize listing performance, the Accelerated Dataloader for PyTorch library utilizes a “fast listing” algorithm developed to balance the listing workload among parallelized GCS object listing processes. Therefore, the library will cause more listing API calls to be made than a regular sequential listing, which are charged as [Class A operations](https://cloud.google.com/storage/pricing).

### Composite Objects
To optimize the download performance of small files, the Accelerated Dataloader for PyTorch library utilizes the [GCS Compose API](https://cloud.google.com/storage/docs/json_api/v1/objects/compose) to concatenate a set of smaller objects into a new and larger one in the same bucket under a folder named “dataflux-composed-objects”. The new composite objects will be removed at the end of your training loop but in rare cases that they don’t, you can run this command to clean the composed files.
``` shell
gcloud storage rm --recursive gs://<my-bucket>/dataflux-composed-objects/
```

You can turn of this behavior by setting the "disable_compose" parameter to true, or by setting the “max_composite_object_size” parameter to 0 when constructing the dataset. Example:
```python
dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(
    prefix=PREFIX,
    max_composite_object_size=0,
    disable_compose=True,
  ),
)
```

Note that turning off this behavior may cause the training loop to take significantly longer to complete when working with small files. However, composed download will hit QPS and throughput limits at a lower scale than downloading files directly, so you should disable this behavior when running at high multi-node scales where you are able to hit project QPS or throughput limits without composed download.


### Soft Delete
To avoid storage charges for retaining the temporary composite objects, consider disabling the [Soft Delete](https://cloud.google.com/storage/docs/soft-delete) retention duration on the bucket.

### Google Cloud Storage Class
Due to the quick creation and deletion of composite objects, we recommend that only the [Standard storage class](https://cloud.google.com/storage/docs/storage-classes) is applied to your bucket to minimize cost and maximize performance.

### Throughput and QPS Limits
Many machine learning efforts opt for a highly distributed training model leveraging tools such as Pytorch Lightning and Ray. These models are compatible with Dataflux, but can often trigger the rate limits of Google Cloud Storage. This typically manifest in a 429 error, or slower than expected speeds while running distributed operations. Details on specific quotas and limits can be found [here](https://cloud.google.com/storage/quotas).

#### Egress Throughput Limits
429 errors acompanied with messages indicating `This workload is drawing too much egress bandwidth from Google Cloud Storage` or `triggered the Cloud Storage Egress Bandwidth Cap` indicate that the data throughput rate of your workload is exceeding the maximum capacity of your Google Cloud Project. To address these issues, you can take the following steps:

1. Check that other workloads executing within your project are not drawing excess bandwidth. Bandwidth usage can be viewed by following steps [here](https://cloud.google.com/storage/docs/bandwidth-usage#bandwidth-monitoring).
2. Follow [these instructions](https://cloud.google.com/storage/docs/bandwidth-usage#increase) to apply for a quota increae of up to 1 Tbps.
3. If more than 1 Tpbs is required, contact your Technical Account Manager or Google representative to file a request on your behalf. This request should specify that you wish to increase the GCS bandwidth caps on your project.

#### QPS Limits
QPS limits can trigger 429 errors with a body message indicating `Too many Requests`, but more commonly manifest in slower than expected execution times. QPS bottlenecks are more common when operating on high volumes of small files. Note that bucket QPS limits will [naturally scale over time](https://cloud.google.com/storage/docs/request-rate#best-practices), so allowing a grace period for warmup can often lead to faster performance. To get more detail on the performance of a target bucket, look at the `Observability` tab when viewing your bucket from the Cloud Console.

If your workload is failing with messages similar to `TooManyRequests: 429 GET https://storage.googleapis.com/download/storage/v1/b/<MY-BUCKET>/o/dataflux-composed-objects%2Fa80727ae-7bc9-4ba3-8f9b-13ff001d6418` that contain the "dataflux-composed-objects" keyword, [disabling](#composite-objects) composed objects is the best first troubleshooting step. This can reduce QPS load brought on by the compose API when used at scale.

## Contributing
We welcome your feedback, issues, and bug fixes. We have a tight roadmap at this time so if you have a major feature or change in functionality you'd like to contribute, please open a GitHub Issue for discussion prior to sending a pull request. Please see [CONTRIBUTING](docs/contributing.md) for more information on how to report bugs or submit pull requests.

### Code of Conduct

This project has adopted the Google Open Source Code of Conduct. Please see [code-of-conduct.md](docs/code-of-conduct.md) for more information.

### License

The Accelerated Dataloader for PyTorch library has an Apache License 2.0. Please see the [LICENSE](LICENSE) file for more information.
