# Dataflux Dataset for PyTorch

The Dataflux Dataset for PyTorch is an effort to improve ML-training efficiency when using data stored in GCS for training datasets. Using the Dataflux Dataset for training is up to **3X faster** when the dataset consists of many small files (e.g., 100 - 500 KB).

The Dataflux Dataset for PyTorch implements PyTorch’s [dataset primitive](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that can be used to efficiently load training data from GCS. The library currently supports [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random data access patterns.

Note that the Dataflux Dataset for PyTorch library is in an early preview stage and the team is consistently working on improvements and support for new features.

## Getting started

### Prerequisites
- Python 3.8 or greater is installed (Note: Using 3.12+ is not recommended as PyTorch does not support).

### Installation

```shell
git clone --recurse-submodules https://github.com/GoogleCloudPlatform/dataflux-pytorch
cd dataflux-pytorch
pip install .
```

### Configuration
Authentication must be provided to use the Dataflux Dataset for PyTorch via [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) through one of the following methods:
1. While running this library on a GCE VM, Application Default Credentials will automatically use the VM’s attached service account by default. More details can be found [here](https://cloud.google.com/compute/docs/access/app-authentication-methods).
2. Application Default Credentials can also be configured manually as described [here](https://cloud.google.com/docs/authentication/application-default-credentials). The quickest way is to log in directly using the gcloud CLI:
```shell
gcloud auth application-default login
```

### Examples
Before getting started, please make sure you have installed the library and configured authentication following the instructions above.

Dataflux Dataset for PyTorch can be constructed by specifying the project name, bucket name and an optional prefix.

```python
from dataflux_pytorch import dataflux_mapstyle_dataset

# Please update these fields.
PROJECT_NAME = "<PROJECT_NAME>"
BUCKET_NAME = "<BUCKET_NAME>"
PREFIX = "<PREFIX>"

dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  prefix=PREFIX,
)

# Random access to an object.
sample_object = dataset.objects[0]

# Learn about the name and the size (in bytes) of the object.
name = sample_object[0]
size = sample_object[1]

# Iterate over the datasets.
for each_object in dataset:
  # Raw bytes of the object.
  print(each_object)
```

Dataflux Dataset for PyTorch offers the flexibility to transform the downloaded raw bytes of data into any format of choice. 

```python
# Assume that you have a bucket with image files and you want
# to load them into Numpy arrays for training.
import io
import numpy as np
from PIL import Image

transform = lambda img_in_bytes : np.asarray(Image.open(io.BytesIO(img_in_bytes)))

dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  prefix=PREFIX,
  data_format_fn=transform,
)

for each_object in dataset:
  # each_object is now a Numpy array.
  print(each_object)
```

## Performance
We tested Dataflux's early performance using [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) simulations with standard mean file-sizes and dataset sizes. A total of 5 training epochs were simulated. For small files (100KB, 500KB), Dataflux can be **2-3x** faster than using GCS native APIs.

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
   <td style="background-color: #d9d9d9">2,459
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Dataflux</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>757</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>500 KiB / 2.2m files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">7,472
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Dataflux</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>2,696</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>3 MiB / 50000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">463
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Dataflux</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>318</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>150 MiB / 5000 files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">1,228
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Dataflux</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>1,288</strong>
   </td>
  </tr>
</table>

*Note: Within each experiment, all training parameters such as batch size and parallelism are consistent. The team is working on publishing a detailed analysis soon.*

## Limitations

### Billing
To optimize listing performance, the Dataflux Dataset for PyTorch library utilizes a “fast listing” algorithm developed to balance the listing workload among parallelized GCS object listing processes. Therefore, the library will cause more listing API calls to be made than a regular sequential listing, which are charged as [Class A operations](https://cloud.google.com/storage/pricing).

### Composite Objects
To optimize the download performance of small files, the Dataflux Dataset for PyTorch library utilizes the [GCS Compose API](https://cloud.google.com/storage/docs/json_api/v1/objects/compose) to concatenate a set of smaller objects into a new and larger one in the same bucket under a folder named “dataflux-composed-objects”. The new composite objects will be removed at the end of your training loop but in rare cases that they don’t, you can run this command to clean the composed files.
``` shell
gcloud storage rm --recursive gs://<my-bucket>/dataflux-composed-objects/
```

You can also turn off this behavior by setting the “max_composite_object_size” parameter to 0 when constructing the dataset. Example:

```python
dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  prefix=PREFIX,
  config=dataflux_mapstyle_dataset.Config(max_composite_object_size=0),
)
```

Note that turning off this behavior will cause the training loop to take significantly longer to complete when working with small files.

### Google Cloud Storage Class
Due to the quick creation and deletion of composite objects, we recommend that only the [Standard storage class](https://cloud.google.com/storage/docs/storage-classes) is applied to your bucket to minimize cost and maximize performance.

## Contributing
We welcome your feedback, issues, and bug fixes. We have a tight roadmap at this time so if you have a major feature or change in functionality you'd like to contribute, please open a GitHub Issue for discussion prior to sending a pull request. Please see [CONTRIBUTING](docs/contributing.md) for more information on how to report bugs or submit pull requests.

### Code of Conduct

This project has adopted the Google Open Source Code of Conduct. Please see [code-of-conduct.md](docs/code-of-conduct.md) for more information.

### License

The Dataflux Python Client has an Apache License 2.0. Please see the [LICENSE](LICENSE) file for more information.