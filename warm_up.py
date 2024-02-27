from dataflux_pytorch import dataflux_mapstyle_dataset
from collections.abc import Sequence
from absl import flags, app
import logging
from torch.utils import data
import time
import os

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "project",
    "zimbruplayground",
    "the name of the GCP project. If not specified, it will be set to gcs-tess.",
)
flags.DEFINE_string("bucket", "", "the name of the bucket")
flags.DEFINE_integer(
    "warm_up_hrs",
    10,
    "the hours to warm up the bucket for. This is directly set to the number of epochs run because we will sleep for an hour between epochs",
)


def main(argv: Sequence[str]) -> None:
    # To yield the warnings coming from the GCS compose operation.
    logging.getLogger().setLevel(logging.ERROR)

    ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
        project_name=FLAGS.project,
        bucket_name=FLAGS.bucket,
        config=dataflux_mapstyle_dataset.Config(
            # perform download on only the smaller objects to get close to the
            # 5000 reads/s setup.
            prefix="UNet3D/micro/100KB-50GB/train"
        ),
    )

    print(len(ds.objects))
    data_loader = data.DataLoader(
        dataset=ds,
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count(),
        prefetch_factor=2,
    )

    for i in range(FLAGS.warm_up_hrs):
        total_objects = 0
        epoch_start = time.time()
        for batch in data_loader:
            total_objects += len(batch)
        epoch_end = time.time()
        print(
            f"Epoch {i} took {epoch_end - epoch_start} seconds to iterate over {total_objects} objects."
        )
        if i != FLAGS.warm_up_hrs - 1:
            time.sleep(60 * 60)  # one hour


if __name__ == "__main__":
    app.run(main)
