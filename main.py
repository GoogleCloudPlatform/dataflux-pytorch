import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from base64 import b64decode
from io import BytesIO
from PIL import Image
from datasets import load_dataset

import shutil


def main():
    # files = [
    #     "/mnt/disks/ssd-array/raw-dataset/test.parquet",
    #     "/mnt/disks/ssd-array/raw-dataset/test1.parquet",
    #     "/mnt/disks/ssd-array/raw-dataset/test2.parquet",
    #     "/mnt/disks/ssd-array/raw-dataset/test3.parquet",
    # ]

    # schema = pq.ParquetFile(files[0]).schema_arrow
    # with pq.ParquetWriter("output.parquet", schema=schema) as writer:
    #     for file in files:
    #         writer.write_table(pq.read_table(file, schema=schema))

    # df = pd.read_parquet("/mnt/disks/ssd-array/dataflux-pytorch/output.parquet")
    # print(df.loc[0]["image_base64_str"][0])
    # image_0 = Image.open(BytesIO(b64decode(df.loc[0]["image_base64_str"][0])))
    # print(image_0)
    # print(df.dtypes)
    # print(df.shape)

    source = "/mnt/disks/ssd-array/dataflux-pytorch/output.parquet"

    for i in range(10000):
        destination = "/mnt/disks/ssd-array/dataflux-pytorch/generated-data/" + (
            "{:04d}".format(i) + ".parquet"
        )

        shutil.copyfile(source, destination)

        if i == 10:
            break


if __name__ == "__main__":
    main()
