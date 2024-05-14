import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from datasets import load_dataset


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

    df = pd.read_parquet("/mnt/disks/ssd-array/dataflux-pytorch/output.parquet")
    print(df.loc[0])
    print(df.dtypes)
    print(df.size)


if __name__ == "__main__":
    main()
