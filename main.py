import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import datasets
from datasets import load_dataset


def main():
    ds = pd.read_parquet("/mnt/disks/ssd-array/raw-dataset/train/0001.parquet")
    print(ds.size)


if __name__ == "__main__":
    main()
