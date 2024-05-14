import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import datasets
from datasets import load_dataset


def main():
    df = pd.read_parquet("/mnt/disks/ssd-array/raw-dataset/test.parquet")
    print(df.loc[0])
    print(df.dtypes)
    print(df.shape)


if __name__ == "__main__":
    main()
