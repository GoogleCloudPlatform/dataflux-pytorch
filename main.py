import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import datasets
from datasets import load_dataset


def main():
    ds_name = "imagenet-es"
    dataset = load_dataset("MMInstruction/M3IT-80", ds_name)
    dataset.save_to_disk("/mnt/disks/ssd-array/raw-dataset")


if __name__ == "__main__":
    main()
