import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import datasets


def main():
    print(
        ft.read_feather("/mnt/disks/ssd-array/dataset/train/data-00000-of-00078.arrow")
    )


if __name__ == "__main__":
    main()
