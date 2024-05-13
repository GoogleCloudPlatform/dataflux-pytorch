import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import datasets


def main():
    ds = datasets.load_from_disk("/mnt/disks/ssd-array/dataset")
    print(ds)

    for train_instance in ds["train"]:
        instruction = train_instance["instruction"]  # str
        inputs = train_instance["inputs"]  # str
        outputs = train_instance["outputs"]  # str
        image_base64_str_list = train_instance["image_base64_str"]  # str (base64)
        print(instruction)
        print(inputs)
        print(outputs)
        print(image_base64_str_list)
        break


if __name__ == "__main__":
    main()
