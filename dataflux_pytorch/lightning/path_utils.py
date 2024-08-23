from pathlib import Path
from typing import Union, Tuple


def process_input_path(path: Union[str, Path]) -> str:
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        # When casting from Path object to string, it considers cloud URLs as Network URLs and gets rid of //
        scheme, rest = str(path).split(":/")
        return str(scheme) + "://" + str(rest)
    else:
        raise TypeError(
            "path argument must be of type string or pathlib.Path object")


def parse_gcs_path(path: Union[str, Path]) -> Tuple[str, str]:
    if not path:
        raise ValueError("Path cannot be empty")
    input_path = process_input_path(path)
    if not (input_path.startswith("gcs://")
            or input_path.startswith("gs://")):
        raise ValueError("Path needs to begin with gcs:// or gs://")
    input_path = input_path.split("//", maxsplit=1)
    if not input_path or len(input_path) < 2:
        raise ValueError("Bucket name must be non-empty")
    split = input_path[1].split("/", maxsplit=1)
    bucket_name = ""
    if len(split) == 1:
        bucket_name = split[0]
        prefix = ""
    else:
        bucket_name, prefix = split
    if not bucket_name:
        raise ValueError("Bucket name must be non-empty")
    return bucket_name, prefix
