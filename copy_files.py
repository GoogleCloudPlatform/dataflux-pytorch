from google.cloud import storage


def main():
    storage_client = storage.Client()
    source_bucket = storage_client.bucket("xai-hf-dataset-parquet")
    destination_bucket = storage_client.bucket("xai-hf-dataset-parquet")

    for i in range(371):
        source_name = "{:04d}".format(i) + ".parquet"
        destination_name = "{:04d}".format(i + 371) + ".parquet"

        source_blob = source_bucket.blob(source_name)
        source_bucket.copy_blob(source_blob, destination_bucket, destination_name)
        print(f"Copied to destination object {destination_name}")


if __name__ == "__main__":
    main()
