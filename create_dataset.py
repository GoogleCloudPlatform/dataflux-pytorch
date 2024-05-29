from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

# https://cloud.google.com/storage/docs/retry-strategy#python.
MODIFIED_RETRY = DEFAULT_RETRY.with_deadline(300.0).with_delay(
    initial=1.0, multiplier=1.2, maximum=45.0
)


# def main():
#     storage_client = storage.Client()
#     source_bucket = storage_client.bucket("xai-hf-dataset-parquet-may-19")
#     destination_bucket = storage_client.bucket("xai-hf-dataset-parquet-may-19")

#     for i in range(529033):
#         source_name = "000000.parquet"

#         # This file represents the P10 of all files.
#         if (i + 1) % 9 == 0:
#             source_name = "000008.parquet"

#         destination_name = "{:06d}".format(i) + ".parquet"

#         source_blob = source_bucket.blob(source_name)
#         source_bucket.copy_blob(
#             source_blob, destination_bucket, destination_name, retry=MODIFIED_RETRY
#         )
#         print(f"Copied to destination object {destination_name}")


# if __name__ == "__main__":
#     main()


def copy_blob_worker(workload, source_bucket_name, destination_bucket_name):
    """Worker function to copy blobs within its assigned workload."""

    storage_client = storage.Client()
    source_bucket = storage_client.bucket(source_bucket_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    for i in workload:
        source_name = "000000.parquet"
        if (i + 1) % 9 == 0:
            source_name = "000008.parquet"
        destination_name = f"{i:06d}.parquet"

        source_blob = source_bucket.blob(source_name)
        source_bucket.copy_blob(
            source_blob, destination_bucket, destination_name, retry=MODIFIED_RETRY
        )
        print(f"Copied to destination object {destination_name}")


if __name__ == "__main__":
    import multiprocessing

    # Replace with your actual bucket names
    source_bucket_name = "xai-hf-dataset-parquet-may-19"
    destination_bucket_name = "xai-hf-dataset-parquet-may-19"

    num_processes = 128
    total_files = 529033

    # Split workload into chunks for each process
    chunk_size = total_files // num_processes
    workloads = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else total_files
        workloads.append(range(start, end))

    # Create and start processes
    processes = []
    for workload in workloads:
        process = multiprocessing.Process(
            target=copy_blob_worker,
            args=(workload, source_bucket_name, destination_bucket_name),
        )
        processes.append(process)
        process.start()

    # Wait for processes to finish
    for process in processes:
        process.join()
