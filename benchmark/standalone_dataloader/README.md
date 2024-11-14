This benchmark is a work-in-progress.

## Running locally

These instructions are run relative to this directory.

1.  Install requirements:
    ```sh
    pip install -r requirements.txt
    ```
2.  Set environment variables (update as needed):
    ```sh
    export JOB_INDEX=0 JOB_COMPLETION_INDEX=0 PROCESSES_IN_JOB=1 JAX_PROCESS_COUNT=1 JAX_COORDINATOR_ADDRESS=localhost
    export RUN_NAME=$USER-dataflux-maxtext-$(date +"%Y-%m-%d-%H-%M-%S")

    export PROJECT="<YOUR-PROJECT>"
    export BUCKET="<YOUR-BUCKET>"
    export PREFIX="<DATA-PREFIX>"
    export EPOCHS=2
    export MAX_STEPS=-1
    export LOCAL_BATCH_SIZE=32
    export PREFETCH_FACTOR=2
    export DATA_LOADER_NUM_WORKERS=10
    export PER_STEP_INTERVAL=0.1
    export GCS_METRICS_BUCKET="<METRICS-BUCKET>"

    export COMMON_RUN_FLAGS="enable_checkpointing=False hardware=cpu"
    export BENCHMARK_RUN_FLAGS="run_name=${RUN_NAME} dataset_directory=${DATASET_DIRECTORY} epochs=${EPOCHS} max_steps=${MAX_STEPS} local_batch_size=${LOCAL_BATCH_SIZE} prefetch_factor=${PREFETCH_FACTOR} data_loader_num_workers=${DATA_LOADER_NUM_WORKERS} per_step_interval=${PER_STEP_INTERVAL} gcs_metrics_bucket=${GCS_METRICS_BUCKET}"                  
    ```
3.  Run the trainer:
    ```sh
    JAX_PLATFORMS=cpu python3 standalone_dataloader.py maxtext/MaxText/configs/base.yml ${BENCHMARK_RUN_FLAGS} ${COMMON_RUN_FLAGS}
    ```

## Running on GKE

### Build the Docker image

In the following commands, update `gcs-tess` to your project ID as needed.

1. `docker build -t dataflux-list-and-download .`
2. `docker tag dataflux-list-and-download gcr.io/gcs-tess/dataflux-maxtext`
3. `docker push gcr.io/gcs-tess/dataflux-maxtext`

### Run the benchmark

1. Update any needed flags/configs in `benchmark/standalone_dataloader/deployment.yaml`
    * Notably the job name, completions/parallelism, image name, and any flags
2. `kubectl apply -f benchmark/standalone_dataloader/deployment.yaml`
