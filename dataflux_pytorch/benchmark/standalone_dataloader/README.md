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
    export DATA_LOADER_STRATEGY_NAME="FileParallelSequentialRead"
    export GCS_METRICS_BUCKET="distributed-training-metrics"

    export COMMON_RUN_FLAGS="enable_checkpointing=False hardware=cpu"
    export BENCHMARK_RUN_FLAGS="run_name=${RUN_NAME} dataset_directory=${DATASET_DIRECTORY} epochs=${EPOCHS} max_steps=${MAX_STEPS} local_batch_size=${LOCAL_BATCH_SIZE} prefetch_factor=${PREFETCH_FACTOR} data_loader_num_workers=${DATA_LOADER_NUM_WORKERS} per_step_interval=${PER_STEP_INTERVAL} data_loader_strategy_name=${DATA_LOADER_STRATEGY_NAME} gcs_metrics_bucket=${GCS_METRICS_BUCKET}"                  
    ```
3.  Run the trainer:
    ```sh
    JAX_PLATFORMS=cpu python3 standalone_dataloader.py maxtext/MaxText/configs/base.yml ${BENCHMARK_RUN_FLAGS} ${COMMON_RUN_FLAGS}
    ```
