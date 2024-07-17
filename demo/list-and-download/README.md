# Demo loops

## Running locally

### Run the map-style loop

1. `python3 -m demo.list-and-download.map.simple_map_style_dataset --project=gcs-tess --bucket=bernardhan-diffusion-small --sleep-per-step=0 --num-workers=48 --epochs=2`

### Run the iterable-style loop

1. `python3 -m demo.list-and-download.iterable.simple_iterable_dataset --project=gcs-tess --bucket=bernardhan-diffusion-small --sleep-per-step=0 --num-workers=48 --epochs=2`

## Running on GKE

### Build the Docker image

In the following commands, update `gcs-tess` to your project ID as needed.

1. `docker build -t dataflux-list-and-download .`
2. `docker tag dataflux-list-and-download gcr.io/gcs-tess/dataflux-list-and-download`
3. `docker push gcr.io/gcs-tess/dataflux-list-and-download`

### Run the map-style loop

1. Update any needed flags/configs in `demo/list-and-download/map/deployment.yaml`
    * Notably the job name, completions/parallelism, image name, and any flags
2. `kubectl apply -f demo/list-and-download/map/deployment.yaml`

### Run the iterable-style loop

1. Update any needed flags/configs in `demo/list-and-download/iterable/deployment.yaml`
    * Notably the job name, completions/parallelism, image name, and any flags
2. `kubectl apply -f demo/list-and-download/iterable/deployment.yaml`
