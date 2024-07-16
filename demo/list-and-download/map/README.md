# Map-style demo loop

## Running on GKE

1. `docker build -t dataflux-list-and-download .`
2. `docker tag dataflux-list-and-download gcr.io/gcs-tess/dataflux-list-and-download`
3. `docker push gcr.io/gcs-tess/dataflux-list-and-download`
4. `kubectl apply -f demo/list-and-download/map/deployment.yaml`
