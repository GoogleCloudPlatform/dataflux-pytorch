docker build -t yashsha-container .
docker tag yashsha-container gcr.io/gcs-tess/yashsha-container
docker push gcr.io/gcs-tess/yashsha-container
kubectl delete jobset yashsha-temp-4
kubectl apply -f demo/lightning/checkpoint/multinode/benchmark-deploy.yaml
