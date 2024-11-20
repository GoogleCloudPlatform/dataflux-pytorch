# Single-node checkpoint save & async save

The Connector for PyTorch allows the user to save and load model checkpoints directly to/from a Google Cloud Storage (GCS) bucket. This demo presents a simple example of checkpoint saving within a training workload. 

The demo can optionally be run using the `--use-async` flag to enable asynchrenous checkpoint saving. This is useful when training a model that results in a large checkpoint file size and avoid blocking network calls during saving.

## Demo usage

```bash
python demo/checkpointing/train.py --help                                                                       
usage: train.py [-h] --project PROJECT --bucket BUCKET [--use-async] [--embedding-dim EMBEDDING_DIM] [--hidden-dim HIDDEN_DIM] [--output-dim OUTPUT_DIM] [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--learning-rate LEARNING_RATE]

Configuration for single-node async checkpoint demo.

options:
  -h, --help            show this help message and exit
  --project PROJECT     GCS project ID.
  --bucket BUCKET       GCS bucket for saving checkpoints.
  --embedding-dim EMBEDDING_DIM
                        The size of each embedding vector.
  --hidden-dim HIDDEN_DIM
                        The number of features in the hidden state.
  --output-dim OUTPUT_DIM
                        The size of each output sample.
  --batch-size BATCH_SIZE
                        How many samples per batch to load.
  --num-epochs NUM_EPOCHS
                        The number of full passes through training.
  --learning-rate LEARNING_RATE
                        The size of the optimizer steps taken during the training process.
  --use-async           If checkpoint save should use async.
```

Example call

```bash
$ python3 -u demo/checkpointing/train.py \
      --project=<gcs_project_id> \
      --bucket=<bucket_name> \
      --embedding-dim=100 \
      --use-async
```
