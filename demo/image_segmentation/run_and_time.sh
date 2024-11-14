# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e

SEED=${1:--1}

MAX_EPOCHS=100
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=1000
EVALUATE_EVERY=20
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=20
NUM_WORKERS=8
NUM_DATALOADER_THREADS=8
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
SEED=5
IMAGES_PREFIX="{IMAGES_PREFIX}"
LABELS_PREFIX="{LABELS_PREFIX}"
GCP_PROJECT="{YOUR_GCP_PROJECT}"
GCS_BUCKET="{YOUR_COPIED_GCS_BUCKET}"

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

python train.py \
  --num_workers=${NUM_WORKERS} \
  --num_dataloader_threads=${NUM_DATALOADER_THREADS} \
  --epochs=${MAX_EPOCHS} \
  --evaluate_every=${EVALUATE_EVERY} \
  --start_eval_at=${START_EVAL_AT} \
  --quality_threshold=${QUALITY_THRESHOLD} \
  --batch_size=${BATCH_SIZE} \
  --optimizer sgd \
  --ga_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate=${LEARNING_RATE} \
  --seed=${SEED} \
  --lr_warmup_epochs=${LR_WARMUP_EPOCHS} \
  --input_shape 32 32 32 \
  --normalization="batchnorm" \
  --gcp_project=${GCP_PROJECT} \
  --gcs_bucket=${GCS_BUCKET} \
  --images_prefix=${IMAGES_PREFIX} \
  --labels_prefix=${LABELS_PREFIX}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="image_segmentation"

echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"