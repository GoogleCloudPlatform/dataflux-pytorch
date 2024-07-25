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

IMAGES_PREFIX={PATH-TO-IMAGES}
LABELS_PREFIX={PATH-TO-LABELS}
GCP_PROJECT={YOUR-PROJECT}
GCS_BUCKET={YOUR-BUCKET}

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

python3  train.py \
  --gcp_project=${GCP_PROJECT} \
  --images_prefix=${IMAGES_PREFIX} \
  --labels_prefix=${LABELS_PREFIX} \
  --gcs_bucket=${GCS_BUCKET}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="image_segmentation"

echo "RESULT,$result_name,$result,$USER,$start_fmt"
