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

# Fail on any error.
set -e

# Code under repo is checked out to this directory.
cd "${KOKORO_ARTIFACTS_DIR}/github/dataflux-pytorch"

function setup_virtual_envs() {
    sudo apt-get -y update

    echo Setting up Python virtual environment.
    sudo apt install -y python3.8-venv
    python3 -m venv venv
    source venv/bin/activate
}

function run_git_commands() {
    echo Setting git permissions.
    git config --global --add safe.directory "*" 

    echo Installing git submodules.
    git submodule init

    echo Updating git submodules.
    git submodule update
}

function install_requirements() {
    echo Installing python3-pip.
    sudo apt-get -y install python3-pip

    echo Installing requirements.
    pip install -r requirements.txt

    echo Installing parquet demo requirements.
    pip install -r ./demo/lightning/text-based/distributed/requirements.txt

    echo Installing image training demo requirements.
    pip install -r ./demo/lightning/image-segmentation/requirements.txt

    echo Installing required dependencies.
    pip install .
}

function run_benchmarks(){
    echo Running benchmarks.
    python3 -u ./demo/lightning/text-based/distributed/model.py --local --project=dataflux-project --bucket=fineweb-df-benchmark --num-workers=2 --num-nodes=1 --devices=5 --batch-size=512 --epochs=5 --limit-train-batches=1000 --log-level=ERROR;
    python3 -u ./demo/lightning/image-segmentation/train.py --local --gcp_project=dataflux-project --gcs_bucket=dataflux-demo-public --images_prefix=image-segmentation-dataset/images --labels_prefix=image-segmentation-dataset/labels --num_nodes=1 --num_devices=5 --epochs=5;
}

setup_virtual_envs
run_git_commands
install_requirements
run_benchmarks
