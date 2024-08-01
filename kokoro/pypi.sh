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
# cd "${KOKORO_ARTIFACTS_DIR}/github/dataflux-pytorch"

function setup_virtual_envs() {
    sudo apt-get -y update

    echo Setting up Python virtual environment.
    sudo apt install -y python3.8-venv
    python3 -m venv venv
    source venv/bin/activate
}

function install_requirements() {
    echo Installing python3-pip.
    sudo apt-get -y install python3-pip

    echo Installing required dependencies.
    pip install .
    python3 -m pip install --upgrade pip
    pip install gcs-torch-dataflux
}

function run_unit_tests() {
    echo Running unit tests.
    python3 -m pytest dataflux_pytorch/tests -vv --junit-xml="${KOKORO_ARTIFACTS_DIR}/unit_tests/sponge_log.xml" --log-cli-level=DEBUG
}

setup_virtual_envs
install_requirements
run_unit_tests
