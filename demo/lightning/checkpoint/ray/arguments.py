"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
import os

PARSER = argparse.ArgumentParser(description="Ray-Lightning")

PARSER.add_argument("--log_dir", dest="log_dir", type=str, default="/tmp")
PARSER.add_argument("--epochs", dest="epochs", type=int, default=5)
PARSER.add_argument("--batch_size", dest="batch_size", type=int, default=2)
PARSER.add_argument("--num_workers", dest="num_workers", type=int, default=8)
PARSER.add_argument("--run_name", dest="run_name", type=str, default="")
PARSER.add_argument("--default_root_dir", dest="default_root_dir", type=str, default="/tmp")

PARSER.add_argument(
    "--gcp_project",
    dest="gcp_project",
    type=str,
    default=False,
)

PARSER.add_argument(
    "--gcs_bucket",
    dest="gcs_bucket",
    type=str,
    default=False,
)
