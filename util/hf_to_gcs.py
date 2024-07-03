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

from huggingface_hub import snapshot_download

PARSER = argparse.ArgumentParser(
    description="Copy files from HuggingFace to GCS.")

PARSER.add_argument("--repo_id", dest="repo_id", type=str, required=True)
PARSER.add_argument("--repo_type", dest="repo_type",
                    type=str, default="dataset")
PARSER.add_argument("--revision", dest="revision", type=str)
PARSER.add_argument("--dest", dest="dest", type=str,
                    required=True, help="gcs://<bucket>/<path>")
PARSER.add_argument("--allow_patterns", dest="allow_patterns", type=str)
PARSER.add_argument("--ignore_patterns",
                    dest="ignore_patterns", type=str)
PARSER.add_argument("--max_workers", dest="max_workers", type=int, default=8)


def download(flags):
    """Downloads files from HuggingFace to a GCS location."""
    snapshot_download(repo_id=flags.repo_id, repo_type=flags.repo_type, revision=flags.revision, local_dir=flags.dest,
                      allow_patterns=flags.allow_patterns, ignore_patterns=flags.ignore_patterns, max_workers=flags.max_workers)


if __name__ == "__main__":
    flags = PARSER.parse_args()
    download(flags)
