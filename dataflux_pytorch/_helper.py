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

import logging

import dataflux_core
from google.cloud import storage


def _get_missing_permissions(storage_client: any, bucket_name: str,
                             project_name: str, required_perm: any):
    """Returns a list of missing permissions of the client from the required permissions list."""
    if not storage_client:
        storage_client = storage.Client(project=project_name)
    dataflux_core.user_agent.add_dataflux_user_agent(storage_client)
    bucket = storage_client.bucket(bucket_name)

    try:
        perm = bucket.test_iam_permissions(required_perm)
    except Exception as e:
        logging.exception(f"Error testing permissions: {e}")
        raise

    return [p for p in required_perm if p not in perm]
