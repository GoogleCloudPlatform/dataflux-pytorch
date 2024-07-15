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

import io
import unittest

from dataflux_client_python.dataflux_core.tests import fake_gcs
from dataflux_pytorch import dataflux_checkpoint


class CheckpointTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.project_name = "foo"
        self.bucket_name = "bar"
        self.object_name = "testObject"

        client = fake_gcs.Client()
        self.ckpt = dataflux_checkpoint.DatafluxCheckpoint(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            storage_client=client,
        )

    def test_reader(self):
        got_reader = self.ckpt.reader(self.object_name)
        self.assertIsInstance(got_reader, io.BytesIO)
        self.assertTrue(
            self.ckpt.storage_client._connection.user_agent.startswith(
                "dataflux"))

    def test_writer(self):
        got_writer = self.ckpt.writer(self.object_name)
        self.assertIsInstance(got_writer, fake_gcs.FakeBlobWriter)
        self.assertTrue(
            self.ckpt.storage_client._connection.user_agent.startswith(
                "dataflux"))


if __name__ == "__main__":
    unittest.main()
