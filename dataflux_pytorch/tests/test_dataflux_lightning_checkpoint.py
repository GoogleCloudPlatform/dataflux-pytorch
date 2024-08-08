import unittest
from typing import Any, Dict
from pathlib import Path
import torch

from dataflux_client_python.dataflux_core.tests import fake_gcs
from dataflux_pytorch.lightning.dataflux_lightning_checkpoint import \
    DatafluxLightningCheckpoint


class LightningCheckpointTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.project_name = "foo"
        self.bucket_name = "fake_bucket"
        self.object_name = "testObject"

        client = fake_gcs.Client()
        self.ckpt = DatafluxLightningCheckpoint(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            storage_client=client,
        )
        self.bucket = fake_gcs.Bucket("fake_bucket")

    def test_invalid_string_path_save(self):
        ckpt_path = "fake_bucket/checkpoint.ckpt"
        try:
            self.ckpt.save_checkpoint(None, ckpt_path)
        except:
            return
        self.fail("Saving with an invalid path did not fail")

    def test_invalid_object_path_save(self):
        ckpt_path = Path("fake_bucket/checkpoint.ckpt")
        try:
            self.ckpt.save_checkpoint(None, ckpt_path)
        except:
            return
        self.fail("Saving with an invalid path did not fail")

    def test_invalid_string_path_load(self):
        ckpt_path = "fake_bucket/checkpoint.ckpt"
        try:
            self.ckpt.load_checkpoint(ckpt_path)
        except:
            return
        self.fail("Loading with an invalid path did not fail")

    def test_invalidobject_path_load(self):
        ckpt_path = Path("fake_bucket/checkpoint.ckpt")
        try:
            self.ckpt.load_checkpoint(ckpt_path)
        except:
            return
        self.fail("Loading with an invalid path did not fail")

    def test_invalid_string_path_remove(self):
        ckpt_path = "fake_bucket/checkpoint.ckpt"
        try:
            self.ckpt.remove_checkpoint(ckpt_path)
        except:
            return
        self.fail("Removing an invalid path did not fail")

    def test_invalid_object_path_remove(self):
        ckpt_path = Path("fake_bucket/checkpoint.ckpt")
        try:
            self.ckpt.remove_checkpoint(ckpt_path)
        except:
            return
        self.fail("Removing an invalid path did not fail")

    def test_empty_bucket_name(self):
        try:
            self.ckpt._parse_gcs_path("gcs://")
        except:
            return
        self.fail("Empty bucket name expected to fail when parsed")

    def test_invalid_bucket_name(self):
        try:
            self.ckpt._parse_gcs_path("gcs://invalid_bucket/ckpt.ckpt")
        except:
            return
        self.fail(
            "Attempted to save to a path with an unexpect bucket, expects this to fail"
        )

    def test_valid_string_path(self):
        paths = {
            f'gs://{self.bucket_name}/path', f'gcs://{self.bucket_name}/path'
        }
        for p in paths:
            try:
                self.ckpt._parse_gcs_path(p)
            except:
                self.fail(msg=f'Valid path: {p} parsed but failed')

    def test_valid_object_path(self):
        paths = {
            Path(f'gs://{self.bucket_name}/path'),
            Path(f'gcs://{self.bucket_name}/path')
        }
        for p in paths:
            try:
                self.ckpt._parse_gcs_path(p)
            except:
                self.fail(msg=f'Valid path: {p} parsed but failed')

    def test_save_and_load_checkpoint_string_path(self):
        tensor = torch.rand(3, 10, 10)
        ckpt_path = "gcs://fake_bucket/checkpoint.ckpt"
        self.ckpt.save_checkpoint(tensor, ckpt_path)
        loaded_checkpoint = self.ckpt.load_checkpoint(ckpt_path)
        assert torch.equal(tensor, loaded_checkpoint)
        self.assertTrue(
            self.ckpt.storage_client._connection.user_agent.startswith(
                "dataflux"))

    def test_save_and_load_checkpoint_object_path(self):
        tensor = torch.rand(3, 10, 10)
        ckpt_path = Path("gcs://fake_bucket/checkpoint.ckpt")
        self.ckpt.save_checkpoint(tensor, ckpt_path)
        loaded_checkpoint = self.ckpt.load_checkpoint(ckpt_path)
        assert torch.equal(tensor, loaded_checkpoint)
        self.assertTrue(
            self.ckpt.storage_client._connection.user_agent.startswith(
                "dataflux"))

    def test_delete_checkpoint_string_path(self):
        tensor = torch.rand(3, 10, 10)
        ckpt_path = "gcs://fake_bucket/checkpoint.ckpt"
        self.ckpt.save_checkpoint(tensor, ckpt_path)
        loaded_checkpoint = self.ckpt.load_checkpoint(ckpt_path)
        assert torch.equal(tensor, loaded_checkpoint)
        self.ckpt.remove_checkpoint(ckpt_path)

    def test_delete_checkpoint_object_path(self):
        tensor = torch.rand(3, 10, 10)
        ckpt_path = Path("gcs://fake_bucket/checkpoint.ckpt")
        self.ckpt.save_checkpoint(tensor, ckpt_path)
        loaded_checkpoint = self.ckpt.load_checkpoint(ckpt_path)
        assert torch.equal(tensor, loaded_checkpoint)
        self.ckpt.remove_checkpoint(ckpt_path)

    def test_invalid_path_type(self):
        ckpt_path = dict()
        try:
            self.ckpt._parse_gcs_path(ckpt_path)
        except:
            return
        self.fail(
            "Attempted to save to a path with an unexpect bucket, expects this to fail"
        )


if __name__ == "__main__":
    unittest.main()
