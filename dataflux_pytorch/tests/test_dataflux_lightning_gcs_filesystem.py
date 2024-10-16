import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

from dataflux_client_python.dataflux_core.tests import fake_gcs
from dataflux_pytorch.lightning.gcs_filesystem import GCSFileSystem


class GCSFileSystemTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.project_name = "project_name"
        self.bucket_name = "fake_bucket"
        self.bucket = fake_gcs.Bucket("fake_bucket")
        self.client = fake_gcs.Client()
        self.fake_gcs = GCSFileSystem(project_name=self.project_name,
                                      debug=False,
                                      storage_client=self.client)

    def test_create_stream_invalid_path_string(self):
        path = "fake_bucket/checkpoint.ckpt"
        # Expect a ValueError since parse_gcs_path should fail
        with self.assertRaises(ValueError):
            with self.fake_gcs.create_stream(path, "wb") as stream:
                pass

    def test_create_stream_invalid_path_object(self):
        path = Path("fake_bucket/checkpoint.ckpt")
        with self.assertRaises(ValueError):
            with self.fake_gcs.create_stream(path, "wb") as stream:
                pass

    @mock.patch(
        "dataflux_pytorch.lightning.gcs_filesystem.DatafluxCheckpointBuffer.close"
    )
    def test_create_stream_valid_path_string_write_mode(self, mock_close):
        path = f'gs://{self.bucket_name}/some-dir/'
        with self.fake_gcs.create_stream(path, "wb") as stream:
            pass

    @mock.patch(
        "dataflux_pytorch.lightning.gcs_filesystem.DatafluxCheckpointBuffer.close"
    )
    def test_create_stream_valid_path_object_write_mode(self, mock_close):
        path = Path(f'gs://{self.bucket_name}/some-dir/')
        with self.fake_gcs.create_stream(path, "wb") as stream:
            pass

    def test_create_stream_valid_path_string_read_mode(self):
        path = f'gs://{self.bucket_name}/some-dir/'
        with self.fake_gcs.create_stream(path, "rb") as stream:
            pass

    def test_create_stream_valid_path_object_read_mode(self):
        path = Path(f'gs://{self.bucket_name}/some-dir/')
        with self.fake_gcs.create_stream(path, "rb") as stream:
            pass

    def test_create_stream_valid_path_object_invalid_mode(self):
        path = Path(f'gs://{self.bucket_name}/some-dir/')
        with self.assertRaises(ValueError) as context:
            with self.fake_gcs.create_stream(path, "invalid_mode"):
                pass

        self.assertEqual(
            str(context.exception),
            "Invalid mode argument, create_stream only supports rb (read mode) & wb (write mode)"
        )

    def test_create_stream_valid_path_string_invalid_mode(self):
        path = f'gs://{self.bucket_name}/some-dir/'
        with self.assertRaises(ValueError) as context:
            with self.fake_gcs.create_stream(path, "invalid_mode"):
                pass
        self.assertEqual(
            str(context.exception),
            "Invalid mode argument, create_stream only supports rb (read mode) & wb (write mode)"
        )

    def test_concat_path_with_string(self):
        path = "/base/path"
        suffix = "file.txt"
        expected = Path("/base/path/file.txt")

        result = self.fake_gcs.concat_path(path, suffix)
        self.assertEqual(result, expected)

    def test_concat_path_with_path_object(self):
        path = Path("/base/path")
        suffix = "file.txt"
        expected = Path("/base/path/file.txt")

        result = self.fake_gcs.concat_path(path, suffix)
        self.assertEqual(result, expected)

    def test_init_path_with_string(self):
        path_str = "/some/path"
        expected_path = Path(path_str)
        result = self.fake_gcs.init_path(path_str)

        self.assertEqual(result, expected_path)

    def test_init_path_with_path_object(self):
        original_path = Path("/some/path")
        result = self.fake_gcs.init_path(original_path)
        self.assertIs(result, original_path)

    def test_exists_invalid_path_string(self):
        path = "fake_bucket/checkpoint.ckpt"
        try:
            self.fake_gcs.exists(path)
        except:
            return
        self.fail("Test with an invalid path did not fail")

    def test_exists_missing_path_object(self):
        path = Path(f'fake_bucket/checkpoint.ckpt')
        try:
            self.fake_gcs.exists(path)
        except:
            return
        self.fail("Test with an invalid path did not fail")

    def test_exists_valid_path_string(self):
        path = f'gs://{self.bucket_name}/'
        ans = self.fake_gcs.exists(path)
        self.assertTrue(ans)

    def test_exists_missing_path_string(self):
        path = f'gs://missing-bucket/missing-path'
        self.bucket = fake_gcs.Bucket("fake_bucket")
        ans = self.fake_gcs.exists(path)
        self.assertFalse(ans)

    def test_rm_file_invalid_path_string(self):
        path = "fake_bucket/checkpoint.ckpt"
        try:
            self.fake_gcs.rm_file(path)
        except:
            return
        self.fail("Test with an invalid path did not fail")

    def test_rm_file_invalid_path_object(self):
        path = Path("fake_bucket/checkpoint.ckpt")
        try:
            self.fake_gcs.rm_file(path)
        except:
            return
        self.fail("Test with an invalid path did not fail")

    def test_rm_file_valid_path_string(self):
        path = "gs://fake_bucket/checkpoint.ckpt"
        try:
            self.fake_gcs.rm_file(path)
        except:
            self.fail("Test with an valid path failed")

    def test_rm_file_valid_path_object(self):
        path = Path("gs://fake_bucket/checkpoint.ckpt")
        try:
            self.fake_gcs.rm_file(path)
        except:
            self.fail("Test with an valid path failed")

    def test_validate_checkpoint_ids_valid_object(self):
        path = Path('some-valid-path')
        ans = GCSFileSystem.validate_checkpoint_id(path)
        self.assertTrue(ans)

    def test_validate_checkpoint_ids_valid_string(self):
        path = f'gs://{self.bucket_name}'
        ans = GCSFileSystem.validate_checkpoint_id(path)
        self.assertTrue(ans)

    def test_validate_checkpoint_ids_invalid_string(self):
        path = f'{self.bucket_name}'
        try:
            GCSFileSystem.validate_checkpoint_id(path)
        except:
            return
        self.fail("Test with an invalid path did not fail")
