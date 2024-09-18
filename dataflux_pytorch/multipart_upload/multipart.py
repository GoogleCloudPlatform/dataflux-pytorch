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
import concurrent.futures

from google.api_core import exceptions
from google.cloud.storage import Client
from google.cloud.storage import Blob
from google.cloud.storage.blob import _get_host_name
from google.cloud.storage.blob import _quote
from google.cloud.storage.constants import _DEFAULT_TIMEOUT
from google.cloud.storage._helpers import _api_core_retry_to_resumable_media_retry
from google.cloud.storage.retry import DEFAULT_RETRY
from google.cloud.storage.transfer_manager import (
    _api_core_retry_to_resumable_media_retry, )
from google.cloud.storage.transfer_manager import _headers_from_metadata
from google.cloud.storage.transfer_manager import _get_pool_class_and_requirements
from google.resumable_media import _helpers

import google_crc32c

from google.resumable_media.requests.upload import XMLMPUContainer
from google.resumable_media.requests.upload import XMLMPUPart
from google.resumable_media.common import DataCorruption

TM_DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024
DEFAULT_MAX_WORKERS = 8

# Constants to be passed in as `worker_type`.
PROCESS = "process"
THREAD = "thread"


def upload_chunks_concurrently_from_bytesio(
    bytesio,
    blob,
    content_type=None,
    chunk_size=TM_DEFAULT_CHUNK_SIZE,
    deadline=None,
    max_workers=DEFAULT_MAX_WORKERS,
    *,
    checksum="crc32c",
    timeout=_DEFAULT_TIMEOUT,
    retry=DEFAULT_RETRY,
):
    """Upload a single BytesIO object in chunks, concurrently.

    This function uses the XML MPU API to initialize an upload and upload a
    file in chunks, concurrently with a worker pool.

    The XML MPU API is significantly different from other uploads; please review
    the documentation at `https://cloud.google.com/storage/docs/multipart-uploads`
    before using this feature.

    The library will attempt to cancel uploads that fail due to an exception.
    If the upload fails in a way that precludes cancellation, such as a
    hardware failure, process termination, or power outage, then the incomplete
    upload may persist indefinitely. To mitigate this, set the
    `AbortIncompleteMultipartUpload` with a nonzero `Age` in bucket lifecycle
    rules, or refer to the XML API documentation linked above to learn more
    about how to list and delete individual downloads.

    ACL information cannot be sent with this function and should be set
    separately with :class:`ObjectACL` methods.

    :type bytesio: str
    :param bytesio:
        An io.BytesIO object containing the data to upload.

    :type blob: :class:`google.cloud.storage.blob.Blob`
    :param blob:
        The blob to which to upload.

    :type content_type: str
    :param content_type: (Optional) Type of content being uploaded.

    :type chunk_size: int
    :param chunk_size:
        The size in bytes of each chunk to send. The optimal chunk size for
        maximum throughput may vary depending on the exact network environment
        and size of the blob. The remote API has restrictions on the minimum
        and maximum size allowable, see: `https://cloud.google.com/storage/quotas#requests`

    :type deadline: int
    :param deadline:
        The number of seconds to wait for all threads to resolve. If the
        deadline is reached, all threads will be terminated regardless of their
        progress and `concurrent.futures.TimeoutError` will be raised. This can
        be left as the default of `None` (no deadline) for most use cases.

    :type max_workers: int
    :param max_workers:
        The maximum number of workers to create to handle the workload.

        How many workers is optimal depends heavily on the specific use case,
        and the default is a conservative number that should work okay in most
        cases without consuming excessive resources.

    :type checksum: str
    :param checksum:
        (Optional) The checksum scheme to use: either "md5", "crc32c" or None.
        Each individual part is checksummed. At present, the selected checksum
        rule is only applied to parts and a separate checksum of the entire
        resulting blob is not computed. Please compute and compare the checksum
        of the file to the resulting blob separately if needed, using the
        "crc32c" algorithm as per the XML MPU documentation.

    :type timeout: float or tuple
    :param timeout:
        (Optional) The amount of time, in seconds, to wait
        for the server response.  See: :ref:`configuring_timeouts`

    :type retry: google.api_core.retry.Retry
    :param retry: (Optional) How to retry the RPC. A None value will disable
        retries. A `google.api_core.retry.Retry` value will enable retries,
        and the object will configure backoff and timeout options. Custom
        predicates (customizable error codes) are not supported for media
        operations such as this one.

        This function does not accept `ConditionalRetryPolicy` values because
        preconditions are not supported by the underlying API call.

        See the retry.py source code and docstrings in this package
        (`google.cloud.storage.retry`) for information on retry types and how
        to configure them.

    :raises: :exc:`concurrent.futures.TimeoutError` if deadline is exceeded.
    """

    bucket = blob.bucket
    client = blob.client
    transport = blob._get_transport(client)

    hostname = _get_host_name(client._connection)
    url = "{hostname}/{bucket}/{blob}".format(hostname=hostname,
                                              bucket=bucket.name,
                                              blob=_quote(blob.name))

    base_headers, object_metadata, content_type = blob._get_upload_arguments(
        client, content_type, filename=None, command="tm.upload_sharded")
    headers = {**base_headers, **_headers_from_metadata(object_metadata)}

    if blob.user_project is not None:
        headers["x-goog-user-project"] = blob.user_project

    # When a Customer Managed Encryption Key is used to encrypt Cloud Storage object
    # at rest, object resource metadata will store the version of the Key Management
    # Service cryptographic material. If a Blob instance with KMS Key metadata set is
    # used to upload a new version of the object then the existing kmsKeyName version
    # value can't be used in the upload request and the client instead ignores it.
    if blob.kms_key_name is not None and "cryptoKeyVersions" not in blob.kms_key_name:
        headers["x-goog-encryption-kms-key-name"] = blob.kms_key_name

    container = XMLMPUContainer(url, None, headers=headers)
    container._retry_strategy = _api_core_retry_to_resumable_media_retry(retry)

    container.initiate(transport=transport, content_type=content_type)
    upload_id = container.upload_id

    view = bytesio.getbuffer()
    size = len(view)
    num_of_parts = -(size // -chunk_size)  # Ceiling division

    futures = []

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        for part_number in range(1, num_of_parts + 1):
            start = (part_number - 1) * chunk_size
            end = min(part_number * chunk_size, size)

            futures.append(
                executor.submit(
                    _buffer_view_upload_part,
                    client,
                    url,
                    upload_id,
                    view,
                    start=start,
                    end=end,
                    part_number=part_number,
                    checksum=checksum,
                    headers=headers,
                    retry=retry,
                ))

        concurrent.futures.wait(futures,
                                timeout=deadline,
                                return_when=concurrent.futures.ALL_COMPLETED)

    try:
        # Harvest results and raise exceptions.
        for future in futures:
            part_number, etag = future.result()
            container.register_part(part_number, etag)

        container.finalize(blob._get_transport(client))
    except Exception:
        container.cancel(blob._get_transport(client))
        raise


class _BufferViewXMLMPUPart(XMLMPUPart):

    def __init__(
        self,
        upload_url,
        upload_id,
        view,
        start,
        end,
        part_number,
        headers=None,
        checksum=None,
    ):
        super().__init__(upload_url, upload_id, None, start, end, part_number,
                         headers, checksum)
        self._view = view

    def _prepare_upload_request(self):
        """Prepare the contents of HTTP request to upload a part.

        This is everything that must be done before a request that doesn't
        require network I/O. This is based on the `sans-I/O`_ philosophy.

        For the time being, this **does require** some form of I/O to read
        a part from ``stream`` (via :func:`get_part_payload`). However, this
        will (almost) certainly not be network I/O.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always PUT)
              * the URL for the request
              * the body of the request
              * headers for the request

            The headers incorporate the ``_headers`` on the current instance.

        Raises:
            ValueError: If the current upload has finished.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("This part has already been uploaded.")

        MPU_PART_QUERY_TEMPLATE = "?partNumber={part}&uploadId={upload_id}"

        payload = bytes(self._view[self._start:self._end])

        self._checksum_object = _helpers._get_checksum_object(
            self._checksum_type)
        if self._checksum_object is not None:
            self._checksum_object.update(payload)

        part_query = MPU_PART_QUERY_TEMPLATE.format(part=self._part_number,
                                                    upload_id=self._upload_id)
        upload_url = self.upload_url + part_query
        return "PUT", upload_url, payload, self._headers


def _buffer_view_upload_part(
    maybe_pickled_client,
    url,
    upload_id,
    view,
    start,
    end,
    part_number,
    checksum,
    headers,
    retry,
):
    """Helper function that runs inside a thread or subprocess to upload a part.

    `maybe_pickled_client` is either a Client (for threads) or a specially
    pickled Client (for processes) because the default pickling mangles Client
    objects."""

    if isinstance(maybe_pickled_client, Client):
        client = maybe_pickled_client
    else:
        client = pickle.loads(maybe_pickled_client)
    part = _BufferViewXMLMPUPart(
        url,
        upload_id,
        view,
        start=start,
        end=end,
        part_number=part_number,
        checksum=checksum,
        headers=headers,
    )
    part._retry_strategy = _api_core_retry_to_resumable_media_retry(retry)
    part.upload(client._http)
    return (part_number, part.etag)
