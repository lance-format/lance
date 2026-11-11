# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Optional, Union

import pyarrow as pa


class CachedDataset:
    def __init__(
        self,
        stream: Iterable[pa.RecordBatch],
        cache: Optional[Union[str, Path, bool]] = None,
    ):
        # TemporaryDirectory owns its exit finalizer. Retaining this cache in an
        # atexit callback would also retain discarded caches' source streams.
        self.cache_dir: Optional[TemporaryDirectory] = None
        if cache is None or cache is True:
            self.cache_dir = TemporaryDirectory(
                prefix="lance-torch-dataset",
            )
        elif isinstance(cache, (str, Path)):
            self.cache_dir = TemporaryDirectory(
                prefix="lance-torch-dataset",
                dir=cache,
            )
        else:
            raise ValueError(f"Unsupported cache type: {type(cache)}")
        self.cache_file = None
        self.stream = stream
        self.finished_origin_stream = False

    def close(self):
        """Close the dataset and delete tmp files"""
        if self.cache_dir is not None:
            self.cache_dir.cleanup()
            self.cache_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            raise

    def __del__(self):
        self.close()

    def __iter__(self):
        if self.finished_origin_stream and self.cache_file is None:
            return
        if self.cache_file is None:
            # First iteration.
            writer: Optional[pa.ipc.RecordBatchStreamWriter] = None
            try:
                with ExitStack() as resources:
                    for batch in self.stream:
                        if writer is None:
                            self.cache_file = Path(self.cache_dir.name) / "cache.arrow"
                            writer = resources.enter_context(
                                pa.ipc.new_stream(str(self.cache_file), batch.schema)
                            )
                        writer.write(batch)
                        yield batch
                        del batch
                self.finished_origin_stream = True
            finally:
                if not self.finished_origin_stream:
                    # Partial caches cannot be replayed. Close the writer before
                    # removing its directory, including on Windows.
                    self.close()
        else:
            # Follow up iteration
            if not self.finished_origin_stream:
                raise RuntimeError(
                    "CachedDataset: the iteration over original data has not finished"
                )
            reader = pa.ipc.open_stream(self.cache_file)
            for batch in reader:
                yield batch
                del batch
