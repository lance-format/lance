# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Dict, Optional

import pyarrow as pa

from lance.file import LanceFileReader, LanceFileWriter


class PqModel:
    """A class that represents a Product Quantization model.

    Can be saved / loaded to checkpoint progress.
    """

    def __init__(
        self, num_subvectors: int, codebook: pa.FixedSizeListArray, *, num_bits: int = 8
    ):
        self.num_subvectors = num_subvectors
        """The number of subvectors to divide source vectors into"""
        self.num_bits = num_bits
        """The number of bits used to encode each PQ centroid"""
        self.codebook = codebook
        """The centroids of the PQ clusters"""

    @property
    def dimension(self):
        """The dimension of the vectors this model was trained on"""
        return self.codebook.type.list_size

    def save(self, uri: str, *, storage_options: Optional[Dict[str, str]] = None):
        """
        Save the PQ model to a lance file.

        Parameters
        ----------

        uri: str
            The URI to save the model to.  The URI can be a local file path or a
            cloud storage path.
        storage_options : optional, dict
            Extra options for the storage backend (e.g. S3 credentials).
        """
        with LanceFileWriter(
            uri,
            pa.schema(
                [pa.field("codebook", self.codebook.type)],
                metadata={
                    b"num_subvectors": str(self.num_subvectors).encode(),
                    b"num_bits": str(self.num_bits).encode(),
                },
            ),
            storage_options=storage_options,
        ) as writer:
            batch = pa.table([self.codebook], names=["codebook"])
            writer.write_batch(batch)

    @classmethod
    def load(cls, uri: str, *, storage_options: Optional[Dict[str, str]] = None):
        """
        Load a PQ model from a lance file.

        Parameters
        ----------

        uri: str
            The URI to load the model from.  The URI can be a local file path or a
            cloud storage path.
        storage_options : optional, dict
            Extra options for the storage backend (e.g. S3 credentials).
        """
        reader = LanceFileReader(uri, storage_options=storage_options)
        num_rows = reader.metadata().num_rows
        metadata = reader.metadata().schema.metadata or {}
        num_subvectors = int(metadata[b"num_subvectors"].decode())
        num_bits = int(metadata.get(b"num_bits", b"8").decode())
        codebook = (
            reader.read_all(batch_size=num_rows).to_table().column("codebook").chunk(0)
        )
        return cls(num_subvectors, codebook, num_bits=num_bits)
