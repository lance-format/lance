# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum

from .. import lance as _lance
from .builder import IndexConfig, IndicesBuilder
from .ivf import IvfModel
from .pq import PqModel

IndexSegmentDescription = _lance.indices.IndexSegmentDescription
IndexSegment = _lance.indices.IndexSegment
IndexScanCoverage = _lance.indices.IndexScanCoverage
IndexScanIvfStatistics = _lance.indices.IndexScanIvfStatistics
IndexScanSegmentStatistics = _lance.indices.IndexScanSegmentStatistics
IndexScanStatistics = _lance.indices.IndexScanStatistics

__all__ = [
    "IndicesBuilder",
    "IndexConfig",
    "PqModel",
    "IvfModel",
    "IndexFileVersion",
    "IndexSegment",
    "IndexSegmentDescription",
    "IndexScanCoverage",
    "IndexScanIvfStatistics",
    "IndexScanSegmentStatistics",
    "IndexScanStatistics",
]


class IndexFileVersion(str, Enum):
    LEGACY = "Legacy"
    V3 = "V3"


class SupportedDistributedIndices(str, Enum):
    # Scalar index types
    BTREE = "BTREE"
    INVERTED = "INVERTED"

    # Precise vector index types supported by distributed merge
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    IVF_SQ = "IVF_SQ"

    # Deprecated generic placeholder (kept for backward compatibility)
    VECTOR = "VECTOR"
