# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
File format compatibility tests for Lance.

Tests that Lance files can be read and written across different versions,
covering various data types and file format versions.
"""

from pathlib import Path

import lance
import pytest
from lance.file import LanceFileReader, LanceFileWriter

from .compat_decorator import (
    UpgradeDowngradeTest,
    compat_test,
)
from .util import build_basic_types, build_large, safe_data_storage_version


@pytest.mark.parametrize(
    ("lance_version", "expected_storage_version"),
    [
        ("0.36.0", "2.0"),
        ("0.38.0", "2.1"),
        ("0.39.0", "2.1"),
        ("11.0.0", "2.1"),
    ],
)
def test_safe_data_storage_version(lance_version, expected_storage_version):
    assert safe_data_storage_version(lance_version) == expected_storage_version


# We start testing against the first release where 2.1 was stable. Before that
# the format was unstable so the readers will panic.
@compat_test(min_version="0.38.0")
class BasicTypes2_1(UpgradeDowngradeTest):
    """Test file format 2.1 compatibility with basic data types."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        batch = build_basic_types()
        with LanceFileWriter(
            str(self.path), version="2.1", schema=batch.schema
        ) as writer:
            writer.write_batch(batch)

    def check_read(self):
        reader = LanceFileReader(str(self.path))
        table = reader.read_all().to_table()
        assert table == build_basic_types()

    def check_write(self):
        # Test with overwrite
        with LanceFileWriter(str(self.path), version="2.1") as writer:
            writer.write_batch(build_basic_types())


# File format 2.2 is not in the stable 2.0.x line; gate this on the first
# available pre-release that includes 2.2 support.
@compat_test(min_version="4.0.0b1")
class BasicTypes2_2(UpgradeDowngradeTest):
    """Test file format 2.2 compatibility with basic data types."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        batch = build_basic_types()
        with LanceFileWriter(
            str(self.path), version="2.2", schema=batch.schema
        ) as writer:
            writer.write_batch(batch)

    def check_read(self):
        reader = LanceFileReader(str(self.path))
        table = reader.read_all().to_table()
        assert table == build_basic_types()

    def check_write(self):
        with LanceFileWriter(str(self.path), version="2.2") as writer:
            writer.write_batch(build_basic_types())


@compat_test(min_version="0.16.0")
@pytest.mark.parametrize(
    "data_factory,name",
    [
        (build_basic_types, "basic_types"),
        (build_large, "large"),
    ],
    ids=["basic_types", "large"],
)
class FileCompat(UpgradeDowngradeTest):
    """Test file format compatibility with different data types.

    Tests both basic types (scalars, strings, etc.) and large data (vectors, binary).
    """

    def __init__(self, path: Path, data_factory, name: str):
        self.path = path
        self.data_factory = data_factory
        self.name = name

    def create(self):
        """Create Lance file with test data."""
        batch = self.data_factory()
        with LanceFileWriter(
            str(self.path), version="2.0", schema=batch.schema
        ) as writer:
            writer.write_batch(batch)

    def check_read(self):
        """Verify file can be read and data matches."""
        reader = LanceFileReader(str(self.path))
        table = reader.read_all().to_table()
        expected = self.data_factory()
        assert table.equals(expected), f"Data mismatch for {self.name}"

    def check_write(self):
        """Verify can overwrite the file."""
        batch = self.data_factory()
        with LanceFileWriter(str(self.path), version="2.0") as writer:
            writer.write_batch(batch)


@compat_test(min_version="0.16.0")
class BasicTypesLegacy(UpgradeDowngradeTest):
    """Test legacy data storage version 0.1 compatibility."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        batch = build_basic_types()
        lance.write_dataset(batch, self.path, data_storage_version="0.1")

    def check_read(self):
        ds = lance.dataset(self.path)
        table = ds.to_table()
        assert table == build_basic_types()

    def check_write(self):
        ds = lance.dataset(self.path)
        ds.delete("true")
        lance.write_dataset(
            build_basic_types(), self.path, data_storage_version="0.1", mode="append"
        )
