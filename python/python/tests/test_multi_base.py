# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for multi-base dataset functionality.
"""

import shutil
import tempfile
import uuid
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pytest
from lance import DatasetBasePath
from lance.fragment import write_fragments


class TestMultiBase:
    """Test multi-base dataset functionality with local file system."""

    def setup_method(self):
        """Set up test directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_id = str(uuid.uuid4())[:8]

        # Create primary and additional path directories
        self.primary_uri = str(Path(self.test_dir) / "primary")
        self.path1_uri = str(Path(self.test_dir) / f"path1_{self.test_id}")
        self.path2_uri = str(Path(self.test_dir) / f"path2_{self.test_id}")
        self.path3_uri = str(Path(self.test_dir) / f"path3_{self.test_id}")

        # Create directories
        for uri in [self.primary_uri, self.path1_uri, self.path2_uri, self.path3_uri]:
            Path(uri).mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directories after each test."""
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_data(self, num_rows=500, id_offset=0):
        """Create test data for multi-base tests."""
        return pd.DataFrame(
            {
                "id": range(id_offset, id_offset + num_rows),
                "value": [f"value_{i}" for i in range(id_offset, id_offset + num_rows)],
                "score": [i * 0.1 for i in range(id_offset, id_offset + num_rows)],
            }
        )

    def test_multi_base_create_and_read(self):
        """Test creating a multi-base dataset and reading it back."""
        data = self.create_test_data(500)

        # Create dataset with multi-base layout
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                DatasetBasePath(self.path3_uri, name="path3"),
            ],
            target_bases=["path2"],  # Write data to path2
            max_rows_per_file=100,  # Force multiple fragments
        )

        assert dataset is not None
        assert dataset.uri == self.primary_uri

        # Verify we can read the data back
        result = dataset.to_table().to_pandas()
        assert len(result) == 500

        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            data.sort_values("id").reset_index(drop=True),
        )

    def test_write_target_all_bases(self):
        """target_all_bases=True rotates across primary and all initial bases."""
        data = self.create_test_data(300)

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_all_bases=True,
            max_rows_per_file=100,
        )

        base_ids = [f.data_files()[0].base_id for f in dataset.get_fragments()]
        assert base_ids == [None, 1, 2]
        assert len(dataset.to_table()) == 300

    def test_base_scoped_storage_options(self):
        """base_<id>.<key> storage options flow through write and read."""
        data = self.create_test_data(200)

        # Local stores ignore these options; this verifies base-scoped entries
        # are resolved per base without breaking the write or read path.
        storage_options = {
            "shared_option": "shared",
            "base_1.scoped_option": "base1-value",
        }

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.path1_uri, name="path1")],
            target_bases=["path1"],
            max_rows_per_file=100,
            storage_options=storage_options,
        )
        assert dataset.count_rows() == 200

        # Data files land in the scoped base, not the primary path
        assert list(Path(self.path1_uri).glob("**/*.lance"))
        assert not list((Path(self.primary_uri) / "data").glob("*.lance"))

        # Reopen with the same flat options and read through the base store
        dataset = lance.dataset(self.primary_uri, storage_options=storage_options)
        result = dataset.to_table().to_pandas()
        pd.testing.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            data.sort_values("id").reset_index(drop=True),
        )

    def test_multi_base_append_mode(self):
        """Test appending data to a multi-base dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(300)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=100,
        )

        # Create additional data to append
        append_data = self.create_test_data(100, id_offset=300)

        # Append to different path
        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_bases=["path2"],  # Write to path2
            max_rows_per_file=50,
        )

        # Verify total data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 400

        # Verify all data is present
        expected_ids = set(range(400))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

    def test_multi_base_overwrite_mode_inherits_bases(self):
        """Test OVERWRITE mode inherits existing base configuration."""
        # Create initial dataset with multi-base configuration
        initial_data = self.create_test_data(200)

        lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=100,
        )

        # Create new data for overwrite
        overwrite_data = self.create_test_data(150, id_offset=100)

        # Overwrite - should inherit existing bases (path1, path2)
        # Write to path2 (existing base, referenced by name)
        updated_dataset = lance.write_dataset(
            overwrite_data,
            self.primary_uri,
            mode="overwrite",
            target_bases=["path2"],  # Reference existing base by name
            max_rows_per_file=75,
        )

        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 150

        # Verify data content
        expected_ids = set(range(100, 250))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify base_paths are preserved from initial dataset
        base_paths = updated_dataset._ds.base_paths()
        assert len(base_paths) == 2
        assert any(bp.name == "path1" for bp in base_paths.values())
        assert any(bp.name == "path2" for bp in base_paths.values())

    def test_multi_base_overwrite_mode_primary_path_default(self):
        """Test OVERWRITE mode defaults to primary path when no target."""
        # Create initial dataset with explicit data file bases
        initial_data = self.create_test_data(100)

        lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=50,
        )

        # Overwrite without specifying target - should use primary path
        # This clears the old base_paths and writes to primary path only
        overwrite_data = self.create_test_data(75, id_offset=200)

        updated_dataset = lance.write_dataset(
            overwrite_data,
            self.primary_uri,
            mode="overwrite",
            # No target_bases specified - data goes to primary path
            # Old bases (path1, path2) are NOT preserved
            max_rows_per_file=25,
        )

        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 75

        # Verify data content
        expected_ids = set(range(200, 275))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify base_paths are preserved from previous manifest
        base_paths = updated_dataset._ds.base_paths()
        # Old path1 and path2 ARE preserved in manifest
        assert len(base_paths) == 2
        assert any(bp.name == "path1" for bp in base_paths.values())
        assert any(bp.name == "path2" for bp in base_paths.values())

        # Verify that data files were written to primary path (not to path1 or path2)
        fragments = list(updated_dataset.get_fragments())
        for fragment in fragments:
            # Fragment files should be in primary path (base_id should be None)
            data_file = fragment.data_files()[0]
            assert data_file.base_id is None, (
                f"Expected data file to be in primary path (base_id=None), "
                f"but got base_id={data_file.base_id}"
            )

        # Since base_paths exist, we can append to them by name
        append_data = self.create_test_data(25, id_offset=300)

        final_dataset = lance.write_dataset(
            append_data,
            updated_dataset,
            mode="append",
            target_bases=["path2"],  # Specify which base to use
        )

        final_result = final_dataset.to_table().to_pandas()
        assert len(final_result) == 100

    def test_multi_base_append_mode_primary_path_default(self):
        """Test that APPEND mode defaults to primary path when no target specified."""
        # Create initial dataset with explicit data file bases
        initial_data = self.create_test_data(100)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=50,
        )

        # Append without specifying target - should use primary path
        append_data = self.create_test_data(75, id_offset=100)

        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            # No target_bases specified - data goes to primary path
            max_rows_per_file=25,
        )

        # Verify appended data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 175

        # Verify data content
        expected_ids = set(range(175))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify base_paths are preserved from previous manifest
        base_paths = updated_dataset._ds.base_paths()
        assert len(base_paths) == 2
        assert any(bp.name == "path1" for bp in base_paths.values())
        assert any(bp.name == "path2" for bp in base_paths.values())

        # Verify new data files were written to primary path
        # (not to path1 or path2)
        fragments = list(updated_dataset.get_fragments())
        # The first 2 fragments should be in path1 (from initial write)
        # The remaining fragments should be in primary path (from append)
        primary_path_fragments = 0
        path1_fragments = 0

        for fragment in fragments:
            data_file = fragment.data_files()[0]
            if data_file.base_id is None:
                primary_path_fragments += 1
            else:
                base_path = base_paths.get(data_file.base_id)
                if base_path and base_path.name == "path1":
                    path1_fragments += 1

        assert path1_fragments == 2, (
            f"Expected 2 fragments in path1, got {path1_fragments}"
        )
        assert primary_path_fragments >= 3, (
            f"Expected at least 3 fragments in primary path, "
            f"got {primary_path_fragments}"
        )

    def test_multi_base_is_dataset_root_flag(self):
        """Test is_dataset_root flag on DatasetBasePath."""
        # Create initial dataset with one base marked as dataset_root
        initial_data = self.create_test_data(100)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1", is_dataset_root=True),
                DatasetBasePath(self.path2_uri, name="path2", is_dataset_root=False),
            ],
            target_bases=["path1"],
            max_rows_per_file=50,
        )

        # Verify base_paths configuration
        base_paths = dataset._ds.base_paths()
        assert len(base_paths) == 2

        # Find path1 and path2 in base_paths
        path1_base = None
        path2_base = None
        for base_path in base_paths.values():
            if base_path.name == "path1":
                path1_base = base_path
            elif base_path.name == "path2":
                path2_base = base_path

        assert path1_base is not None, "path1 base not found"
        assert path2_base is not None, "path2 base not found"

        # Verify is_dataset_root flags
        assert path1_base.is_dataset_root is True, (
            f"Expected path1.is_dataset_root=True, got {path1_base.is_dataset_root}"
        )
        assert path2_base.is_dataset_root is False, (
            f"Expected path2.is_dataset_root=False, got {path2_base.is_dataset_root}"
        )

        # Verify data is readable
        result = dataset.to_table().to_pandas()
        assert len(result) == 100

    def test_multi_base_target_by_path_uri(self):
        """Test using path URIs instead of names in target_bases."""
        # Create initial dataset with named bases
        initial_data = self.create_test_data(100)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1 using name
            max_rows_per_file=50,
        )

        # Get the base_paths to find the actual path URI for path2
        base_paths = dataset._ds.base_paths()
        path2_base = None
        for base_path in base_paths.values():
            if base_path.name == "path2":
                path2_base = base_path
                break

        assert path2_base is not None, "path2 base not found"

        # Append using the path URI instead of name
        append_data = self.create_test_data(50, id_offset=100)

        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_bases=[path2_base.path],  # Use path URI instead of name
            max_rows_per_file=25,
        )

        # Verify appended data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 150

        # Verify data content
        expected_ids = set(range(150))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify that new fragments are in path2 (not primary or path1)
        fragments = list(updated_dataset.get_fragments())
        base_paths_updated = updated_dataset._ds.base_paths()

        path1_fragments = 0
        path2_fragments = 0

        for fragment in fragments:
            data_file = fragment.data_files()[0]
            if data_file.base_id is not None:
                base_path = base_paths_updated.get(data_file.base_id)
                if base_path and base_path.name == "path1":
                    path1_fragments += 1
                elif base_path and base_path.name == "path2":
                    path2_fragments += 1

        assert path1_fragments == 2, (
            f"Expected 2 fragments in path1, got {path1_fragments}"
        )
        assert path2_fragments == 2, (
            f"Expected 2 fragments in path2, got {path2_fragments}"
        )

    def test_validation_errors(self):
        """Test validation errors for invalid multi-base configurations."""
        data = self.create_test_data(100)

        # Test 1: Target reference not found in new bases for CREATE mode
        with pytest.raises(Exception, match="not found"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                initial_bases=[
                    DatasetBasePath(self.path1_uri, name="path1"),
                    DatasetBasePath(self.path2_uri, name="path2"),
                ],
                target_bases=["path3"],  # Not in new bases
            )

        # Test 2: DatasetBasePath in APPEND mode
        # First create a base dataset
        base_dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],
        )

        with pytest.raises(Exception, match="Cannot register new bases in Append mode"):
            lance.write_dataset(
                data,
                base_dataset,
                mode="append",
                initial_bases=[
                    DatasetBasePath(
                        name="path3", path=self.path3_uri
                    ),  # New base not allowed
                ],
                target_bases=["path3"],
            )

    def test_fragment_distribution(self):
        """Test that fragments are correctly distributed and readable."""
        data = self.create_test_data(1000)

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],
            max_rows_per_file=100,  # Should create 10 fragments
        )

        # Verify fragment count
        fragments = list(dataset.get_fragments())
        assert len(fragments) >= 10  # At least 10 fragments

        # Verify all data is readable
        result = dataset.to_table().to_pandas()
        assert len(result) == 1000

        # Test scanning with filters
        filtered_result = dataset.scanner(filter="id < 500").to_table().to_pandas()
        assert len(filtered_result) == 500
        assert all(filtered_result["id"] < 500)


class TestAddBases:
    """Test the add_bases method for dynamically adding new base paths."""

    def setup_method(self):
        """Set up test directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_id = str(uuid.uuid4())[:8]

        # Create primary and additional path directories
        self.primary_uri = str(Path(self.test_dir) / "primary")
        self.initial_base = str(Path(self.test_dir) / f"initial_{self.test_id}")
        self.new_base1 = str(Path(self.test_dir) / f"new_base1_{self.test_id}")
        self.new_base2 = str(Path(self.test_dir) / f"new_base2_{self.test_id}")
        self.new_base3 = str(Path(self.test_dir) / f"new_base3_{self.test_id}")

        # Create directories
        for uri in [
            self.primary_uri,
            self.initial_base,
            self.new_base1,
            self.new_base2,
            self.new_base3,
        ]:
            Path(uri).mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directories after each test."""
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_bases_basic(self):
        """Test basic add_bases functionality - add one new base and write to it."""
        # Create initial dataset with one base
        initial_data = pd.DataFrame(
            {
                "id": range(50),
                "value": [f"initial_{i}" for i in range(50)],
                "source": ["initial"] * 50,
            }
        )

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
            max_rows_per_file=25,
        )

        assert len(dataset.to_table()) == 50

        # Add a new base using add_bases
        dataset = lance.dataset(self.primary_uri)
        result = dataset.add_bases([DatasetBasePath(self.new_base1, name="new_base1")])

        # Verify it returns self
        assert result is dataset

        # Write to the new base
        new_data = pd.DataFrame(
            {
                "id": range(50, 75),
                "value": [f"new_{i}" for i in range(50, 75)],
                "source": ["new_base1"] * 25,
            }
        )

        dataset = lance.write_dataset(
            new_data,
            dataset,
            mode="append",
            target_bases=["new_base1"],
            max_rows_per_file=25,
        )

        # Verify all data is present
        result = dataset.to_table().to_pandas()
        assert len(result) == 75
        assert len(result[result["source"] == "initial"]) == 50
        assert len(result[result["source"] == "new_base1"]) == 25

    def test_add_bases_multiple(self):
        """Test adding multiple bases at once."""
        # Create initial dataset
        initial_data = pd.DataFrame(
            {
                "id": range(30),
                "value": [f"val_{i}" for i in range(30)],
                "source": ["initial"] * 30,
            }
        )

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        # Add two bases at once
        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases(
            [
                DatasetBasePath(self.new_base1, name="new_base1"),
                DatasetBasePath(self.new_base2, name="new_base2"),
            ]
        )

        # Write to both new bases
        data1 = pd.DataFrame(
            {
                "id": range(30, 45),
                "value": [f"val_{i}" for i in range(30, 45)],
                "source": ["new_base1"] * 15,
            }
        )
        dataset = lance.write_dataset(
            data1, dataset, mode="append", target_bases=["new_base1"]
        )

        data2 = pd.DataFrame(
            {
                "id": range(45, 60),
                "value": [f"val_{i}" for i in range(45, 60)],
                "source": ["new_base2"] * 15,
            }
        )
        dataset = lance.write_dataset(
            data2, dataset, mode="append", target_bases=["new_base2"]
        )

        # Verify
        result = dataset.to_table().to_pandas()
        assert len(result) == 60
        assert len(result[result["source"] == "initial"]) == 30
        assert len(result[result["source"] == "new_base1"]) == 15
        assert len(result[result["source"] == "new_base2"]) == 15

    def test_add_bases_persistence(self):
        """Test that added bases persist across dataset reloads."""
        # Create dataset and add base
        initial_data = pd.DataFrame({"id": range(20), "value": range(20)})
        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases([DatasetBasePath(self.new_base1, name="new_base1")])

        # Write to new base
        new_data = pd.DataFrame({"id": range(20, 35), "value": range(20, 35)})
        dataset = lance.write_dataset(
            new_data, dataset, mode="append", target_bases=["new_base1"]
        )

        # Reload dataset from scratch
        del dataset
        reloaded = lance.dataset(self.primary_uri)

        # Should still be able to write to new_base1 without re-adding
        more_data = pd.DataFrame({"id": range(35, 45), "value": range(35, 45)})
        reloaded = lance.write_dataset(
            more_data, reloaded, mode="append", target_bases=["new_base1"]
        )

        # Verify
        result = reloaded.to_table().to_pandas()
        assert len(result) == 45
        assert set(result["id"]) == set(range(45))

    def test_add_bases_duplicate_name_error(self):
        """Test that adding a base with duplicate name raises an error."""
        # Create dataset
        data = pd.DataFrame({"id": range(10), "value": range(10)})
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        # Try to add another base with the same name
        dataset = lance.dataset(self.primary_uri)
        with pytest.raises(Exception) as excinfo:
            dataset.add_bases(
                [
                    DatasetBasePath(self.new_base1, name="initial_base")  # Duplicate!
                ]
            )

        # Should mention conflict or similar
        error_msg = str(excinfo.value).lower()
        assert (
            "conflict" in error_msg
            or "duplicate" in error_msg
            or "exists" in error_msg
            or "already" in error_msg
        )

    def test_add_bases_multiple_rounds(self):
        """Test adding bases in multiple rounds."""
        # Create initial dataset
        data = pd.DataFrame({"id": range(10), "round": ["initial"] * 10})
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        # Round 1: Add new_base1
        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases([DatasetBasePath(self.new_base1, name="new_base1")])

        data1 = pd.DataFrame({"id": range(10, 20), "round": ["round1"] * 10})
        dataset = lance.write_dataset(
            data1, dataset, mode="append", target_bases=["new_base1"]
        )

        # Round 2: Add new_base2
        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases([DatasetBasePath(self.new_base2, name="new_base2")])

        data2 = pd.DataFrame({"id": range(20, 30), "round": ["round2"] * 10})
        dataset = lance.write_dataset(
            data2, dataset, mode="append", target_bases=["new_base2"]
        )

        # Verify all rounds
        result = dataset.to_table().to_pandas()
        assert len(result) == 30
        assert len(result[result["round"] == "initial"]) == 10
        assert len(result[result["round"] == "round1"]) == 10
        assert len(result[result["round"] == "round2"]) == 10

    def test_add_bases_empty_list(self):
        """Test that adding an empty list of bases doesn't cause issues."""
        # Create dataset
        data = pd.DataFrame({"id": range(10), "value": range(10)})
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        # Add empty list - should not error
        dataset = lance.dataset(self.primary_uri)
        result = dataset.add_bases([])

        # Should return self
        assert result is dataset

        # Should still be able to use the dataset
        assert len(dataset.to_table()) == 10

    def test_add_bases_then_scan(self):
        """Test that scanner works correctly with dynamically added bases."""
        # Create dataset across two bases
        data1 = pd.DataFrame({"id": range(50), "value": range(50)})
        dataset = lance.write_dataset(
            data1,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases([DatasetBasePath(self.new_base1, name="new_base1")])

        data2 = pd.DataFrame({"id": range(50, 100), "value": range(50, 100)})
        dataset = lance.write_dataset(
            data2, dataset, mode="append", target_bases=["new_base1"]
        )

        # Test scanner with filter
        scanner = dataset.scanner(filter="id < 75")
        result = scanner.to_table().to_pandas()

        assert len(result) == 75
        assert all(result["id"] < 75)

        # Test limit
        scanner = dataset.scanner(limit=10)
        result = scanner.to_table().to_pandas()
        assert len(result) == 10

    def test_add_bases_verify_base_paths(self):
        """Test that get_base_paths returns added bases."""
        # Create dataset
        data = pd.DataFrame({"id": range(10), "value": range(10)})
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        # Add new bases
        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases(
            [
                DatasetBasePath(self.new_base1, name="new_base1"),
                DatasetBasePath(self.new_base2, name="new_base2"),
            ]
        )

        # Get base paths
        base_paths = dataset._ds.base_paths()

        # Should have 3 bases now (initial + 2 new)
        assert len(base_paths) == 3

        # Check that all bases are present
        names = [bp.name for bp in base_paths.values()]
        assert "initial_base" in names
        assert "new_base1" in names
        assert "new_base2" in names

    def test_add_bases_large_data_distribution(self):
        """Test adding bases and distributing large amounts of data."""
        # Create initial dataset with some data
        data = pd.DataFrame(
            {
                "id": range(100),
                "value": [f"val_{i}" for i in range(100)],
                "source": ["initial"] * 100,
            }
        )

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
            max_rows_per_file=50,
        )

        # Add multiple bases
        dataset = lance.dataset(self.primary_uri)
        dataset.add_bases(
            [
                DatasetBasePath(self.new_base1, name="new_base1"),
                DatasetBasePath(self.new_base2, name="new_base2"),
                DatasetBasePath(self.new_base3, name="new_base3"),
            ]
        )

        # Write large amounts of data to each new base
        data1 = pd.DataFrame(
            {
                "id": range(100, 300),
                "value": [f"val_{i}" for i in range(100, 300)],
                "source": ["new_base1"] * 200,
            }
        )
        dataset = lance.write_dataset(
            data1,
            dataset,
            mode="append",
            target_bases=["new_base1"],
            max_rows_per_file=50,
        )

        data2 = pd.DataFrame(
            {
                "id": range(300, 500),
                "value": [f"val_{i}" for i in range(300, 500)],
                "source": ["new_base2"] * 200,
            }
        )
        dataset = lance.write_dataset(
            data2,
            dataset,
            mode="append",
            target_bases=["new_base2"],
            max_rows_per_file=50,
        )

        data3 = pd.DataFrame(
            {
                "id": range(500, 700),
                "value": [f"val_{i}" for i in range(500, 700)],
                "source": ["new_base3"] * 200,
            }
        )
        dataset = lance.write_dataset(
            data3,
            dataset,
            mode="append",
            target_bases=["new_base3"],
            max_rows_per_file=50,
        )

        # Verify all data
        result = dataset.to_table().to_pandas()
        assert len(result) == 700

        # Verify distribution
        assert len(result[result["source"] == "initial"]) == 100
        assert len(result[result["source"] == "new_base1"]) == 200
        assert len(result[result["source"] == "new_base2"]) == 200
        assert len(result[result["source"] == "new_base3"]) == 200

        # Verify all IDs are present
        assert set(result["id"]) == set(range(700))

    def test_add_bases_with_transaction_properties(self):
        """Test that transaction properties are stored with add_bases operation."""
        from datetime import datetime

        # Create initial dataset
        data = pd.DataFrame({"id": range(20), "value": range(20)})
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[DatasetBasePath(self.initial_base, name="initial_base")],
            target_bases=["initial_base"],
        )

        # Add new base with transaction properties
        dataset = lance.dataset(self.primary_uri)
        properties = {
            "user": "alice@example.com",
            "timestamp": datetime.now().isoformat(),
            "purpose": "Adding new region storage for testing",
            "team": "data-platform",
            "__lance_commit_message": "Add new_base1 for improved performance",
        }

        dataset.add_bases(
            [DatasetBasePath(self.new_base1, name="new_base1")],
            transaction_properties=properties,
        )

        # Verify the base was added
        base_paths = dataset._ds.base_paths()
        names = [bp.name for bp in base_paths.values()]
        assert "new_base1" in names

        # Verify transaction properties are stored
        transaction = dataset.read_transaction(
            2
        )  # Version 2 is the add_bases operation
        props = transaction.transaction_properties

        assert props.get("user") == "alice@example.com"
        assert props.get("purpose") == "Adding new region storage for testing"
        assert props.get("team") == "data-platform"
        assert (
            props.get("__lance_commit_message")
            == "Add new_base1 for improved performance"
        )
        assert "timestamp" in props

        # Verify we can still write to the new base
        new_data = pd.DataFrame({"id": range(20, 30), "value": range(20, 30)})
        dataset = lance.write_dataset(
            new_data, dataset, mode="append", target_bases=["new_base1"]
        )

        # Verify all data
        result = dataset.to_table().to_pandas()
        assert len(result) == 30
        assert set(result["id"]) == set(range(30))


class TestWriteFragmentsWithTargetBases:
    """Test write_fragments with target_bases parameter."""

    def setup_method(self):
        """Set up test directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_id = str(uuid.uuid4())[:8]

        # Create primary and additional path directories
        self.primary_uri = str(Path(self.test_dir) / "primary")
        self.base1_uri = str(Path(self.test_dir) / f"base1_{self.test_id}")
        self.base2_uri = str(Path(self.test_dir) / f"base2_{self.test_id}")

        # Create directories
        for uri in [self.primary_uri, self.base1_uri, self.base2_uri]:
            Path(uri).mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directories after each test."""
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_write_fragments_with_target_bases(self):
        """Test write_fragments with target_bases parameter."""
        # Create initial dataset with multiple bases
        initial_data = pd.DataFrame(
            {
                "id": range(50),
                "value": [f"initial_{i}" for i in range(50)],
            }
        )

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.base1_uri, name="base1"),
                DatasetBasePath(self.base2_uri, name="base2"),
            ],
            target_bases=["base1"],
            max_rows_per_file=25,
        )

        # Verify initial data is written
        assert len(dataset.to_table()) == 50

        # Write fragments using write_fragments with target_bases
        fragment_data = pd.DataFrame(
            {
                "id": range(50, 75),
                "value": [f"fragment_{i}" for i in range(50, 75)],
            }
        )

        # Use write_fragments with target_bases set to base2
        fragments = write_fragments(
            pa.Table.from_pandas(fragment_data),
            dataset,
            mode="append",
            target_bases=["base2"],
            base_store_params={self.base2_uri: {}},
            max_rows_per_file=25,
        )

        # Fragments should be created
        assert len(fragments) > 0

        # Commit the fragments using dataset.commit
        operation = lance.LanceOperation.Append(fragments)
        dataset = lance.LanceDataset.commit(
            dataset.uri, operation, read_version=dataset.version
        )

        # Verify all data is present
        result = dataset.to_table().to_pandas()
        assert len(result) == 75
        assert set(result["id"]) == set(range(75))

        # Verify fragments are in the correct base
        # Check that some fragments exist in base2
        base2_path = Path(self.base2_uri)
        data_files = list(base2_path.glob("**/*.lance"))
        assert len(data_files) > 0, "Expected data files in base2"

    def test_write_fragments_with_base_store_params(self):
        """Test write_fragments inherits base_store_params from LanceDataset."""
        initial_data = pd.DataFrame({"id": range(10), "value": range(10)})

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.base1_uri, name="base1"),
                DatasetBasePath(self.base2_uri, name="base2"),
            ],
            target_bases=["base1"],
            base_store_params={self.base2_uri: {}},
            max_rows_per_file=10,
        )

        fragment_data = pd.DataFrame({"id": range(10, 20), "value": range(10, 20)})
        fragments = write_fragments(
            pa.Table.from_pandas(fragment_data),
            dataset,
            mode="append",
            target_bases=["base2"],
            max_rows_per_file=10,
        )

        operation = lance.LanceOperation.Append(fragments)
        dataset = lance.LanceDataset.commit(
            dataset, operation, read_version=dataset.version
        )

        result = dataset.to_table().to_pandas()
        assert len(result) == 20
        assert set(result["id"]) == set(range(20))
        assert list(Path(self.base2_uri).glob("**/*.lance"))

    def test_write_fragments_transaction_with_target_bases(self):
        """Test write_fragments with return_transaction and target_bases."""
        # Create initial dataset
        initial_data = pd.DataFrame({"id": range(30), "value": range(30)})

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.base1_uri, name="base1"),
                DatasetBasePath(self.base2_uri, name="base2"),
            ],
            target_bases=["base1"],
            max_rows_per_file=15,
        )

        # Use write_fragments with return_transaction=True and target_bases
        new_data = pd.DataFrame({"id": range(30, 50), "value": range(30, 50)})

        transaction = write_fragments(
            pa.Table.from_pandas(new_data),
            dataset,
            mode="append",
            return_transaction=True,
            target_bases=["base2"],
            max_rows_per_file=10,
        )

        # Commit the transaction
        dataset = lance.LanceDataset.commit(
            dataset.uri, transaction, read_version=dataset.version
        )

        # Verify data
        result = dataset.to_table().to_pandas()
        assert len(result) == 50
        assert set(result["id"]) == set(range(50))

    def test_write_fragments_overwrite_mode_with_target_bases(self):
        """Test write_fragments in OVERWRITE mode with target_bases."""
        # Create initial dataset
        initial_data = pd.DataFrame(
            {
                "id": range(30),
                "value": [f"initial_{i}" for i in range(30)],
            }
        )

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.base1_uri, name="base1"),
                DatasetBasePath(self.base2_uri, name="base2"),
            ],
            target_bases=["base1"],
            max_rows_per_file=15,
        )

        assert len(dataset.to_table()) == 30

        # Use write_fragments with mode="overwrite" to replace all data
        overwrite_data = pd.DataFrame(
            {
                "id": range(100, 120),
                "value": [f"overwrite_{i}" for i in range(100, 120)],
            }
        )

        fragments = write_fragments(
            pa.Table.from_pandas(overwrite_data),
            dataset,
            mode="overwrite",
            target_bases=["base2"],  # Write to base2 this time
            max_rows_per_file=10,
        )

        assert len(fragments) > 0

        # Commit with Overwrite operation
        operation = lance.LanceOperation.Overwrite(
            pa.Table.from_pandas(overwrite_data).schema, fragments
        )
        dataset = lance.LanceDataset.commit(
            dataset.uri, operation, read_version=dataset.version
        )

        # Verify data was overwritten (only new data should exist)
        result = dataset.to_table().to_pandas()
        assert len(result) == 20
        assert set(result["id"]) == set(range(100, 120))
        # Old data (0-29) should be gone
        assert not any(result["id"] < 100)

        # Verify fragments are in base2
        base2_path = Path(self.base2_uri)
        data_files = list(base2_path.glob("**/*.lance"))
        assert len(data_files) > 0, "Expected data files in base2"

    def test_write_fragments_create_mode_with_initial_bases(self):
        """Test write_fragments in CREATE mode with initial_bases."""
        # Create a new dataset URI (doesn't exist yet)
        dataset_uri = Path(self.test_dir) / "new_dataset_with_commit"

        # Create base paths
        base1_path = Path(self.test_dir) / "base1_new"
        base2_path = Path(self.test_dir) / "base2_new"
        base1_path.mkdir(parents=True, exist_ok=True)
        base2_path.mkdir(parents=True, exist_ok=True)

        # Define initial bases to register using DatasetBasePath objects
        initial_bases = [
            lance.DatasetBasePath(path=str(base1_path), name="base1"),
            lance.DatasetBasePath(path=str(base2_path), name="base2"),
        ]

        # Write fragments in CREATE mode with both initial_bases and target_bases
        # Use return_transaction=True so that the Rust code properly assigns
        # IDs to initial_bases
        data = pa.table({"id": range(20), "value": [f"val_{i}" for i in range(20)]})
        transaction = write_fragments(
            data,
            str(dataset_uri),
            mode="create",
            target_bases=["base1"],
            initial_bases=initial_bases,
            return_transaction=True,
        )

        # Commit the transaction (initial_bases with proper IDs are already in
        # the transaction)
        dataset = lance.LanceDataset.commit(str(dataset_uri), transaction)

        # Verify dataset was created
        assert dataset.count_rows() == 20
        result = dataset.to_table().to_pandas()
        assert len(result) == 20
        assert set(result["id"]) == set(range(20))

        # Verify base paths are registered
        base_paths = dataset._ds.base_paths()
        assert len(base_paths) == 2  # 2 bases (base1, base2)
        # Check that our named bases are registered
        base_names = [bp.name for bp in base_paths.values() if bp.name is not None]
        assert "base1" in base_names
        assert "base2" in base_names

        # Verify data files are in base1 (not in dataset root)
        data_files_base1 = list(base1_path.glob("**/*.lance"))
        assert len(data_files_base1) > 0, "Expected data files in base1"

        # Dataset root should not have data files (only manifest)
        dataset_root = Path(dataset_uri)
        data_files_root = list(dataset_root.glob("*.lance"))
        assert len(data_files_root) == 0, "Should not have data files in root"


class TestDataReplacementWithBases:
    """DataReplacement must preserve DataFile.base_id so replacement files can
    live in (and resolve against) a storage base other than the dataset root."""

    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.primary_uri = str(Path(self.test_dir) / "primary")
        self.base1_uri = str(Path(self.test_dir) / "base1")
        Path(self.base1_uri).mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_two_base_dataset(self) -> "lance.LanceDataset":
        """Fragment 0 at the dataset root, fragment 1 in base1; column a."""
        ds = lance.write_dataset(
            pa.table({"a": list(range(8))}),
            self.primary_uri,
            max_rows_per_file=8,
        )
        ds.add_bases([DatasetBasePath(self.base1_uri, name="b1", is_dataset_root=True)])
        return lance.write_dataset(
            pa.table({"a": list(range(8, 16))}),
            self.primary_uri,
            mode="append",
            max_rows_per_file=8,
            target_bases=["b1"],
        )

    def _write_bare_file(self, data_dir: str, data: pa.Table) -> str:
        from lance.file import LanceFileWriter

        name = f"{uuid.uuid4()}.lance"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with LanceFileWriter(f"{data_dir}/{name}") as writer:
            writer.write_batch(data)
        return name

    def _b_data_file(self, name: str, base_id=None) -> "lance.fragment.DataFile":
        from lance.file import stable_version
        from lance.fragment import DataFile

        return DataFile(
            path=name,
            fields=[1],  # field id of column "b" (a=0, b=1)
            column_indices=[0],
            file_major_version=int(stable_version().split(".")[0]),
            file_minor_version=int(stable_version().split(".")[1]),
            base_id=base_id,
        )

    def test_data_replacement_new_column_into_base(self):
        """The all-NULL-column special case: the new column's data file for a
        base fragment is written into that base and must keep its base_id."""
        ds = self._make_two_base_dataset()
        ds.add_columns(pa.field("b", pa.int32()))
        ds = lance.dataset(self.primary_uri)

        root_name = self._write_bare_file(
            f"{self.primary_uri}/data",
            pa.table({"b": pa.array([x * 10 for x in range(8)], pa.int32())}),
        )
        base_name = self._write_bare_file(
            f"{self.base1_uri}/data",
            pa.table({"b": pa.array([x * 10 for x in range(8, 16)], pa.int32())}),
        )

        op = lance.LanceOperation.DataReplacement(
            [
                lance.LanceOperation.DataReplacementGroup(
                    0, self._b_data_file(root_name)
                ),
                lance.LanceOperation.DataReplacementGroup(
                    1, self._b_data_file(base_name, base_id=1)
                ),
            ]
        )
        ds = lance.LanceDataset.commit(self.primary_uri, op, read_version=ds.version)

        frags = ds.get_fragments()
        root_files = {f.path: f.base_id for f in frags[0].data_files()}
        base_files = {f.path: f.base_id for f in frags[1].data_files()}
        assert root_files[root_name] is None
        assert base_files[base_name] == 1

        table = ds.to_table()
        assert table.column("b").to_pylist() == [x * 10 for x in range(16)]

    def test_data_replacement_replace_existing_file_in_base(self):
        """The replace-existing-file branch must take the new file's base_id."""
        ds = self._make_two_base_dataset()
        ds.add_columns(pa.field("b", pa.int32()))
        ds = lance.dataset(self.primary_uri)

        first = self._write_bare_file(
            f"{self.base1_uri}/data",
            pa.table({"b": pa.array([0] * 8, pa.int32())}),
        )
        op = lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(1, self._b_data_file(first, 1))]
        )
        ds = lance.LanceDataset.commit(self.primary_uri, op, read_version=ds.version)

        # Replace the same column file again, still in base1.
        second = self._write_bare_file(
            f"{self.base1_uri}/data",
            pa.table({"b": pa.array([x * 10 for x in range(8, 16)], pa.int32())}),
        )
        op = lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(1, self._b_data_file(second, 1))]
        )
        ds = lance.LanceDataset.commit(self.primary_uri, op, read_version=ds.version)

        base_files = {f.path: f.base_id for f in ds.get_fragments()[1].data_files()}
        assert base_files[second] == 1
        assert first not in base_files
        assert ds.to_table().column("b").to_pylist()[8:] == [
            x * 10 for x in range(8, 16)
        ]


class TestMergeInsertMultiBase:
    """Test merge insert on multi-base datasets."""

    def setup_method(self):
        """Set up test directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.primary_uri = str(Path(self.test_dir) / "primary")
        self.base1_uri = str(Path(self.test_dir) / "base1")
        self.base2_uri = str(Path(self.test_dir) / "base2")
        for uri in [self.primary_uri, self.base1_uri, self.base2_uri]:
            Path(uri).mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directories after each test."""
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_dataset(self):
        """Dataset with two registered bases and initial data in base1."""
        initial_data = pd.DataFrame(
            {
                "id": range(100),
                "value": [f"initial_{i}" for i in range(100)],
            }
        )
        return lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.base1_uri, name="base1"),
                DatasetBasePath(self.base2_uri, name="base2"),
            ],
            target_bases=["base1"],
            max_rows_per_file=50,
        )

    def base_name_of(self, dataset, data_file):
        base_paths = dataset._ds.base_paths()
        if data_file.base_id is None:
            return None
        return base_paths[data_file.base_id].name

    def test_merge_insert_without_target_bases(self):
        """Merge insert on a multi-base dataset writes to primary by default."""
        dataset = self.create_dataset()

        new_data = pd.DataFrame(
            {
                "id": range(50, 150),
                "value": [f"updated_{i}" for i in range(50, 150)],
            }
        )
        stats = (
            dataset.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(new_data)
        )
        assert stats["num_updated_rows"] == 50
        assert stats["num_inserted_rows"] == 50

        result = dataset.to_table().to_pandas().sort_values("id")
        assert len(result) == 150
        assert list(result[result["id"] >= 50]["value"]) == [
            f"updated_{i}" for i in range(50, 150)
        ]

        # New fragments (merge output) are in primary storage.
        max_initial_fragment_id = 1  # 100 rows / 50 per file -> fragments 0, 1
        for fragment in dataset.get_fragments():
            is_initial = fragment.fragment_id <= max_initial_fragment_id
            for data_file in fragment.data_files():
                expected = "base1" if is_initial else None
                assert self.base_name_of(dataset, data_file) == expected

    def test_merge_insert_with_target_bases(self):
        """Merge insert routes new fragments to the requested base."""
        dataset = self.create_dataset()

        new_data = pd.DataFrame(
            {
                "id": range(50, 150),
                "value": [f"updated_{i}" for i in range(50, 150)],
            }
        )
        stats = (
            dataset.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .target_bases(["base2"])
            .execute(new_data)
        )
        assert stats["num_updated_rows"] == 50
        assert stats["num_inserted_rows"] == 50

        result = dataset.to_table().to_pandas().sort_values("id")
        assert len(result) == 150
        assert list(result[result["id"] >= 50]["value"]) == [
            f"updated_{i}" for i in range(50, 150)
        ]

        merge_files = 0
        for fragment in dataset.get_fragments():
            if fragment.fragment_id > 1:
                for data_file in fragment.data_files():
                    assert self.base_name_of(dataset, data_file) == "base2"
                    merge_files += 1
        assert merge_files > 0
        assert list(Path(self.base2_uri).glob("**/*.lance"))

        # The routed dataset stays readable from a fresh instance.
        reloaded = lance.dataset(self.primary_uri)
        assert reloaded.count_rows() == 150

    def test_merge_insert_with_unknown_target_base(self):
        """Merge insert referencing an unregistered base fails."""
        dataset = self.create_dataset()

        new_data = pd.DataFrame({"id": [1], "value": ["updated_1"]})
        with pytest.raises(Exception, match="not found in available bases"):
            (
                dataset.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .target_bases(["nonexistent"])
                .execute(new_data)
            )

    def test_merge_insert_target_primary_via_uri(self):
        """The dataset URI in target_bases selects primary storage."""
        dataset = self.create_dataset()

        new_data = pd.DataFrame({"id": [200], "value": ["inserted_200"]})
        (
            dataset.merge_insert("id")
            .when_not_matched_insert_all()
            .target_bases([dataset.uri, "base2"])
            .execute(new_data)
        )
        assert dataset.count_rows() == 101

        # The single new file lands in the first slot: primary storage.
        for fragment in dataset.get_fragments():
            if fragment.fragment_id > 1:
                for data_file in fragment.data_files():
                    assert self.base_name_of(dataset, data_file) is None

    def test_merge_insert_target_all_bases(self):
        """target_all_bases spreads new files across all bases, primary first."""
        dataset = self.create_dataset()

        new_data = pd.DataFrame({"id": [300], "value": ["inserted_300"]})
        (
            dataset.merge_insert("id")
            .when_not_matched_insert_all()
            .target_all_bases()
            .execute(new_data)
        )
        assert dataset.count_rows() == 101
        # A single new file lands in the first slot: primary storage.
        newest = max(f.fragment_id for f in dataset.get_fragments())
        for fragment in dataset.get_fragments():
            if fragment.fragment_id == newest:
                for data_file in fragment.data_files():
                    assert self.base_name_of(dataset, data_file) is None

        new_data = pd.DataFrame({"id": [301], "value": ["inserted_301"]})
        (
            dataset.merge_insert("id")
            .when_not_matched_insert_all()
            .target_all_bases(include_primary=False)
            .execute(new_data)
        )
        assert dataset.count_rows() == 102
        newest = max(f.fragment_id for f in dataset.get_fragments())
        for fragment in dataset.get_fragments():
            if fragment.fragment_id == newest:
                for data_file in fragment.data_files():
                    assert self.base_name_of(dataset, data_file) == "base1"
