// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use pyo3::{pyclass, pymethods};

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupStats {
    pub bytes_removed: u64,
    pub old_versions: u64,
    pub data_files_removed: u64,
    pub transaction_files_removed: u64,
    pub index_files_removed: u64,
    pub deletion_files_removed: u64,
    /// Files that could not be deleted; not counted as removed.
    pub failed_deletes: u64,
}

#[pymethods]
impl CleanupStats {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// Result of expiring versions. Manifests only; no data files are touched.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ExpireVersionsStats {
    pub versions_removed: u64,
    pub versions_retained: u64,
    pub bytes_removed: u64,
    /// Manifests that could not be deleted; not counted as removed.
    pub failed_deletes: u64,
}

#[pymethods]
impl ExpireVersionsStats {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// What `expire_versions` would remove, without removing it.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ExpireVersionsPlan {
    pub versions: Vec<u64>,
    pub stats: ExpireVersionsStats,
    /// Tagged versions the policy would have expired but will keep.
    pub tagged_but_kept: Vec<u64>,
}

#[pymethods]
impl ExpireVersionsPlan {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupCandidateFile {
    pub path: String,
    pub kind: String,
    pub unverified: bool,
    pub size_bytes: u64,
}

#[pymethods]
impl CleanupCandidateFile {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupReferencedBranch {
    pub name: String,
    pub referenced_version: u64,
    pub cleanup_candidate: bool,
}

#[pymethods]
impl CleanupReferencedBranch {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupExplanation {
    pub read_version: u64,
    pub stats: CleanupStats,
    pub candidate_files: Vec<CleanupCandidateFile>,
    pub candidate_files_truncated: bool,
    pub candidate_file_limit: usize,
    pub referenced_branches: Vec<CleanupReferencedBranch>,
    pub warnings: Vec<String>,
}

#[pymethods]
impl CleanupExplanation {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}
