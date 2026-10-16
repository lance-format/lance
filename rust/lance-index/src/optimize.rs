// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use crate::progress::{IndexBuildProgress, noop_progress};

/// Options for optimizing all indices.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    /// Number of delta indices to merge for one column. Default: 1.
    ///
    /// If `num_indices_to_merge` is None, lance will create a new delta index if no partition is split, otherwise it will merge all delta indices.
    /// If `num_indices_to_merge` is Some(N), the delta updates and latest N indices
    /// will be merged into one single index.
    ///
    /// It is up to the caller to decide how many indices to merge / keep. Callers can
    /// find out how many indices are there by calling `Dataset::index_statistics`.
    ///
    /// A common usage pattern will be that, the caller can keep a large snapshot of the index of the base version,
    /// and accumulate a few delta indices, then merge them into the snapshot.
    pub num_indices_to_merge: Option<usize>,

    /// the index names to optimize. If None, all indices will be optimized.
    pub index_names: Option<Vec<String>>,

    /// whether to retrain the whole index. Default: false.
    ///
    /// If true, the index will be retrained based on the current data,
    /// `num_indices_to_merge` will be ignored, and all indices will be merged into one.
    /// If false, the index will be optimized by merging `num_indices_to_merge` indices.
    ///
    /// This is useful when the data distribution has changed significantly,
    /// and we want to retrain the index to improve the search quality.
    /// This would be faster than re-create the index from scratch.
    ///
    /// NOTE: this option is only supported for v3 vector indices.
    pub retrain: bool,

    /// Transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed index jobs).
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,

    /// Progress callback for index building during optimization.
    pub progress: Arc<dyn IndexBuildProgress>,

    /// Upper bound on the rows one build task covers. Default: unbounded.
    ///
    /// Unindexed fragments are packed into tasks of at most this many rows,
    /// and of at most `max_rows_per_segment`, since a task's output is never
    /// split (nor is a fragment); with neither set each index is one task.
    ///
    /// Setting either bound splits the build of every index that can merge
    /// segments, in `optimize_indices` as in the distributed plan; the
    /// `lance::index::optimize` module documents how that differs.
    pub max_rows_per_task: Option<u64>,

    /// Upper bound on the rows one committed segment holds. Default: unbounded.
    ///
    /// Existing segments and new tasks are packed into output segments of at
    /// most this many rows; `None` merges everything into one segment. See
    /// `max_rows_per_task` for what setting it changes.
    pub max_rows_per_segment: Option<u64>,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            num_indices_to_merge: None,
            index_names: None,
            retrain: false,
            transaction_properties: None,
            progress: noop_progress(),
            max_rows_per_task: None,
            max_rows_per_segment: None,
        }
    }
}

impl OptimizeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(num: usize) -> Self {
        Self {
            num_indices_to_merge: Some(num),
            index_names: None,
            ..Default::default()
        }
    }

    pub fn append() -> Self {
        Self {
            num_indices_to_merge: Some(0),
            index_names: None,
            ..Default::default()
        }
    }

    pub fn retrain() -> Self {
        Self {
            num_indices_to_merge: None,
            index_names: None,
            retrain: true,
            ..Default::default()
        }
    }

    pub fn num_indices_to_merge(mut self, num: Option<usize>) -> Self {
        self.num_indices_to_merge = num;
        self
    }

    pub fn index_names(mut self, names: Vec<String>) -> Self {
        self.index_names = Some(names);
        self
    }

    /// Set transaction properties to store in the commit manifest.
    pub fn transaction_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.transaction_properties = Some(Arc::new(properties));
        self
    }

    /// Set progress callback for index building during optimization.
    pub fn progress(mut self, progress: Arc<dyn IndexBuildProgress>) -> Self {
        self.progress = progress;
        self
    }

    /// Bound the rows one build task covers.
    pub fn max_rows_per_task(mut self, rows: Option<u64>) -> Self {
        self.max_rows_per_task = rows;
        self
    }

    /// Bound the rows one committed segment holds.
    pub fn max_rows_per_segment(mut self, rows: Option<u64>) -> Self {
        self.max_rows_per_segment = rows;
        self
    }
}
