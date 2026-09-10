// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::transaction::Transaction;
use crate::Dataset;
use crate::Result;
use crate::dataset::fragment::FileFragment;
use crate::dataset::rowids::load_row_id_sequence;
use crate::dataset::scanner::{
    BATCH_SIZE_FALLBACK, DatasetRecordBatchStream, get_default_batch_size,
};
use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::Schema as ArrowSchema;
use arrow_schema::SortOptions;
use chrono::{DateTime, Utc};
use datafusion::common::NullEquality;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::JoinType;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::joins::SortMergeJoinExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_expr::expressions::Column;
use futures::Stream;
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::Error;
use lance_core::ROW_CREATED_AT_VERSION;
use lance_core::ROW_ID;
use lance_core::ROW_ID_FIELD;
use lance_core::ROW_LAST_UPDATED_AT_VERSION;
use lance_core::WILDCARD;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::exec::{LanceExecutionOptions, OneShotExec, execute_plan};
use lance_table::format::Fragment;
use lance_table::rowids::RowIdSequence;
use lance_table::rowids::segment::U64Segment;
use std::collections::HashMap;
use std::sync::Arc;

/// Rows per batch of [`DatasetDelta::get_deleted_row_ids`], taken from the
/// scanner so it matches the sibling readers.
fn deleted_row_id_batch_rows() -> usize {
    batch_rows(get_default_batch_size())
}

/// The largest batch this reader will emit whatever the configuration says:
/// the batch is the unit of buffering, so an unbounded setting would defeat
/// the chunking.
const DELETED_ROW_ID_BATCH_CAP: usize = 64 * 1024;

/// A configured size of zero would mean no bound at all, so it is refused in
/// favour of the default; an oversized one is clamped to the cap.
fn batch_rows(configured: Option<usize>) -> usize {
    configured
        .filter(|rows| *rows > 0)
        .unwrap_or(BATCH_SIZE_FALLBACK)
        .min(DELETED_ROW_ID_BATCH_CAP)
}

/// Builder for creating a [`DatasetDelta`] to explore changes between dataset versions.
///
/// # Example
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::delta::DatasetDeltaBuilder;
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// // Compare against a specific version
/// let delta = DatasetDeltaBuilder::new(dataset.clone())
///     .compared_against_version(5)
///     .build()?;
///
/// // Or specify explicit version range
/// let delta = DatasetDeltaBuilder::new(dataset.clone())
///     .with_begin_version(3)
///     .with_end_version(7)
///     .build()?;
///
/// // Or specify explicit time range
/// let delta = DatasetDeltaBuilder::new(dataset.clone())
///     .with_begin_date(chrono::Utc::now())
///     .with_end_date(chrono::Utc::now())
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct DatasetDeltaBuilder {
    dataset: Dataset,
    compared_against_version: Option<u64>,
    begin_version: Option<u64>,
    end_version: Option<u64>,
    begin_timestamp: Option<DateTime<Utc>>,
    end_timestamp: Option<DateTime<Utc>>,
}

impl DatasetDeltaBuilder {
    /// Create a new builder for the given dataset.
    pub fn new(dataset: Dataset) -> Self {
        Self {
            dataset,
            compared_against_version: None,
            begin_version: None,
            end_version: None,
            begin_timestamp: None,
            end_timestamp: None,
        }
    }

    /// Compare the current dataset version against the specified version.
    ///
    /// The delta will automatically order the versions so that `begin_version` < `end_version`.
    /// Cannot be used together with explicit `with_begin_version` and `with_end_version`.
    pub fn compared_against_version(mut self, version: u64) -> Self {
        self.compared_against_version = Some(version);
        self
    }

    /// Set the beginning version for the delta (exclusive).
    ///
    /// Must be used together with `with_end_version`.
    /// Cannot be used together with `compared_against_version`.
    pub fn with_begin_version(mut self, version: u64) -> Self {
        self.begin_version = Some(version);
        self
    }

    /// Set the ending version for the delta (inclusive).
    ///
    /// Must be used together with `with_begin_version`.
    /// Cannot be used together with `compared_against_version`.
    pub fn with_end_version(mut self, version: u64) -> Self {
        self.end_version = Some(version);
        self
    }

    /// Set the beginning timestamp for the delta (exclusive).
    ///
    /// Must be used together with `with_end_date`.
    /// Cannot be used together with `compared_against_version` or explicit version range.
    pub fn with_begin_date(mut self, timestamp: DateTime<Utc>) -> Self {
        self.begin_timestamp = Some(timestamp);
        self
    }

    /// Set the ending timestamp for the delta (inclusive).
    ///
    /// Must be used together with `with_begin_date`.
    /// Cannot be used together with `compared_against_version` or explicit version range.
    pub fn with_end_date(mut self, timestamp: DateTime<Utc>) -> Self {
        self.end_timestamp = Some(timestamp);
        self
    }

    /// Build the [`DatasetDelta`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Both `compared_against_version` and explicit version range are specified
    /// - Neither `compared_against_version` nor explicit version range are specified
    /// - Only one of `with_begin_version` or `with_end_version` is specified
    pub fn build(self) -> Result<DatasetDelta> {
        // Validate incompatible combinations
        if self.compared_against_version.is_some()
            && (self.begin_version.is_some()
                || self.end_version.is_some()
                || self.begin_timestamp.is_some()
                || self.end_timestamp.is_some())
        {
            return Err(Error::invalid_input(
                "Cannot combine compared_against_version with explicit begin/end versions or dates",
            ));
        }

        // Resolve parameters and construct DatasetDelta. For date ranges, defer mapping to versions.
        let (begin_version, end_version, begin_ts, end_ts) = match (
            self.compared_against_version,
            self.begin_version,
            self.end_version,
            self.begin_timestamp,
            self.end_timestamp,
        ) {
            (Some(compared), None, None, None, None) => {
                let current_version = self.dataset.version().version;
                if current_version > compared {
                    (compared, current_version, None, None)
                } else {
                    (current_version, compared, None, None)
                }
            }
            (None, Some(begin), Some(end), None, None) => (begin, end, None, None),
            (None, None, None, Some(begin_ts), Some(end_ts)) => {
                (0, 0, Some(begin_ts), Some(end_ts))
            }
            (None, Some(_), None, None, None) | (None, None, Some(_), None, None) => {
                return Err(Error::invalid_input(
                    "Must specify both with_begin_version and with_end_version",
                ));
            }
            (None, None, None, Some(begin_ts), None) => (0, 0, Some(begin_ts), None),
            (None, None, None, None, Some(_)) => {
                return Err(Error::invalid_input(
                    "Must specify with_begin_date when with_end_date is provided",
                ));
            }
            (None, None, None, None, None) => {
                return Err(Error::invalid_input(
                    "Must specify either compared_against_version or both with_begin_version and with_end_version",
                ));
            }
            _ => {
                return Err(Error::invalid_input(
                    "Invalid combination of parameters for DatasetDeltaBuilder",
                ));
            }
        };

        Ok(DatasetDelta {
            begin_version,
            end_version,
            base_dataset: self.dataset,
            begin_timestamp: begin_ts,
            end_timestamp: end_ts,
        })
    }
}

/// APIs for exploring changes between two versions of a dataset.
pub struct DatasetDelta {
    /// The base version number for comparison.
    pub(crate) begin_version: u64,
    /// The end version number for comparison
    pub(crate) end_version: u64,
    /// The Lance dataset to compute delta
    pub(crate) base_dataset: Dataset,
    pub(crate) begin_timestamp: Option<DateTime<Utc>>,
    pub(crate) end_timestamp: Option<DateTime<Utc>>,
}

impl DatasetDelta {
    /// Resolve the effective version range for this delta.
    ///
    /// If a date window is set (`begin_timestamp` and `end_timestamp` provided), this lazily
    /// maps timestamps to version ids by scanning dataset versions:
    /// - Begin is exclusive: pick the greatest version with `timestamp < begin_timestamp`.
    /// - End is inclusive:  pick the greatest version with `timestamp <= end_timestamp`.
    ///
    /// If no date window is set, returns the explicit `begin_version`/`end_version` stored on
    /// the struct.
    async fn resolve_range(&self) -> Result<(u64, u64)> {
        if let (Some(begin_ts), Some(end_ts)) = (self.begin_timestamp, self.end_timestamp) {
            // Load all dataset versions and fold them to a version interval matching the date window
            let versions = self.base_dataset.versions().await?;
            let mut begin_version: u64 = 0;
            let mut end_version: u64 = 0;
            for v in &versions {
                // Exclusive begin: track the largest version strictly before begin_ts
                if v.timestamp < begin_ts && v.version > begin_version {
                    begin_version = v.version;
                }
                // Inclusive end: track the largest version at or before end_ts
                if v.timestamp <= end_ts && v.version > end_version {
                    end_version = v.version;
                }
            }
            Ok((begin_version, end_version))
        } else if let (Some(begin_ts), None) = (self.begin_timestamp, self.end_timestamp) {
            // Open-ended range: use latest version as end
            let versions = self.base_dataset.versions().await?;
            let mut begin_version: u64 = 0;
            for v in &versions {
                if v.timestamp < begin_ts && v.version > begin_version {
                    begin_version = v.version;
                }
            }
            let end_version = self.base_dataset.latest_version_id().await?;
            Ok((begin_version, end_version))
        } else {
            // No date window: use the pre-resolved version interval
            Ok((self.begin_version, self.end_version))
        }
    }

    /// Listing the transactions between two versions.
    pub async fn list_transactions(&self) -> Result<Vec<Transaction>> {
        let (begin_version, end_version) = self.resolve_range().await?;
        stream::iter((begin_version + 1)..=end_version)
            .map(|version| {
                let base_dataset = self.base_dataset.clone();
                async move {
                    let current_ds = match base_dataset.checkout_version(version).await {
                        Ok(ds) => ds,
                        Err(err) => {
                            if matches!(err, Error::DatasetNotFound { .. }) {
                                return Err(Error::VersionNotFound {
                                    message: format!(
                                        "Can not find version {}, please check if it has been cleanup.",
                                        version
                                    ),
                                });
                            } else {
                                return Err(err);
                            }
                        }
                    };
                    current_ds.read_transaction().await
                }
            })
            .buffered(get_num_compute_intensive_cpus())
            .try_filter_map(|result| async move { Ok(result) })
            .try_collect()
            .await
    }

    /// The stable row ids live at the begin version and absent at the end
    /// version, as a stream of batches carrying a single [`ROW_ID`] column.
    /// Rows in a fragment the range removed outright count as deleted.
    ///
    /// Requires stable row ids at both endpoints and an ordered range;
    /// version 0 is the empty snapshot. Runs in bounded memory: subtracting
    /// the still-live ids is a sort-merge anti join that spills past the
    /// session memory pool.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use futures::TryStreamExt;
    /// # async fn example(dataset: &Dataset, previous_version: u64) -> Result<()> {
    /// let delta = dataset
    ///     .delta()
    ///     .compared_against_version(previous_version)
    ///     .build()?;
    /// let mut deleted = delta.get_deleted_row_ids().await?;
    /// while let Some(batch) = deleted.try_next().await? {
    ///     // Each batch holds a `_rowid` column of deleted ids.
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_deleted_row_ids(&self) -> Result<DatasetRecordBatchStream> {
        let (begin_version, end_version) = self.resolve_range().await?;
        if begin_version > end_version {
            // A reversed range would report the rows the range added as
            // deleted.
            return Err(Error::invalid_input(format!(
                "begin version {begin_version} is newer than end version {end_version}"
            )));
        }
        let schema = Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()]));
        // Version 0 is the empty snapshot: nothing is live at it, so nothing
        // is deleted relative to it.
        if begin_version == 0 {
            return Ok(DatasetRecordBatchStream::new(Box::pin(
                RecordBatchStreamAdapter::new(schema, stream::empty()),
            )));
        }
        let begin = Arc::new(self.base_dataset.checkout_version(begin_version).await?);
        let end = Arc::new(self.base_dataset.checkout_version(end_version).await?);
        // Both endpoints: a restore can leave later versions without them.
        for endpoint in [&begin, &end] {
            if !endpoint.manifest.uses_stable_row_ids() {
                return Err(Error::invalid_input(format!(
                    "deleted row ids require stable row ids, version {} does not use them",
                    endpoint.manifest.version
                )));
            }
        }

        let begin_frags = begin.get_fragments();
        let end_frags = end.get_fragments();
        let delta = fragment_delta(
            begin_frags.iter().map(|f| f.metadata()),
            end_frags.iter().map(|f| f.metadata()),
        );
        let out_schema = schema.clone();
        let candidate_end = end.clone();
        let candidate_begin = begin.clone();
        let batches = stream::iter(delta.candidates)
            .map(move |(before, after)| {
                deleted_batches_in_fragment(
                    candidate_begin.clone(),
                    candidate_end.clone(),
                    before,
                    after,
                    schema.clone(),
                )
            })
            .buffered(get_num_compute_intensive_cpus())
            .try_flatten()
            .map_err(DataFusionError::from)
            .try_filter(|batch| std::future::ready(batch.num_rows() > 0));
        let candidates: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(out_schema.clone(), batches));
        if delta.added.is_empty() && delta.changed.is_empty() {
            return Ok(DatasetRecordBatchStream::new(candidates));
        }

        // A candidate is only deleted if it is live nowhere at end: a moved
        // row lands in a fragment the range created, a restored one where a
        // deletion vector shrank. Subtracting those newly live ids is an
        // anti join, run the way merge_insert runs its joins: sorted with
        // spilling past the memory pool, so a delta of any size is bounded.
        let live = live_id_batches(begin.clone(), end, delta.added, delta.changed, out_schema);
        let stream = anti_join(
            candidates,
            live,
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )?;
        Ok(DatasetRecordBatchStream::new(stream))
    }

    /// Get inserted rows between the two versions.
    ///
    /// This returns rows where `_row_created_at_version` is greater than `begin_version`
    /// and less than or equal to `end_version`.
    ///
    /// The result always includes:
    /// - `_row_created_at_version`: Version when the row was created
    /// - `_row_last_updated_at_version`: Version when the row was last updated
    /// - `_rowid`: Row ID
    /// - All other columns from the dataset
    ///
    /// # Returns
    ///
    /// A stream of record batches containing the inserted rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use futures::TryStreamExt;
    /// # async fn example(dataset: &Dataset, previous_version: u64) -> Result<()> {
    /// let delta = dataset.delta()
    ///     .compared_against_version(previous_version)
    ///     .build()?;
    /// let mut inserted = delta.get_inserted_rows().await?;
    /// while let Some(batch) = inserted.try_next().await? {
    ///     // Process batch...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_inserted_rows(&self) -> Result<DatasetRecordBatchStream> {
        let mut scanner = self.base_dataset.scan();

        // Enable version columns
        scanner.project(&[
            WILDCARD,
            ROW_ID,
            ROW_CREATED_AT_VERSION,
            ROW_LAST_UPDATED_AT_VERSION,
        ])?;

        // Filter for rows created in the version range
        let filter = self.build_inserted_rows_filter().await?;
        scanner.filter(&filter)?;

        scanner.try_into_stream().await
    }

    async fn build_inserted_rows_filter(&self) -> Result<String> {
        let (begin_version, end_version) = self.resolve_range().await?;
        Ok(format!(
            "_row_created_at_version > {} AND _row_created_at_version <= {}",
            begin_version, end_version
        ))
    }

    /// Get updated rows between the two versions.
    ///
    /// This returns rows where `_row_last_updated_at_version` is greater than `begin_version`
    /// and less than or equal to `end_version`, but `_row_created_at_version` is less than
    /// or equal to `begin_version` (to exclude newly inserted rows).
    ///
    /// The result always includes:
    /// - `_row_created_at_version`: Version when the row was created
    /// - `_row_last_updated_at_version`: Version when the row was last updated
    /// - `_rowid`: Row ID
    /// - All other columns from the dataset
    ///
    /// # Returns
    ///
    /// A stream of record batches containing the updated rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use futures::TryStreamExt;
    /// # async fn example(dataset: &Dataset, previous_version: u64) -> Result<()> {
    /// let delta = dataset.delta()
    ///     .compared_against_version(previous_version)
    ///     .build()?;
    /// let mut updated = delta.get_updated_rows().await?;
    /// while let Some(batch) = updated.try_next().await? {
    ///     // Process batch...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_updated_rows(&self) -> Result<DatasetRecordBatchStream> {
        let mut scanner = self.base_dataset.scan();

        // Enable version columns
        scanner.project(&[
            WILDCARD,
            ROW_ID,
            ROW_CREATED_AT_VERSION,
            ROW_LAST_UPDATED_AT_VERSION,
        ])?;

        // Filter for rows that were updated (not inserted) in the version range
        let filter = self.build_updated_rows_batch_filter().await?;
        scanner.filter(&filter)?;

        scanner.try_into_stream().await
    }

    async fn build_updated_rows_batch_filter(&self) -> Result<String> {
        let (begin_version, end_version) = self.resolve_range().await?;
        Ok(format!(
            "_row_created_at_version <= {} AND _row_last_updated_at_version > {} AND _row_last_updated_at_version <= {}",
            begin_version, begin_version, end_version
        ))
    }

    /// Get upserted rows between the two versions.
    ///
    /// This returns rows meet following conditions:
    /// Condition 1:
    ///     `_row_last_updated_at_version` is greater than `begin_version`
    ///     and less than or equal to `end_version`, but `_row_created_at_version` is less than
    ///     or equal to `begin_version` (to exclude newly inserted rows).
    /// Condition 2:
    ///     This returns rows where `_row_created_at_version` is greater than `begin_version`
    ///     and less than or equal to `end_version`.
    ///
    /// The result always includes:
    /// - `_row_created_at_version`: Version when the row was created
    /// - `_row_last_updated_at_version`: Version when the row was last updated
    /// - `_rowid`: Row ID
    /// - All other columns from the dataset
    ///
    /// # Returns
    ///
    /// A stream of record batches containing the updated and inserted rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use futures::TryStreamExt;
    /// # async fn example(dataset: &Dataset, previous_version: u64) -> Result<()> {
    /// let delta = dataset.delta()
    ///     .compared_against_version(previous_version)
    ///     .build()?;
    /// let mut updated = delta.get_upserted_rows().await?;
    /// while let Some(batch) = updated.try_next().await? {
    ///     // Process batch...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_upserted_rows(&self) -> Result<DatasetRecordBatchStream> {
        let mut scanner = self.base_dataset.scan();

        // Enable version columns
        scanner.project(&[
            WILDCARD,
            ROW_ID,
            ROW_CREATED_AT_VERSION,
            ROW_LAST_UPDATED_AT_VERSION,
        ])?;

        // Filter for rows that were updated or inserted in the version range
        let filter = self.build_upserted_rows_filter().await?;
        scanner.filter(&filter)?;

        scanner.try_into_stream().await
    }

    async fn build_upserted_rows_filter(&self) -> Result<String> {
        let inserted_row_filter = self.build_inserted_rows_filter().await?;
        let updated_rows_filter = self.build_updated_rows_batch_filter().await?;
        Ok(format!(
            "({}) OR ({})",
            inserted_row_filter, updated_rows_filter
        ))
    }
}

/// A fragment's deletion vector at this version, empty where it has none.
async fn deletion_offsets(
    dataset: Arc<Dataset>,
    fragment: &Fragment,
) -> Result<Arc<DeletionVector>> {
    let fragment = FileFragment::new(dataset, fragment.clone());
    Ok(fragment.get_deletion_vector().await?.unwrap_or_default())
}

/// The fragment-level shape of a version range, from metadata alone: a
/// shared fragment whose deletion file is unchanged has neither lost nor
/// regained a row, so nothing else costs any I/O. Each entry carries the
/// fragment metadata it names, so readers never look ids up in a manifest.
struct FragmentDelta {
    /// Fragments only the end version holds: every live row is newly live.
    added: Vec<Fragment>,
    /// Shared fragments whose deletion vector changed, as (begin, end)
    /// metadata: only rows a shrink revived are newly live.
    changed: Vec<(Fragment, Fragment)>,
    /// Begin fragments that can have lost rows, with their end-version
    /// metadata where they survive: vanished, or a changed deletion vector.
    candidates: Vec<(Fragment, Option<Fragment>)>,
}

fn fragment_delta<'a>(
    begin: impl Iterator<Item = &'a Fragment>,
    end: impl Iterator<Item = &'a Fragment>,
) -> FragmentDelta {
    let begin_meta: HashMap<u64, &Fragment> = begin.map(|f| (f.id, f)).collect();
    let mut added = Vec::new();
    let mut changed = Vec::new();
    let mut end_meta = HashMap::new();
    for fragment in end {
        end_meta.insert(fragment.id, fragment);
        match begin_meta.get(&fragment.id) {
            None => added.push(fragment.clone()),
            Some(before) if before.deletion_file != fragment.deletion_file => {
                changed.push(((*before).clone(), fragment.clone()));
            }
            Some(_) => {}
        }
    }
    let candidates = begin_meta
        .into_values()
        .filter_map(|before| match end_meta.get(&before.id) {
            None => Some((before.clone(), None)),
            Some(after) if after.deletion_file != before.deletion_file => {
                Some((before.clone(), Some((*after).clone())))
            }
            Some(_) => None,
        })
        .collect();
    FragmentDelta {
        added,
        changed,
        candidates,
    }
}

/// The ids one begin-version fragment lost by the end version, a batch at a
/// time. The offsets are iterated straight off the deletion vectors, so a
/// fragment's deletions are never held whole.
async fn deleted_batches_in_fragment(
    begin: Arc<Dataset>,
    end: Arc<Dataset>,
    before: Fragment,
    after: Option<Fragment>,
    schema: Arc<ArrowSchema>,
) -> Result<impl Stream<Item = Result<RecordBatch>> + Send> {
    let before_dv = deletion_offsets(begin.clone(), &before).await?;
    let emit = if let Some(after) = after {
        // The rows it lost are the offsets its deletion vector gained.
        let after_dv = deletion_offsets(end, &after).await?;
        let before_dv = before_dv.clone();
        let gained: Box<dyn Iterator<Item = u32> + Send> = Box::new(
            DeletionVector::clone(&after_dv)
                .into_sorted_iter()
                .filter(move |offset| !before_dv.contains(*offset)),
        );
        Emit::At(gained.peekable())
    } else {
        // Gone: every row it still held at the begin version left with it.
        Emit::Skipping(before_dv)
    };

    let sequence = load_row_id_sequence(&begin, &before).await?;
    Ok(id_batches(SequenceCursor::new(sequence, emit), schema))
}

/// Batches of the ids newly live at the end version: every live row of a
/// fragment the range created, and the rows a shrunk deletion vector
/// revived in a shared one. A growth-only change revives nothing and feeds
/// nothing.
fn live_id_batches(
    begin: Arc<Dataset>,
    end: Arc<Dataset>,
    added: Vec<Fragment>,
    changed: Vec<(Fragment, Fragment)>,
    schema: Arc<ArrowSchema>,
) -> SendableRecordBatchStream {
    // Paired with the begin-version metadata for a shared fragment; a
    // fragment the range created has none.
    let fragments: Vec<(Fragment, Option<Fragment>)> = added
        .into_iter()
        .map(|f| (f, None))
        .chain(
            changed
                .into_iter()
                .map(|(before, after)| (after, Some(before))),
        )
        .collect();
    let batches = stream::iter(fragments)
        .map(move |(fragment, before)| {
            let (begin, end, schema) = (begin.clone(), end.clone(), schema.clone());
            async move {
                let end_dv = deletion_offsets(end.clone(), &fragment).await?;
                let sequence = load_row_id_sequence(&end, &fragment).await?;
                let emit = match before {
                    None => Emit::Skipping(end_dv),
                    Some(before) => {
                        let begin_dv = deletion_offsets(begin.clone(), &before).await?;
                        // Lazy: a mass restore revives offsets without ever
                        // holding them whole.
                        let revived: Box<dyn Iterator<Item = u32> + Send> = Box::new(
                            DeletionVector::clone(&begin_dv)
                                .into_sorted_iter()
                                .filter(move |offset| !end_dv.contains(*offset)),
                        );
                        Emit::At(revived.peekable())
                    }
                };
                Ok::<_, Error>(id_batches(SequenceCursor::new(sequence, emit), schema))
            }
        })
        .buffered(get_num_compute_intensive_cpus())
        .try_flatten()
        .map_err(DataFusionError::from);
    let schema = Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()]));
    Box::pin(RecordBatchStreamAdapter::new(schema, batches))
}

/// One forward traversal of a row id sequence, resumable across batches.
/// The cursor keeps only positions and reads storage through the shared
/// sequence each round, so nothing is cloned, and resuming re-walks no
/// prefix.
struct SequenceCursor {
    sequence: Arc<RowIdSequence>,
    segment: usize,
    /// Length of the current segment, computed once on entry: encoded
    /// cardinality is not constant-time.
    segment_len: Option<usize>,
    /// Rows of the current segment already consumed.
    consumed: usize,
    /// Global offset of the next unconsumed row.
    offset: u32,
    /// Value resume point for the sorted range-backed encodings; the
    /// array-backed ones resume by element through `consumed`.
    next_value: u64,
    emit: Emit,
}

/// Which of the traversed ids to emit.
enum Emit {
    /// Every offset the deletion vector does not hold.
    Skipping(Arc<DeletionVector>),
    /// Exactly these offsets, ascending.
    At(std::iter::Peekable<Box<dyn Iterator<Item = u32> + Send>>),
}

/// A segment's ids from a resume point, without cloning storage or
/// re-walking what came before.
fn segment_ids<'a>(
    segment: &'a U64Segment,
    consumed: usize,
    next_value: u64,
) -> Box<dyn Iterator<Item = u64> + 'a> {
    match segment {
        U64Segment::Range(range) => Box::new(next_value.max(range.start)..range.end),
        U64Segment::RangeWithHoles { range, holes } => {
            let start = next_value.max(range.start);
            Box::new((start..range.end).filter(move |&v| holes.binary_search(v).is_err()))
        }
        U64Segment::RangeWithBitmap { range, bitmap } => {
            let (base, start) = (range.start, next_value.max(range.start));
            Box::new((start..range.end).filter(move |&v| bitmap.get((v - base) as usize)))
        }
        U64Segment::SortedArray(array) | U64Segment::Array(array) => {
            Box::new((consumed..array.len()).filter_map(move |i| array.get(i)))
        }
    }
}

impl SequenceCursor {
    fn new(sequence: Arc<RowIdSequence>, emit: Emit) -> Self {
        Self {
            sequence,
            segment: 0,
            segment_len: None,
            consumed: 0,
            offset: 0,
            next_value: 0,
            emit,
        }
    }

    /// Append up to `cap` emitted ids to `out`, stopping early when the
    /// traversal is exhausted.
    fn fill(&mut self, out: &mut Vec<u64>, cap: usize) {
        while out.len() < cap {
            let Some(segment) = self.sequence.segments().get(self.segment) else {
                return;
            };
            let segment_len = *self.segment_len.get_or_insert_with(|| segment.len());
            let remaining = segment_len - self.consumed;
            if remaining == 0 {
                self.segment += 1;
                self.segment_len = None;
                self.consumed = 0;
                self.next_value = 0;
                continue;
            }
            // Hop the rest of a segment with no wanted offset in it without
            // touching its encoding.
            if let Emit::At(wanted) = &mut self.emit {
                let Some(target) = wanted.peek().copied() else {
                    return;
                };
                if (target - self.offset) as usize >= remaining {
                    self.offset += remaining as u32;
                    self.segment += 1;
                    self.segment_len = None;
                    self.consumed = 0;
                    self.next_value = 0;
                    continue;
                }
            }
            let mut ids = segment_ids(segment, self.consumed, self.next_value);
            match &mut self.emit {
                Emit::Skipping(dv) => {
                    let take = remaining.min(cap - out.len());
                    for _ in 0..take {
                        let Some(id) = ids.next() else {
                            debug_assert!(false, "sequence shorter than segment lengths");
                            return;
                        };
                        if !dv.contains(self.offset) {
                            out.push(id);
                        }
                        self.offset += 1;
                        self.consumed += 1;
                        self.next_value = id.saturating_add(1);
                    }
                }
                Emit::At(wanted) => {
                    while out.len() < cap {
                        let Some(target) = wanted.peek().copied() else {
                            return;
                        };
                        let skip = (target - self.offset) as usize;
                        if skip >= segment_len - self.consumed {
                            break;
                        }
                        let Some(id) = ids.nth(skip) else {
                            debug_assert!(false, "sequence shorter than segment lengths");
                            return;
                        };
                        out.push(id);
                        wanted.next();
                        self.consumed += skip + 1;
                        self.offset = target + 1;
                        self.next_value = id.saturating_add(1);
                    }
                }
            }
        }
    }
}

/// The cursor's ids, batched.
fn id_batches(
    cursor: SequenceCursor,
    schema: Arc<ArrowSchema>,
) -> impl Stream<Item = Result<RecordBatch>> + Send {
    let rows = deleted_row_id_batch_rows();
    stream::try_unfold(cursor, move |mut cursor| {
        let schema = schema.clone();
        async move {
            let mut ids: Vec<u64> = Vec::with_capacity(rows);
            cursor.fill(&mut ids, rows);
            if ids.is_empty() {
                return Ok(None);
            }
            let batch =
                RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(ids)) as ArrayRef])?;
            Ok(Some((batch, cursor)))
        }
    })
}

/// Candidates minus the live ids, streamed. Sort-merge rather than hash:
/// the sorts spill past the memory pool where a hash build cannot, so a
/// delta of any size runs in bounded memory.
fn anti_join(
    candidates: SendableRecordBatchStream,
    live: SendableRecordBatchStream,
    options: LanceExecutionOptions,
) -> Result<SendableRecordBatchStream> {
    let sorted = |stream: SendableRecordBatchStream| -> Result<Arc<dyn ExecutionPlan>> {
        let key = Column::new_with_schema(ROW_ID, stream.schema().as_ref())?;
        let ordering = LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(key),
            SortOptions::default(),
        )])
        .expect("one sort key");
        Ok(Arc::new(SortExec::new(
            ordering,
            Arc::new(OneShotExec::new(stream)),
        )))
    };
    let candidate_key = Column::new_with_schema(ROW_ID, candidates.schema().as_ref())?;
    let live_key = Column::new_with_schema(ROW_ID, live.schema().as_ref())?;
    let joined = Arc::new(SortMergeJoinExec::try_new(
        sorted(candidates)?,
        sorted(live)?,
        vec![(Arc::new(candidate_key), Arc::new(live_key))],
        None,
        JoinType::LeftAnti,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
    )?);
    execute_plan(joined, options)
}

#[cfg(test)]
mod tests {

    async fn collect_deleted(delta: &super::DatasetDelta) -> Vec<u64> {
        let mut ids = Vec::new();
        let mut stream = delta.get_deleted_row_ids().await.unwrap();
        while let Some(batch) = stream.try_next().await.unwrap() {
            ids.extend(
                batch[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied(),
            );
        }
        ids.sort_unstable();
        ids
    }

    use crate::dataset::transaction::Operation;
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::types::UInt64Type;
    use chrono::Duration;
    use futures::TryStreamExt;
    use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
    use lance_datagen::{BatchCount, RowCount, array};
    use mock_instant::thread_local::MockClock;
    use std::sync::Arc;

    async fn create_test_dataset(
        rows: usize,
        batches: usize,
        value: &str,
        stable_row_ids: bool,
    ) -> Dataset {
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8(value.to_string()))
            .into_reader_rows(
                RowCount::from(rows as u64),
                BatchCount::from(batches as u32),
            );

        let write_params = WriteParams {
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        Dataset::write(data, "memory://", Some(write_params))
            .await
            .unwrap()
    }

    async fn write_dataset_temp(
        dir: &lance_core::utils::tempfile::TempStrDir,
        start_key: i32,
        rows: usize,
        batches: usize,
        value: &str,
        stable_row_ids: bool,
        append: bool,
    ) -> Dataset {
        let data = lance_datagen::gen_batch()
            .col("key", array::step_custom::<Int32Type>(start_key, 1))
            .col("value", array::fill_utf8(value.to_string()))
            .into_reader_rows(
                RowCount::from(rows as u64),
                BatchCount::from(batches as u32),
            );

        let write_params = WriteParams {
            enable_stable_row_ids: stable_row_ids,
            mode: if append {
                crate::dataset::WriteMode::Append
            } else {
                crate::dataset::WriteMode::Create
            },
            ..Default::default()
        };
        Dataset::write(data, dir, Some(write_params)).await.unwrap()
    }

    async fn update_where<T: Into<Arc<Dataset>>>(ds: T, predicate: &str, value: &str) -> Dataset {
        let updated = crate::dataset::UpdateBuilder::new(ds.into())
            .update_where(predicate)
            .unwrap()
            .set("value", &format!("'{}'", value))
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        Arc::try_unwrap(updated.new_dataset).unwrap_or_else(|arc| arc.as_ref().clone())
    }

    async fn scan_project_filter(
        ds: &Dataset,
        cols: &[&str],
        filter: Option<&str>,
    ) -> arrow_array::RecordBatch {
        let mut scanner = ds.scan();
        scanner.project(cols).unwrap();
        if let Some(f) = filter {
            scanner.filter(f).unwrap();
        }
        scanner.try_into_batch().await.unwrap()
    }

    // Optional: collect a stream of RecordBatch into a single batch
    async fn collect_stream(
        stream: crate::dataset::scanner::DatasetRecordBatchStream,
    ) -> arrow_array::RecordBatch {
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[tokio::test]
    async fn test_list_no_transaction() {
        let ds = create_test_dataset(1_000, 10, "value", false).await;
        let delta = ds.delta().compared_against_version(1).build().unwrap();
        let result = delta.list_transactions().await;
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_list_single_transaction() {
        let mut ds = create_test_dataset(1_000, 10, "value", false).await;
        ds.delete("key = 5").await.unwrap();

        let delta_struct = ds
            .delta()
            .with_begin_version(1)
            .with_end_version(ds.version().version)
            .build()
            .unwrap();
        let txs = delta_struct.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 1);
        assert!(matches!(txs[0].operation, Operation::Delete { .. }));
    }

    #[tokio::test]
    async fn test_list_multiple_transactions() {
        let mut ds = create_test_dataset(1_000, 10, "value", false).await;
        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();

        let delta_struct = ds
            .delta()
            .with_begin_version(1)
            .with_end_version(ds.version().version)
            .build()
            .unwrap();
        let txs = delta_struct.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 2);
    }

    #[tokio::test]
    async fn test_list_contains_deleted_transaction() {
        MockClock::set_system_time(std::time::Duration::from_secs(1));

        let mut ds = create_test_dataset(1_000, 10, "value", false).await;

        MockClock::set_system_time(std::time::Duration::from_secs(2));

        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();
        ds.delete("key = 7").await.unwrap();

        MockClock::set_system_time(std::time::Duration::from_secs(3));

        let end_version = ds.version().version;
        let base_dataset = ds.clone();

        MockClock::set_system_time(std::time::Duration::from_secs(4));

        ds.cleanup_old_versions(Duration::seconds(1), Some(true), None)
            .await
            .expect("Cleanup old versions failed");

        MockClock::set_system_time(std::time::Duration::from_secs(5));

        let delta_struct = base_dataset
            .delta()
            .with_begin_version(1)
            .with_end_version(end_version)
            .build()
            .unwrap();

        let result = delta_struct.list_transactions().await;
        match result {
            Err(lance_core::Error::VersionNotFound { message }) => {
                assert!(message.contains("Can not find version"));
            }
            _ => panic!("Expected VersionNotFound error."),
        }
    }

    #[tokio::test]
    async fn test_row_created_at_version_basic() {
        // Create dataset with stable row IDs enabled
        let ds = create_test_dataset(100, 1, "value", true).await;

        assert_eq!(ds.version().version, 1);

        // Scan with _row_created_at_version
        let result = scan_project_filter(&ds, &["key", ROW_CREATED_AT_VERSION], None).await;

        // All rows should have _row_created_at_version = 1
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();

        assert_eq!(result.num_rows(), 100);
        for version in created_at.iter() {
            assert_eq!(*version, 1);
        }
    }

    #[tokio::test]
    async fn test_row_last_updated_at_version_basic() {
        // Create dataset with stable row IDs enabled
        let ds = create_test_dataset(100, 1, "value", true).await;

        assert_eq!(ds.version().version, 1);

        // Update some rows (version 2)
        let ds = update_where(ds, "key < 30", "updated_v2").await;
        assert_eq!(ds.version().version, 2);

        // Update different rows (version 3)
        let ds = update_where(ds, "key >= 30 AND key < 50", "updated_v3").await;
        assert_eq!(ds.version().version, 3);

        // Update some rows again (version 4) - these rows were updated in v2
        let ds = update_where(ds, "key >= 10 AND key < 20", "updated_v4").await;
        assert_eq!(ds.version().version, 4);

        // Scan with _row_last_updated_at_version
        let result = scan_project_filter(&ds, &["key", ROW_LAST_UPDATED_AT_VERSION], None).await;

        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        assert_eq!(result.num_rows(), 100);

        for i in 0..result.num_rows() {
            let key = keys[i];
            if (10..20).contains(&key) {
                // Updated in v2, then again in v4 - should show v4
                assert_eq!(updated_at[i], 4);
            } else if key < 30 {
                // Updated only in v2 (but not in the 10-20 range)
                assert_eq!(updated_at[i], 2);
            } else if (30..50).contains(&key) {
                // Updated only in v3
                assert_eq!(updated_at[i], 3);
            } else {
                // Never updated - still at v1
                assert_eq!(updated_at[i], 1);
            }
        }
    }

    #[tokio::test]
    async fn test_row_version_metadata_after_update() {
        // Create dataset with stable row IDs enabled
        let ds = create_test_dataset(100, 1, "value", true).await;

        assert_eq!(ds.version().version, 1);

        // Update some rows (version 2)
        let ds = update_where(ds, "key < 10", "updated_v2").await;
        assert_eq!(ds.version().version, 2);

        // Update different rows (version 3)
        let ds = update_where(ds, "key >= 20 AND key < 30", "updated_v3").await;
        assert_eq!(ds.version().version, 3);

        // Update some of the same rows again (version 4)
        let ds = update_where(ds, "key >= 5 AND key < 15", "updated_v4").await;
        assert_eq!(ds.version().version, 4);

        // Scan with both version metadata columns
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            None,
        )
        .await;

        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        assert_eq!(result.num_rows(), 100);

        for i in 0..result.num_rows() {
            let key = keys[i];
            // All rows were created at version 1
            assert_eq!(created_at[i], 1);

            if (5..15).contains(&key) {
                // Updated in v4 (some also updated in v2)
                assert_eq!(updated_at[i], 4);
            } else if key < 10 {
                // Updated in v2 only (keys 0-4)
                assert_eq!(updated_at[i], 2);
            } else if (20..30).contains(&key) {
                // Updated in v3 only
                assert_eq!(updated_at[i], 3);
            } else {
                // Never updated - still at v1
                assert_eq!(updated_at[i], 1);
            }
        }
    }

    #[tokio::test]
    async fn test_row_version_metadata_after_append() {
        // Create initial dataset
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let ds = write_dataset_temp(&temp_dir, 0, 50, 1, "value", true, false).await;

        assert_eq!(ds.version().version, 1);

        // Append more data
        let ds = write_dataset_temp(&temp_dir, 50, 50, 1, "appended", true, true).await;

        assert_eq!(ds.version().version, 2);

        // Scan with both version metadata columns
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            None,
        )
        .await;

        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        assert_eq!(result.num_rows(), 100);

        for i in 0..result.num_rows() {
            let key = keys[i];
            if key < 50 {
                // Original rows created at version 1
                assert_eq!(created_at[i], 1);
                assert_eq!(updated_at[i], 1);
            } else {
                // Appended rows created at version 2
                assert_eq!(created_at[i], 2);
                assert_eq!(updated_at[i], 2);
            }
        }
    }

    #[tokio::test]
    async fn test_row_version_metadata_after_delete() {
        // Create dataset with stable row IDs enabled
        let mut ds = create_test_dataset(100, 1, "value", true).await;

        assert_eq!(ds.version().version, 1);

        // Delete some rows
        ds.delete("key < 10").await.unwrap();
        assert_eq!(ds.version().version, 2);

        // Scan with both version metadata columns
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            None,
        )
        .await;

        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        // Should have 90 rows remaining (100 - 10 deleted)
        assert_eq!(result.num_rows(), 90);

        for i in 0..result.num_rows() {
            let key = keys[i];
            // All remaining rows should be key >= 10
            assert!(key >= 10);
            // All rows were created at version 1
            assert_eq!(created_at[i], 1);
            // All rows still have last_updated at version 1 (delete doesn't update rows)
            assert_eq!(updated_at[i], 1);
        }
    }

    #[tokio::test]
    async fn test_row_version_metadata_combined() {
        // Create dataset with stable row IDs enabled
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let ds = Dataset::write(data, "memory://", Some(write_params))
            .await
            .unwrap();

        // Version 1: Initial write
        assert_eq!(ds.version().version, 1);

        // Version 2: Update some rows
        let updated = crate::dataset::UpdateBuilder::new(Arc::new(ds))
            .update_where("key >= 40 AND key < 50")
            .unwrap()
            .set("value", "'updated1'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let ds = updated.new_dataset;

        // Version 3: Update different rows
        let updated = crate::dataset::UpdateBuilder::new(ds)
            .update_where("key >= 50 AND key < 60")
            .unwrap()
            .set("value", "'updated2'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut ds = Arc::try_unwrap(updated.new_dataset).expect("no other Arc references");

        // Version 4: Delete some rows
        ds.delete("key < 10").await.unwrap();

        assert_eq!(ds.version().version, 4);

        // Scan with all metadata columns
        let result = ds
            .scan()
            .with_row_id()
            .project(&["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        // Should have 90 rows (100 - 10 deleted)
        assert_eq!(result.num_rows(), 90);

        for i in 0..result.num_rows() {
            let key = keys[i];
            let _row_id = row_ids[i];

            // All rows were created at version 1
            assert_eq!(created_at[i], 1);

            // Check last_updated_at_version based on key range
            if (40..50).contains(&key) {
                // Updated at version 2
                assert_eq!(updated_at[i], 2);
            } else if (50..60).contains(&key) {
                // Updated at version 3
                assert_eq!(updated_at[i], 3);
            } else {
                // Not updated, still at version 1
                assert_eq!(updated_at[i], 1);
            }
        }
    }

    #[tokio::test]
    async fn test_filter_by_row_created_at_version() {
        // Create initial dataset
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let ds = write_dataset_temp(&temp_dir, 0, 50, 1, "value", true, false).await;

        assert_eq!(ds.version().version, 1);

        // Append more data (version 2)
        let ds = write_dataset_temp(&temp_dir, 50, 50, 1, "appended", true, true).await;

        assert_eq!(ds.version().version, 2);

        // Test 1: Filter for rows created at version 1
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION],
            Some("_row_created_at_version = 1"),
        )
        .await;

        assert_eq!(result.num_rows(), 50);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 1);
            assert!(keys[i] < 50);
        }

        // Test 2: Filter for rows created at version 2
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION],
            Some("_row_created_at_version = 2"),
        )
        .await;

        assert_eq!(result.num_rows(), 50);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 2);
            assert!(keys[i] >= 50);
        }

        // Test 3: Filter for rows created at version >= 2
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION],
            Some("_row_created_at_version >= 2"),
        )
        .await;

        assert_eq!(result.num_rows(), 50);
        for i in 0..result.num_rows() {
            let created_at_val = result[ROW_CREATED_AT_VERSION]
                .as_primitive::<UInt64Type>()
                .value(i);
            assert!(created_at_val >= 2);
        }
    }

    #[tokio::test]
    async fn test_filter_by_row_last_updated_at_version() {
        // Create dataset with stable row IDs enabled
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let ds = Dataset::write(data, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(ds.version().version, 1);

        // Update some rows (version 2)
        let updated = crate::dataset::UpdateBuilder::new(Arc::new(ds))
            .update_where("key < 30")
            .unwrap()
            .set("value", "'updated_v2'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let ds = updated.new_dataset;
        assert_eq!(ds.version().version, 2);

        // Update different rows (version 3)
        let updated = crate::dataset::UpdateBuilder::new(ds)
            .update_where("key >= 30 AND key < 50")
            .unwrap()
            .set("value", "'updated_v3'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let ds = updated.new_dataset;
        assert_eq!(ds.version().version, 3);

        // Test 1: Filter for rows last updated at version 1
        let result = ds
            .scan()
            .project(&["key", ROW_LAST_UPDATED_AT_VERSION])
            .unwrap()
            .filter("_row_last_updated_at_version = 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Should have 50 rows (keys 50-99 that were never updated)
        assert_eq!(result.num_rows(), 50);
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(updated_at[i], 1);
            assert!(keys[i] >= 50);
        }

        // Test 2: Filter for rows last updated at version 2
        let result = ds
            .scan()
            .project(&["key", ROW_LAST_UPDATED_AT_VERSION])
            .unwrap()
            .filter("_row_last_updated_at_version = 2")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Should have 30 rows (keys 0-29)
        assert_eq!(result.num_rows(), 30);
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(updated_at[i], 2);
            assert!(keys[i] < 30);
        }

        // Test 3: Filter for rows last updated at version 3
        let result = ds
            .scan()
            .project(&["key", ROW_LAST_UPDATED_AT_VERSION])
            .unwrap()
            .filter("_row_last_updated_at_version = 3")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Should have 20 rows (keys 30-49)
        assert_eq!(result.num_rows(), 20);
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(updated_at[i], 3);
            assert!(keys[i] >= 30 && keys[i] < 50);
        }

        // Test 4: Filter for rows last updated at version > 1
        let result = ds
            .scan()
            .project(&["key", ROW_LAST_UPDATED_AT_VERSION])
            .unwrap()
            .filter("_row_last_updated_at_version > 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Should have 50 rows (30 from v2 + 20 from v3)
        assert_eq!(result.num_rows(), 50);
        for i in 0..result.num_rows() {
            let updated_at_val = result[ROW_LAST_UPDATED_AT_VERSION]
                .as_primitive::<UInt64Type>()
                .value(i);
            assert!(updated_at_val > 1);
        }
    }

    #[tokio::test]
    async fn test_filter_by_combined_version_columns() {
        // Create initial dataset
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let ds = write_dataset_temp(&temp_dir, 0, 50, 1, "value", true, false).await;

        assert_eq!(ds.version().version, 1);

        // Append more data (version 2)
        let ds = write_dataset_temp(&temp_dir, 50, 50, 1, "appended", true, true).await;

        assert_eq!(ds.version().version, 2);

        // Update some of the original rows (version 3)
        let ds = update_where(ds, "key >= 20 AND key < 30", "updated_v3").await;
        assert_eq!(ds.version().version, 3);

        // Test 1: Filter for rows created at v1 AND last updated at v1
        // (Original rows that were never updated)
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            Some("_row_created_at_version = 1 AND _row_last_updated_at_version = 1"),
        )
        .await;

        // Should have 40 rows (keys 0-19 and 30-49)
        assert_eq!(result.num_rows(), 40);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 1);
            assert_eq!(updated_at[i], 1);
            assert!(keys[i] < 50);
            assert!(keys[i] < 20 || keys[i] >= 30);
        }

        // Test 2: Filter for rows created at v1 AND last updated at v3
        // (Original rows that were updated in v3)
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            Some("_row_created_at_version = 1 AND _row_last_updated_at_version = 3"),
        )
        .await;

        // Should have 10 rows (keys 20-29)
        assert_eq!(result.num_rows(), 10);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 1);
            assert_eq!(updated_at[i], 3);
            assert!(keys[i] >= 20 && keys[i] < 30);
        }

        // Test 3: Filter for rows where created_at = last_updated_at
        // (Rows that were never updated after creation)
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            Some("_row_created_at_version = _row_last_updated_at_version"),
        )
        .await;

        // Should have 90 rows (40 from v1 that weren't updated + 50 from v2)
        assert_eq!(result.num_rows(), 90);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], updated_at[i]);
        }

        // Test 4: Filter for rows where created_at != last_updated_at
        // (Rows that were updated after creation)
        let result = scan_project_filter(
            &ds,
            &["key", ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION],
            Some("_row_created_at_version != _row_last_updated_at_version"),
        )
        .await;

        // Should have 10 rows (keys 20-29 that were updated)
        assert_eq!(result.num_rows(), 10);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_ne!(created_at[i], updated_at[i]);
            assert_eq!(created_at[i], 1);
            assert_eq!(updated_at[i], 3);
            assert!(keys[i] >= 20 && keys[i] < 30);
        }
    }

    #[tokio::test]
    async fn test_filter_version_columns_with_other_columns() {
        // Create dataset
        let ds = create_test_dataset(100, 1, "value", true).await;

        // Update some rows (version 2)
        let ds = update_where(ds, "key >= 30 AND key < 60", "updated").await;

        // Test: Combine version filter with regular column filter
        // Find rows where key < 50 AND last_updated_at_version = 2
        let result = scan_project_filter(
            &ds,
            &["key", "value", ROW_LAST_UPDATED_AT_VERSION],
            Some("key < 50 AND _row_last_updated_at_version = 2"),
        )
        .await;

        // Should have 20 rows (keys 30-49 that were updated in v2)
        assert_eq!(result.num_rows(), 20);
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(updated_at[i], 2);
            assert!(keys[i] >= 30 && keys[i] < 50);
        }
    }

    #[tokio::test]
    async fn test_get_inserted_rows() {
        // Create initial dataset (version 1)
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let ds = write_dataset_temp(&temp_dir, 0, 50, 1, "value", true, false).await;

        assert_eq!(ds.version().version, 1);

        // Append more data (version 2)
        let ds = write_dataset_temp(&temp_dir, 50, 30, 1, "appended_v2", true, true).await;

        assert_eq!(ds.version().version, 2);

        // Append more data (version 3)
        let ds = write_dataset_temp(&temp_dir, 80, 20, 1, "appended_v3", true, true).await;

        assert_eq!(ds.version().version, 3);

        // Test 1: Get all inserted rows between version 0 and 3
        let delta = ds
            .delta()
            .with_begin_version(0)
            .with_end_version(3)
            .build()
            .unwrap();

        let stream = delta.get_inserted_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should have all 100 rows
        assert_eq!(result.num_rows(), 100);
        assert!(result.column_by_name(ROW_ID).is_some());
        assert!(result.column_by_name(ROW_CREATED_AT_VERSION).is_some());
        assert!(result.column_by_name(ROW_LAST_UPDATED_AT_VERSION).is_some());

        // Test 2: Get inserted rows between version 1 and 2
        let delta = ds
            .delta()
            .with_begin_version(1)
            .with_end_version(2)
            .build()
            .unwrap();

        let stream = delta.get_inserted_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should have 30 rows (inserted in version 2)
        assert_eq!(result.num_rows(), 30);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 2);
            assert!(keys[i] >= 50 && keys[i] < 80);
        }

        // Test 3: Get inserted rows between version 2 and 3
        let delta = ds
            .delta()
            .with_begin_version(2)
            .with_end_version(3)
            .build()
            .unwrap();

        let stream = delta.get_inserted_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should have 20 rows (inserted in version 3)
        assert_eq!(result.num_rows(), 20);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 3);
            assert!(keys[i] >= 80 && keys[i] < 100);
        }
    }

    /// One deleted row must not drag the fragment's survivors through the
    /// join: a growth-only change revives nothing.
    #[tokio::test]
    async fn test_one_row_deletion_on_a_large_fragment() {
        let mut dataset = create_test_dataset(200_000, 1, "value", true).await;
        let begin = dataset.manifest.version;
        dataset.delete("key = 123456").await.unwrap();
        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        assert_eq!(collect_deleted(&delta).await, vec![123456]);
    }

    /// The cursor walks a many-segment sequence once, with either a skip
    /// set or a wanted list; a naive per-offset read is its oracle. The
    /// sequence covers every segment encoding, asserted below.
    #[test]
    fn test_sequence_cursor_matches_naive_reads() {
        use lance_core::utils::deletion::DeletionVector;
        use lance_table::rowids::RowIdSequence;
        use lance_table::rowids::segment::U64Segment;

        let mut sequence = RowIdSequence::from(100..200);
        sequence.extend(RowIdSequence::try_from_iter([5, 900, 42]).unwrap());
        sequence.extend(RowIdSequence::from(300..350));
        sequence.extend(RowIdSequence::try_from_iter((20_000..26_000).step_by(2)).unwrap());
        sequence.extend(
            RowIdSequence::try_from_iter((50_000..53_000).filter(|v| v % 997 != 0)).unwrap(),
        );
        // Large enough to span many bounded batches during the resume loops.
        sequence.extend(RowIdSequence::try_from_iter((100_000..500_000).step_by(4)).unwrap());
        sequence.extend(
            RowIdSequence::try_from_iter((600_000..700_000).filter(|v| v % 9973 != 0)).unwrap(),
        );
        let sequence = Arc::new(sequence);
        for expected in [
            |s: &U64Segment| matches!(s, U64Segment::Range(_)),
            |s: &U64Segment| matches!(s, U64Segment::Array(_) | U64Segment::SortedArray(_)),
            |s: &U64Segment| matches!(s, U64Segment::RangeWithBitmap { .. }),
            |s: &U64Segment| matches!(s, U64Segment::RangeWithHoles { .. }),
        ] {
            assert!(sequence.segments().iter().any(expected), "encoding missing");
        }
        let len = sequence.len() as u32;
        let dv = Arc::new(DeletionVector::from_iter(
            (0..len).step_by(97).chain([3u32, 101, 152]),
        ));

        let mut skipped: Vec<u64> = Vec::new();
        super::SequenceCursor::new(sequence.clone(), super::Emit::Skipping(dv.clone()))
            .fill(&mut skipped, usize::MAX);
        let naive: Vec<u64> = sequence
            .iter()
            .enumerate()
            .filter(|(offset, _)| !dv.contains(*offset as u32))
            .map(|(_, id)| id)
            .collect();
        assert_eq!(skipped, naive);

        // A tiny cap forces many resumes, covering the position keeping
        // across batches.
        let mut resumed: Vec<u64> = Vec::new();
        let mut cursor = super::SequenceCursor::new(sequence.clone(), super::Emit::Skipping(dv));
        loop {
            let before = resumed.len();
            cursor.fill(&mut resumed, before + 3);
            if resumed.len() == before {
                break;
            }
        }
        assert_eq!(resumed, naive);

        // The second list leaves whole segments and a consumed tail
        // unwanted, covering the hops.
        for wanted in [
            vec![
                0u32, 99, 100, 102, 152, 153, 154, 500, 3152, 3153, 4000, 6149,
            ],
            vec![5u32, 200, 6000, 6150, 106_000, 206_139],
        ] {
            let lazy: Box<dyn Iterator<Item = u32> + Send> = Box::new(wanted.clone().into_iter());
            let mut at: Vec<u64> = Vec::new();
            let mut cursor =
                super::SequenceCursor::new(sequence.clone(), super::Emit::At(lazy.peekable()));
            loop {
                let before = at.len();
                cursor.fill(&mut at, before + 1);
                if at.len() == before {
                    break;
                }
            }
            let naive: Vec<u64> = wanted
                .iter()
                .filter_map(|offset| sequence.get(*offset as usize))
                .collect();
            assert_eq!(at, naive);
        }
    }

    /// A range spanning the stable-id migration has a bare begin endpoint
    /// and is rejected, naming the offending version.
    #[tokio::test]
    async fn test_mixed_stable_id_endpoints_are_rejected() {
        let dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = write_dataset_temp(&dir, 0, 10, 1, "v1", false, false).await;
        dataset.migrate_to_stable_row_ids().await.unwrap();

        let delta = dataset.delta().compared_against_version(1).build().unwrap();
        let err = delta.get_deleted_row_ids().await.err().unwrap();
        assert!(
            err.to_string().contains("stable row ids") && err.to_string().contains("version 1"),
            "{err}"
        );
    }

    /// One deletion per fragment across many fragments.
    #[tokio::test]
    async fn test_deletes_across_many_fragments_are_reported() {
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(160), BatchCount::from(1));
        let params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let mut dataset = Dataset::write(data, "memory://", Some(params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 16);
        let begin = dataset.manifest.version;

        dataset.delete("key % 10 = 3").await.unwrap();

        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let expected: Vec<u64> = (0..16).map(|i| i * 10 + 3).collect();
        assert_eq!(collect_deleted(&delta).await, expected);
    }

    /// An append neither removes nor moves a row: the stream is empty.
    #[tokio::test]
    async fn test_appends_report_no_deleted_row_ids() {
        let dir = lance_core::utils::tempfile::TempStrDir::default();
        write_dataset_temp(&dir, 0, 10, 1, "v1", true, false).await;
        let ds = write_dataset_temp(&dir, 10, 10, 1, "v2", true, true).await;
        let delta = ds.delta().compared_against_version(1).build().unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(deleted.is_empty(), "an append deletes nothing: {deleted:?}");
    }

    /// Deletes on both sides of a merging compaction are all reported.
    #[tokio::test]
    async fn test_deletes_after_a_merging_compaction_are_reported() {
        use crate::dataset::optimize::{CompactionOptions, compact_files};

        let dir = lance_core::utils::tempfile::TempStrDir::default();
        write_dataset_temp(&dir, 0, 100, 1, "v1", true, false).await;
        let mut dataset = write_dataset_temp(&dir, 100, 100, 1, "v2", true, true).await;
        let begin = dataset.manifest.version;

        dataset.delete("key = 0 OR key = 100").await.unwrap();
        let options = CompactionOptions {
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        };
        compact_files(&mut dataset, options, None).await.unwrap();
        dataset.delete("key = 150").await.unwrap();

        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert_eq!(deleted, vec![0, 100, 150], "one id per deleted row");
    }

    /// Repeated partial updates leave fully tombstoned outputs; the result
    /// stays exact regardless.
    #[tokio::test]
    async fn test_repeated_updates_report_no_deleted_row_ids() {
        let mut dataset = create_test_dataset(100, 1, "value", true).await;
        for round in 0..4 {
            dataset = update_where(dataset, "key < 75", &format!("round {round}")).await;
        }
        let delta = dataset.delta().compared_against_version(1).build().unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(deleted.is_empty(), "updates delete nothing: {deleted:?}");
    }

    /// The anti join must stay correct when its build side exceeds the
    /// memory pool and spills.
    #[tokio::test]
    async fn test_anti_join_is_exact_under_a_tiny_memory_pool() {
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use lance_core::ROW_ID_FIELD;
        use lance_datafusion::exec::LanceExecutionOptions;

        let schema = Arc::new(arrow_schema::Schema::new(vec![ROW_ID_FIELD.clone()]));
        // ~8 MB of candidates against a 2 MB pool: the sorts must spill,
        // and the pool still clears DataFusion's fixed merge reservations.
        let candidate_ids: Vec<u64> = (0..1_000_000).collect();
        let live_ids: Vec<u64> = (0..1_000_000).filter(|id| id % 3 == 0).collect();
        let expected = candidate_ids.len() - live_ids.len();
        let as_stream = |ids: Vec<u64>| -> datafusion::physical_plan::SendableRecordBatchStream {
            let batches: Vec<_> = ids
                .chunks(8192)
                .map(|chunk| {
                    Ok(arrow_array::RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(arrow_array::UInt64Array::from(chunk.to_vec())) as _],
                    )
                    .unwrap())
                })
                .collect();
            Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(batches),
            ))
        };
        let stream = super::anti_join(
            as_stream(candidate_ids),
            as_stream(live_ids),
            LanceExecutionOptions {
                use_spilling: true,
                mem_pool_size: Some(2 * 1024 * 1024),
                ..Default::default()
            },
        )
        .unwrap();
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, expected, "anti join dropped or kept the wrong ids");
    }

    /// Interleaved updates leave every row live; none may read as deleted.
    #[tokio::test]
    async fn test_interleaved_updates_are_not_reported_as_deleted() {
        let dataset = create_test_dataset(100, 2, "value", true).await;
        let begin = dataset.manifest.version;
        let dataset = update_where(dataset, "key % 2 = 0", "even").await;
        let dataset = update_where(dataset, "key % 2 = 1", "odd").await;

        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(
            deleted.is_empty(),
            "every updated row is live at end: {deleted:?}"
        );
    }

    /// A restore must not rewind the row-id high-water mark, or the next
    /// append reuses old ids.
    #[tokio::test]
    async fn test_restore_preserves_the_row_id_high_water_mark() {
        let dir = lance_core::utils::tempfile::TempStrDir::default();
        write_dataset_temp(&dir, 0, 1, 1, "v1", true, false).await;
        // v2: append row A, taking the next stable id.
        let a = write_dataset_temp(&dir, 1, 1, 1, "v2", true, true).await;
        let begin = a.manifest.version;
        // v3: restore v1, dropping row A.
        let mut dataset = a.checkout_version(1).await.unwrap();
        dataset.restore().await.unwrap();
        // v4: append row B, which must not reuse A's id.
        let dataset = write_dataset_temp(&dir, 2, 1, 1, "v4", true, true).await;

        let delta = dataset
            .delta()
            .with_begin_version(begin)
            .with_end_version(dataset.manifest.version)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert_eq!(
            deleted,
            vec![1],
            "row A's id is gone, not reused: {deleted:?}"
        );
    }

    /// A mass delete-and-restore revives every row; the revived offsets are
    /// streamed, and the result is exact at scale.
    #[tokio::test]
    async fn test_mass_restore_reports_no_deleted_row_ids() {
        let mut dataset = create_test_dataset(50_000, 1, "value", true).await;
        dataset.delete("key >= 0").await.unwrap();
        let begin = dataset.manifest.version;
        let mut restored = dataset.checkout_version(1).await.unwrap();
        restored.restore().await.unwrap();

        let delta = restored
            .delta()
            .with_begin_version(begin)
            .with_end_version(restored.manifest.version)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(
            deleted.is_empty(),
            "every row revived: {} ids",
            deleted.len()
        );
    }

    /// A restore drops the deletion vector an update left behind, so the
    /// updated row is live at both endpoints in a fragment both hold.
    #[tokio::test]
    async fn test_restored_updated_rows_are_not_reported_as_deleted() {
        let dataset = create_test_dataset(100, 2, "value", true).await;
        let updated = update_where(dataset, "key = 0", "changed").await;

        let mut restored = updated.checkout_version(1).await.unwrap();
        restored.restore().await.unwrap();
        let delta = restored
            .delta()
            .with_begin_version(2)
            .with_end_version(3)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(
            deleted.is_empty(),
            "a restored row is live at both endpoints: {deleted:?}"
        );
    }

    /// A reversed range would report the rows the range added as deleted.
    #[tokio::test]
    async fn test_deleted_row_ids_rejects_a_reversed_range() {
        let dir = lance_core::utils::tempfile::TempStrDir::default();
        write_dataset_temp(&dir, 0, 10, 1, "v1", true, false).await;
        let ds = write_dataset_temp(&dir, 10, 10, 1, "v2", true, true).await;
        let delta = ds
            .delta()
            .with_begin_version(2)
            .with_end_version(1)
            .build()
            .unwrap();
        let Err(err) = delta.get_deleted_row_ids().await else {
            panic!("a reversed range must be rejected")
        };
        assert!(err.to_string().contains("newer than end version"), "{err}");
    }

    /// A window opening before v1 resolves begin to the version-0 sentinel:
    /// the empty snapshot, relative to which nothing is deleted.
    #[tokio::test]
    async fn test_deleted_row_ids_accepts_the_zero_version_sentinel() {
        MockClock::set_system_time(std::time::Duration::from_secs(100));
        let mut dataset = create_test_dataset(10, 1, "v1", true).await;
        MockClock::set_system_time(std::time::Duration::from_secs(200));
        dataset.delete("key = 0").await.unwrap();

        let delta = dataset
            .delta()
            .with_begin_date(chrono::DateTime::<chrono::Utc>::from_timestamp(50, 0).unwrap())
            .with_end_date(chrono::DateTime::<chrono::Utc>::from_timestamp(250, 0).unwrap())
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(
            deleted.is_empty(),
            "nothing is deleted relative to the empty snapshot: {deleted:?}"
        );
    }

    /// Unchanged shared fragments appear nowhere; changed ones on both
    /// sides; vanished as candidates; added as probed.
    #[test]
    fn test_fragment_delta_classifies_by_metadata() {
        use lance_table::format::{DeletionFile, DeletionFileType, Fragment};

        let deletion_file = |read_version| DeletionFile {
            read_version,
            id: 7,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(1),
            base_id: None,
        };
        let dv_version = |f: &Fragment| f.deletion_file.as_ref().unwrap().read_version;
        let unchanged = Fragment::new(1);
        let mut changed_before = Fragment::new(2);
        changed_before.deletion_file = Some(deletion_file(1));
        let mut changed_after = changed_before.clone();
        changed_after.deletion_file = Some(deletion_file(2));
        let vanished = Fragment::new(3);
        let added = Fragment::new(4);

        let begin = [unchanged.clone(), changed_before, vanished];
        let end = [unchanged, changed_after, added];
        let delta = super::fragment_delta(begin.iter(), end.iter());

        let added: Vec<u64> = delta.added.iter().map(|f| f.id).collect();
        assert_eq!(added, vec![4], "only the new fragment is added");
        let changed: Vec<(u64, u64, u64)> = delta
            .changed
            .iter()
            .map(|(b, a)| (b.id, dv_version(b), dv_version(a)))
            .collect();
        assert_eq!(
            changed,
            vec![(2, 1, 2)],
            "the changed pair carries each side's metadata"
        );
        let mut candidates: Vec<(u64, Option<u64>)> = delta
            .candidates
            .iter()
            .map(|(b, a)| (b.id, a.as_ref().map(dv_version)))
            .collect();
        candidates.sort_unstable();
        assert_eq!(
            candidates,
            vec![(2, Some(2)), (3, None)],
            "changed and vanished bear candidates, with end metadata where it survives"
        );
    }

    /// A configured batch size of zero would leave the stream unbounded.
    #[test]
    fn test_batch_rows_refuses_a_nonpositive_configuration() {
        use super::{BATCH_SIZE_FALLBACK, DELETED_ROW_ID_BATCH_CAP, batch_rows};
        assert_eq!(batch_rows(Some(0)), BATCH_SIZE_FALLBACK);
        assert_eq!(batch_rows(None), BATCH_SIZE_FALLBACK);
        assert_eq!(batch_rows(Some(64)), 64);
        assert_eq!(batch_rows(Some(usize::MAX)), DELETED_ROW_ID_BATCH_CAP);
    }

    /// An update rewrites a row under the same stable id, so the old
    /// fragment gains a deletion offset for a row that still exists.
    #[tokio::test]
    async fn test_updated_rows_are_not_reported_as_deleted() {
        let dataset = create_test_dataset(100, 2, "value", true).await;
        let begin = dataset.manifest.version;

        let dataset = update_where(dataset, "key >= 10 AND key < 20", "changed").await;

        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert!(
            deleted.is_empty(),
            "an update is not a deletion: {deleted:?}"
        );
    }

    /// A fragment's deletions can outnumber one batch.
    #[tokio::test]
    async fn test_deleted_row_ids_arrive_in_bounded_batches() {
        let rows = super::deleted_row_id_batch_rows() * 2 + 100;
        let mut dataset = create_test_dataset(rows, 1, "value", true).await;
        let begin = dataset.manifest.version;
        dataset.delete("true").await.unwrap();

        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let mut stream = delta.get_deleted_row_ids().await.unwrap();
        let mut sizes = Vec::new();
        while let Some(batch) = stream.try_next().await.unwrap() {
            sizes.push(batch.num_rows());
        }
        assert_eq!(sizes.iter().sum::<usize>(), rows, "{sizes:?}");
        assert!(
            sizes
                .iter()
                .all(|n| *n <= super::deleted_row_id_batch_rows()),
            "a batch exceeded the bound: {sizes:?}"
        );
    }

    /// Deleted ids are recoverable even though the rows cannot be scanned,
    /// and a compaction in the range is not mistaken for deletion.
    #[tokio::test]
    async fn test_get_deleted_row_ids() {
        use crate::dataset::optimize::{CompactionOptions, compact_files};

        let mut dataset = create_test_dataset(100, 2, "value", true).await;
        let begin = dataset.manifest.version;

        dataset.delete("key >= 10 AND key < 20").await.unwrap();
        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let deleted = collect_deleted(&delta).await;
        assert_eq!(deleted.len(), 10, "one id per deleted row: {deleted:?}");

        // Compaction rewrites the surviving rows into new fragments; their
        // ids are unchanged, so the deleted set must not grow.
        // Materializing the deletions rewrites the fragment under a new id.
        let options = CompactionOptions {
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(
            metrics.fragments_removed > 0,
            "the compaction case is vacuous unless fragments were actually rewritten"
        );
        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let after_compaction = collect_deleted(&delta).await;
        assert_eq!(
            after_compaction, deleted,
            "compaction moved live rows; only the deleted ids may be reported"
        );

        // A row deleted after its fragment was compacted away is still
        // addressable, so surviving an address lookup does not prove it lives.
        dataset.delete("key >= 20 AND key < 30").await.unwrap();
        let delta = dataset
            .delta()
            .compared_against_version(begin)
            .build()
            .unwrap();
        let after_delete = collect_deleted(&delta).await;
        assert_eq!(
            after_delete.len(),
            20,
            "deletes on both sides of the compaction must be reported: {after_delete:?}"
        );
    }

    #[tokio::test]
    async fn test_get_updated_rows() {
        // Create initial dataset (version 1)
        let ds = create_test_dataset(100, 1, "value", true).await;

        assert_eq!(ds.version().version, 1);

        // Update some rows (version 2)
        let ds = update_where(ds, "key < 30", "updated_v2").await;
        assert_eq!(ds.version().version, 2);

        // Update different rows (version 3)
        let ds = update_where(ds, "key >= 50 AND key < 70", "updated_v3").await;
        assert_eq!(ds.version().version, 3);

        // Update some rows again (version 4)
        let ds = update_where(ds, "key >= 10 AND key < 20", "updated_v4").await;
        assert_eq!(ds.version().version, 4);

        // Test 1: Get updated rows between version 1 and 2
        let delta = ds
            .delta()
            .with_begin_version(1)
            .with_end_version(2)
            .build()
            .unwrap();

        let stream = delta.get_updated_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should have 20 rows (keys 0-9 and 20-29)
        // Note: keys 10-19 were updated in v2 but then updated again in v4,
        // so they have _row_last_updated_at_version = 4, not 2
        assert_eq!(result.num_rows(), 20);
        assert!(result.column_by_name(ROW_ID).is_some());
        assert!(result.column_by_name(ROW_CREATED_AT_VERSION).is_some());
        assert!(result.column_by_name(ROW_LAST_UPDATED_AT_VERSION).is_some());

        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 1); // Created at version 1
            assert_eq!(updated_at[i], 2); // Updated at version 2
            // Keys should be in range [0, 30) but excluding [10, 20)
            assert!(keys[i] < 30);
            assert!(keys[i] < 10 || keys[i] >= 20);
        }

        // Test 2: Get updated rows between version 2 and 3
        let delta = ds
            .delta()
            .with_begin_version(2)
            .with_end_version(3)
            .build()
            .unwrap();

        let stream = delta.get_updated_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should have 20 rows (keys 50-69)
        assert_eq!(result.num_rows(), 20);
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            assert_eq!(updated_at[i], 3);
            assert!(keys[i] >= 50 && keys[i] < 70);
        }

        // Test 3: Get updated rows between version 1 and 4 (includes all updates)
        let delta = ds
            .delta()
            .with_begin_version(1)
            .with_end_version(4)
            .build()
            .unwrap();

        let stream = delta.get_updated_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should have 50 rows total (30 from v2, 20 from v3, 10 from v4)
        // But some rows were updated twice, so we get unique rows
        assert_eq!(result.num_rows(), 50);
        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();

        for i in 0..result.num_rows() {
            assert_eq!(created_at[i], 1); // All created at version 1
        }
    }

    #[tokio::test]
    async fn test_get_upsert_rows() {
        // Create initial dataset (version 1)
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let ds = write_dataset_temp(&temp_dir, 0, 50, 1, "value", true, false).await;

        assert_eq!(ds.version().version, 1);

        // Append inserted rows (version 2)
        let ds = write_dataset_temp(&temp_dir, 50, 20, 1, "appended_v2", true, true).await;
        assert_eq!(ds.version().version, 2);

        // Update some existing rows (version 3)
        let ds = update_where(ds, "key < 10", "updated_v3").await;
        assert_eq!(ds.version().version, 3);

        // Get upserted rows between version 1 and 3
        let delta = ds
            .delta()
            .with_begin_version(1)
            .with_end_version(3)
            .build()
            .unwrap();

        let stream = delta.get_upserted_rows().await.unwrap();
        let result = collect_stream(stream).await;

        // Should include 20 inserted rows (keys 50-69) and 10 updated rows (keys 0-9)
        assert_eq!(result.num_rows(), 30);
        assert!(result.column_by_name(ROW_ID).is_some());
        assert!(result.column_by_name(ROW_CREATED_AT_VERSION).is_some());
        assert!(result.column_by_name(ROW_LAST_UPDATED_AT_VERSION).is_some());

        let created_at = result[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let updated_at = result[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values();
        let keys = result["key"].as_primitive::<Int32Type>().values();

        for i in 0..result.num_rows() {
            let key = keys[i];
            if key < 10 {
                // Updated rows from version 3
                assert_eq!(created_at[i], 1);
                assert_eq!(updated_at[i], 3);
            } else {
                // Inserted rows from version 2
                assert!((50..70).contains(&key));
                assert_eq!(created_at[i], 2);
                assert_eq!(updated_at[i], 2);
            }
        }
    }

    #[tokio::test]
    async fn test_build_with_date_window_basic() {
        MockClock::set_system_time(std::time::Duration::from_secs(10));
        let ds = create_test_dataset(50, 1, "v1", true).await;
        assert_eq!(ds.version().version, 1);

        MockClock::set_system_time(std::time::Duration::from_secs(20));
        let ds = update_where(ds, "key < 10", "v2").await;
        assert_eq!(ds.version().version, 2);

        MockClock::set_system_time(std::time::Duration::from_secs(30));
        let ds = update_where(ds, "key >= 10 AND key < 20", "v3").await;
        assert_eq!(ds.version().version, 3);

        let begin_ts = chrono::DateTime::<chrono::Utc>::from_timestamp(15, 0).unwrap();
        let end_ts = chrono::DateTime::<chrono::Utc>::from_timestamp(25, 0).unwrap();

        let delta = ds
            .delta()
            .with_begin_date(begin_ts)
            .with_end_date(end_ts)
            .build()
            .unwrap();

        let txs = delta.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 1);
    }

    #[tokio::test]
    async fn test_build_with_date_window_edges() {
        MockClock::set_system_time(std::time::Duration::from_secs(100));
        let ds = create_test_dataset(10, 1, "v1", true).await;
        assert_eq!(ds.version().version, 1);

        MockClock::set_system_time(std::time::Duration::from_secs(200));
        let ds = update_where(ds, "key < 5", "v2").await;
        assert_eq!(ds.version().version, 2);

        let begin_ts = chrono::DateTime::<chrono::Utc>::from_timestamp(50, 0).unwrap();
        let end_ts = chrono::DateTime::<chrono::Utc>::from_timestamp(250, 0).unwrap();

        let delta = ds
            .delta()
            .with_begin_date(begin_ts)
            .with_end_date(end_ts)
            .build()
            .unwrap();

        let txs = delta.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 2);
    }

    #[tokio::test]
    async fn test_build_with_date_open_end_uses_latest() {
        MockClock::set_system_time(std::time::Duration::from_secs(10));
        let ds = create_test_dataset(20, 1, "v1", true).await;
        assert_eq!(ds.version().version, 1);

        MockClock::set_system_time(std::time::Duration::from_secs(20));
        let ds = update_where(ds, "key < 5", "v2").await;
        assert_eq!(ds.version().version, 2);

        MockClock::set_system_time(std::time::Duration::from_secs(30));
        let ds = update_where(ds, "key >= 5 AND key < 10", "v3").await;
        assert_eq!(ds.version().version, 3);

        let begin_ts = chrono::DateTime::<chrono::Utc>::from_timestamp(15, 0).unwrap();

        let delta = ds.delta().with_begin_date(begin_ts).build().unwrap();

        let txs = delta.list_transactions().await.unwrap();
        // Should include transactions at v2 and v3
        assert_eq!(txs.len(), 2);
    }
}
