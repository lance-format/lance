// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use lance_core::{Error, Result, datatypes::Schema, utils::allocation::AllocationBudget};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use super::{
    Dataset,
    fragment::{
        AdmittedFragmentScanOutcome, BaseSchedulers, FileFragment, FragReadConfig, FragmentReader,
    },
};

/// Next physical position in the cursor's immutable dataset snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedScanPosition {
    /// Manifest fragment ID, not a manifest index or a stable logical row ID.
    pub fragment_id: u64,
    /// Next physical row offset within that fragment, including deleted rows.
    pub row_offset: u64,
}

/// Complete physical prefix, retryable refusal, or complete dataset exhaustion.
#[derive(Debug)]
pub enum AdmittedDatasetScanOutcome {
    /// Output for a complete prefix in manifest order. Deletion filtering can
    /// produce an empty batch without ending the scan.
    Batch {
        fragment_id: u64,
        rows: Range<u64>,
        batch: RecordBatch,
    },
    /// Drain output or replan the workspace; no physical rows were consumed.
    NeedsCapacity,
    /// All physical rows in every snapshot fragment have been consumed.
    Finished,
}

struct ActiveFragment {
    reader: FragmentReader,
    end: u64,
}

/// Lazy, sequential scan with one shared payload budget across all fragments.
///
/// Only the current fragment reader is retained. Output aliases from previous
/// fragments keep their allocation charges. Opening a fragment still uses the
/// ordinary metadata/cache path, which is outside the payload budget. This API
/// neither replaces the ordinary scanner nor establishes a process RSS bound.
pub struct AdmittedDatasetScan {
    dataset: Arc<Dataset>,
    projection: Schema,
    read_config: FragReadConfig,
    fragment_index: usize,
    row_offset: u64,
    active: Option<ActiveFragment>,
    max_rows: u32,
    workspace_bytes: usize,
    budget: Arc<AllocationBudget>,
}

impl Dataset {
    /// Scan snapshot fragments in manifest order through the admitted take path.
    ///
    /// Projection and read configuration are fixed at construction; no fragment
    /// is opened until `try_next`. Readers and schedulers are reused, with no
    /// future-fragment read-ahead or whole-dataset selection. The per-batch row
    /// ceiling and workspace do not cap cumulative output. Decoder support and
    /// exclusions are those of [`FragmentReader::try_take_admitted_fullzip`].
    ///
    /// Each attempt reserves the same `workspace_bytes` from `budget`, including
    /// selection, dependent input, scratch and output. `NeedsCapacity` is not EOF:
    /// independently drain retained output, or replan if a minimal row cannot fit.
    /// Errors and cancellation preserve the next unconsumed position. Already
    /// dispatched work retains its charge until it ends. Empty fragments may be
    /// skipped while opening; no nonempty physical prefix is skipped on failure.
    ///
    /// Metadata, deletion/ID descriptors, transport/cache and allocator overhead
    /// remain outside this payload budget. There is no filter/index/sort planning
    /// or automatic integration with the ordinary scanner. All returned aliases
    /// must be accounted for when deciding whether another scan can progress.
    ///
    /// ```
    /// use lance::{Dataset, Result, dataset::{AdmittedDatasetScan, AdmittedDatasetScanOutcome}};
    /// use lance_core::utils::allocation::AllocationBudget;
    /// fn begin(dataset: &Dataset) -> Result<AdmittedDatasetScan> {
    ///     dataset.try_scan_admitted_fullzip(
    ///         dataset.schema().clone(), Default::default(), 256, 65536,
    ///         AllocationBudget::new(65536),
    ///     )
    /// }
    /// async fn consume_until_blocked(scan: &mut AdmittedDatasetScan) -> Result<bool> {
    ///     loop {
    ///         match scan.try_next().await? {
    ///             AdmittedDatasetScanOutcome::Batch { batch, .. } => drop(batch),
    ///             // false means incomplete; keep the cursor to resume after draining.
    ///             AdmittedDatasetScanOutcome::NeedsCapacity => return Ok(false),
    ///             AdmittedDatasetScanOutcome::Finished => return Ok(true),
    ///         }
    ///     }
    /// }
    /// ```
    pub fn try_scan_admitted_fullzip(
        &self,
        projection: Schema,
        mut read_config: FragReadConfig,
        max_rows: u32,
        workspace_bytes: usize,
        budget: Arc<AllocationBudget>,
    ) -> Result<AdmittedDatasetScan> {
        if max_rows == 0 || workspace_bytes == 0 {
            return Err(Error::invalid_input(
                "admitted dataset scan requires positive max_rows and workspace_bytes",
            ));
        }
        if read_config.make_deletions_null
            && !read_config.with_row_id
            && !read_config.with_row_address
        {
            return Err(Error::invalid_input(
                "make_deletions_null requires with_row_id or with_row_address",
            ));
        }
        read_config.scan_scheduler.get_or_insert_with(|| {
            ScanScheduler::new(
                self.object_store.clone(),
                SchedulerConfig::max_bandwidth(&self.object_store),
            )
        });
        read_config
            .base_schedulers
            .get_or_insert_with(BaseSchedulers::max_bandwidth);
        Ok(AdmittedDatasetScan {
            dataset: Arc::new(self.clone()),
            projection,
            read_config,
            fragment_index: 0,
            row_offset: 0,
            active: None,
            max_rows,
            workspace_bytes,
            budget,
        })
    }
}

impl AdmittedDatasetScan {
    /// Replan the per-attempt workspace without consuming or restarting the scan.
    ///
    /// This does not change the shared budget or release output still owned by
    /// consumers. A workspace exceeding the pool's limit cannot be admitted;
    /// drain aliases and choose a quantum that fits the existing pool.
    ///
    /// ```
    /// # use lance::{Result, dataset::AdmittedDatasetScan};
    /// fn replan(scan: &mut AdmittedDatasetScan) -> Result<()> {
    ///     scan.set_workspace_bytes(65536)
    /// }
    /// ```
    pub fn set_workspace_bytes(&mut self, workspace_bytes: usize) -> Result<()> {
        if workspace_bytes == 0 {
            return Err(Error::invalid_input(
                "admitted dataset scan requires positive workspace_bytes",
            ));
        }
        self.workspace_bytes = workspace_bytes;
        Ok(())
    }

    /// Next unconsumed physical position; `None` only after snapshot exhaustion.
    pub fn position(&self) -> Option<AdmittedScanPosition> {
        self.dataset
            .manifest
            .fragments
            .get(self.fragment_index)
            .map(|fragment| AdmittedScanPosition {
                fragment_id: fragment.id,
                row_offset: self.row_offset,
            })
    }

    /// Attempt a complete prefix without waiting for credit or prefetching fragments.
    pub async fn try_next(&mut self) -> Result<AdmittedDatasetScanOutcome> {
        loop {
            let Some(position) = self.position() else {
                return Ok(AdmittedDatasetScanOutcome::Finished);
            };
            if self.active.is_none() {
                let fragment = FileFragment::new(
                    self.dataset.clone(),
                    self.dataset.manifest.fragments[self.fragment_index].clone(),
                );
                let end = fragment.physical_rows().await? as u64;
                if end == 0 {
                    self.fragment_index += 1;
                    tokio::task::yield_now().await;
                    continue;
                }
                let reader = fragment
                    .open(&self.projection, self.read_config.clone())
                    .await?;
                self.active = Some(ActiveFragment { reader, end });
            }
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| Error::internal("admitted dataset scan lost its active fragment"))?;
            let end = active.end;
            let mut scan = active.reader.try_scan_admitted_fullzip(
                self.row_offset..end,
                self.max_rows,
                self.workspace_bytes,
                self.budget.clone(),
            )?;
            let result = scan.try_next().await?;
            drop(scan);
            match result {
                AdmittedFragmentScanOutcome::Batch { rows, batch } => {
                    self.row_offset = rows.end;
                    if self.row_offset == end {
                        // Commit and release the old reader before the next open.
                        // Returned buffer owners keep their shared budget charges.
                        self.active = None;
                        self.fragment_index += 1;
                        self.row_offset = 0;
                    }
                    return Ok(AdmittedDatasetScanOutcome::Batch {
                        fragment_id: position.fragment_id,
                        rows,
                        batch,
                    });
                }
                AdmittedFragmentScanOutcome::NeedsCapacity => {
                    return Ok(AdmittedDatasetScanOutcome::NeedsCapacity);
                }
                AdmittedFragmentScanOutcome::Finished => {
                    return Err(Error::internal(
                        "admitted fragment scan exhausted before delivering its remaining prefix",
                    ));
                }
            }
        }
    }
}
