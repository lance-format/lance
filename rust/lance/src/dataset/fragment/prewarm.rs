// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Encoded data-cache warming for a physical fragment selection.

use std::collections::BTreeMap;

use futures::{StreamExt, TryStreamExt, stream};
use lance_core::{Error, Result, datatypes::Schema};
use lance_file::prewarm::{EncodedPagePlan, plan_encoded_pages};
use lance_file::reader::FileReader;
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::reader_projection_from_field_ids;
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use roaring::RoaringBitmap;

use super::{FileFragment, data_file_projection};

/// Bound the transient encoded bytes retained by each warming read to 1 MiB.
const PREWARM_READ_BYTES: u64 = 1024 * 1024;

#[derive(Debug)]
struct FilePrewarmPlan {
    scheduler: FileScheduler,
    pages: EncodedPagePlan,
}

/// A snapshot-specific data plan retaining the same object-store readers as queries.
///
/// Construct with [`FileFragment::plan_data_prewarm`]. Byte planning reads file
/// metadata, but never decodes projected values or builds payload batches.
#[derive(Debug, Default)]
pub struct DataPrewarmPlan {
    files: Vec<FilePrewarmPlan>,
}

impl DataPrewarmPlan {
    /// Encoded bytes selected, excluding the metadata I/O required to plan them.
    pub fn num_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.pages.num_bytes()).sum()
    }

    /// Projected pages selected, before coalescing their encoded buffers.
    pub fn num_pages(&self) -> u64 {
        self.files.iter().map(|f| f.pages.selected_pages).sum()
    }

    /// Reasons individual files required whole-column encoded warming.
    pub fn fallback_reasons(&self) -> impl Iterator<Item = &str> {
        self.files
            .iter()
            .filter_map(|f| f.pages.fallback_reason.as_deref())
    }

    /// Fetch encoded bytes through the query's object-store stack.
    ///
    /// `concurrency` bounds simultaneous reads; each read retains at most 1 MiB.
    /// Completion means reads succeeded, not that cache eviction is prevented.
    pub async fn execute(&self, concurrency: usize) -> Result<()> {
        if concurrency == 0 {
            return Err(Error::invalid_input(
                "prewarm read concurrency must be greater than zero",
            ));
        }
        for file in &self.files {
            let scheduler = file.scheduler.clone();
            // Own the range descriptors across the async boundary. Borrowed
            // nested iterator closures otherwise prevent this future being Send.
            let chunks = read_chunks(file.pages.byte_ranges.clone());
            stream::iter(chunks)
                .map(move |range| {
                    let scheduler = scheduler.clone();
                    async move { scheduler.submit_single(range, 0).await.map(|_| ()) }
                })
                .buffer_unordered(concurrency)
                .try_collect::<Vec<_>>()
                .await?;
        }
        Ok(())
    }
}

fn read_chunks(
    ranges: Vec<std::ops::Range<u64>>,
) -> impl Iterator<Item = std::ops::Range<u64>> + Send {
    ranges.into_iter().flat_map(|range| {
        let end = range.end;
        (range.start..end)
            .step_by(PREWARM_READ_BYTES as usize)
            .map(move |start| start..start.saturating_add(PREWARM_READ_BYTES).min(end))
    })
}

impl FileFragment {
    /// Plan the encoded buffers overlapping fragment-relative physical rows.
    ///
    /// `None` selects the whole fragment; an empty bitmap selects nothing.
    /// Deletions do not renumber this selection. Nested projections can require
    /// whole-column buffers. Legacy files, overlays, and out-of-line blobs return
    /// [`Error::NotSupported`], so callers can explicitly choose a coarser path.
    /// Base paths and object-store wrappers are resolved exactly as in reads.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// for fragment in dataset.get_fragments() {
    ///     let plan = fragment.plan_data_prewarm(dataset.schema(), None).await?;
    ///     plan.execute(4).await?;
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn plan_data_prewarm(
        &self,
        projection: &Schema,
        rows: Option<&RoaringBitmap>,
    ) -> Result<DataPrewarmPlan> {
        let mut plan = DataPrewarmPlan::default();
        if rows.is_some_and(RoaringBitmap::is_empty) {
            return Ok(plan);
        }
        if !self.metadata.overlays.is_empty() {
            return Err(Error::not_supported(
                "encoded data prewarm requires an overlay-aware fallback",
            ));
        }
        for file in &self.metadata.files {
            let schema = data_file_projection(projection, &file.schema(self.dataset.schema()))?;
            if schema.fields.is_empty() {
                continue;
            }
            let version = file.file_version()?;
            if version == ConcreteFileVersion::V1 {
                return Err(Error::not_supported(
                    "encoded data prewarm is unavailable for legacy files",
                ));
            }
            if schema.fields_pre_order().any(|f| f.is_blob()) {
                return Err(Error::not_supported(
                    "encoded data prewarm requires blob descriptor resolution",
                ));
            }
            let store = self.dataset.object_store(file.base_id).await?;
            let scheduler =
                ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
            let directory = self.dataset.data_file_dir(file)?;
            file.validate(&directory)?;
            let path = directory.join(file.path.as_str());
            let scheduler = scheduler
                .open_file(&path, &file.file_size_bytes)
                .await?
                .with_metadata_read_size(lance_file::reader::FOOTER_LEN)?;
            let metadata = FileReader::read_all_metadata(&scheduler).await?;
            let field_columns = file
                .fields
                .iter()
                .zip(file.column_indices.iter())
                .filter(|(_, column)| **column >= 0)
                .map(|(field, column)| (*field as u32, *column as u32))
                .collect::<BTreeMap<_, _>>();
            let projection = reader_projection_from_field_ids(version, &schema, &field_columns)?;
            let pages = plan_encoded_pages(&metadata, &projection, rows)?;
            plan.files.push(FilePrewarmPlan { scheduler, pages });
        }
        Ok(plan)
    }
}
