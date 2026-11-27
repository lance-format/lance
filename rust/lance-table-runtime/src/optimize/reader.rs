// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, mpsc};

use arrow_schema::Schema as ArrowSchema;
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::datatypes::BlobHandling;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn};
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD, Result};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use lance_table::format::Fragment;

use crate::blob::split_batch_by_bytes;
use crate::fragment::read_defaults::{
    BATCH_SIZE_FALLBACK, DEFAULT_FRAGMENT_READAHEAD, LEGACY_DEFAULT_FRAGMENT_READAHEAD,
    get_default_batch_size, get_default_io_buffer_size_override, validate_batch_size,
};
use crate::rowids::{CapturedRowIds, make_rowid_capture_stream};
use crate::{BaseSchedulers, FileFragment, FragReadConfig, Table};

/// Give `config` one scheduler for the table's own files and one per base for
/// files in other bases, sized by `io_buffer_size`, then
/// `LANCE_DEFAULT_IO_BUFFER_SIZE`, then the store's max bandwidth.
pub(super) fn with_read_budget(
    config: FragReadConfig,
    store: &Arc<ObjectStore>,
    io_buffer_size: Option<u64>,
) -> FragReadConfig {
    let io_buffer_size = io_buffer_size.or_else(get_default_io_buffer_size_override);
    let scheduler_config = io_buffer_size.map_or_else(
        || SchedulerConfig::max_bandwidth(store),
        SchedulerConfig::new,
    );
    let base_schedulers =
        io_buffer_size.map_or_else(BaseSchedulers::max_bandwidth, BaseSchedulers::new);
    config
        .with_scan_scheduler(ScanScheduler::new(store.clone(), scheduler_config))
        .with_base_schedulers(base_schedulers)
}

pub(super) struct RewriteReader {
    pub batches: SendableRecordBatchStream,
    pub row_ids: Option<mpsc::Receiver<CapturedRowIds>>,
    pub has_blob_v2: bool,
}

/// Read the task's fragments in order with the fragment reader, which applies
/// overlays before filtering deletions. Legacy blobs need their payload bytes,
/// and Blob V2 rewrites consume descriptors paired with physical addresses.
pub(super) fn prepare_reader(
    table: &Table,
    fragments: &[Fragment],
    batch_size: Option<usize>,
    io_buffer_size: Option<u64>,
    capture_row_ids: bool,
) -> Result<RewriteReader> {
    let has_blob_v2 = table
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob_v2());
    let blob_handling = BlobHandling::SomeBlobsBinary(
        table
            .schema()
            .fields_pre_order()
            .filter(|field| field.is_blob() && !field.is_blob_v2())
            .map(|field| field.id as u32)
            .collect(),
    );
    let mut projection = table.schema().clone();
    projection.fields = projection
        .fields
        .into_iter()
        .map(|field| blob_handling.unload_if_needed(field))
        .collect();
    let mut output = projection.clone();
    if capture_row_ids {
        output.extend(std::slice::from_ref(&ROW_ID_FIELD))?;
    }
    if has_blob_v2 {
        output.extend(std::slice::from_ref(&ROW_ADDR_FIELD))?;
    }
    let schema = Arc::new(ArrowSchema::from(&output));

    let store = &table.state.object_store;
    let is_legacy =
        table.manifest.data_storage_format.lance_file_format() == ConcreteFileVersion::V1;
    let default_batch_size = || {
        get_default_batch_size()
            .unwrap_or_else(|| (store.block_size() / 4).max(BATCH_SIZE_FALLBACK))
    };
    let batch_size = if is_legacy {
        get_default_batch_size()
            .or(batch_size)
            .unwrap_or_else(default_batch_size)
    } else {
        batch_size.unwrap_or_else(default_batch_size)
    };
    let batch_size = validate_batch_size(batch_size)?;
    let fragment_readahead = if is_legacy {
        LEGACY_DEFAULT_FRAGMENT_READAHEAD
    } else {
        DEFAULT_FRAGMENT_READAHEAD
            .unwrap_or_else(|| store.io_parallelism() * 2)
            .max(1)
    };

    let mut config = FragReadConfig::default()
        .with_row_id(capture_row_ids)
        .with_row_address(has_blob_v2);
    let mut batch_size_bytes = None;
    if !is_legacy {
        config = with_read_budget(config, store, io_buffer_size);
        if let Some(file_options) = table.file_reader_options() {
            batch_size_bytes = file_options.batch_size_bytes;
            config = config.with_file_reader_options(file_options.clone());
        }
    }

    let table = Arc::new(table.clone());
    let projection = Arc::new(projection);
    let reads = stream::iter(fragments.to_vec())
        .enumerate()
        .map({
            let table = table.clone();
            move |(priority, fragment)| {
                let fragment = FileFragment::new(table.clone(), fragment);
                let projection = projection.clone();
                let config = config.clone().with_reader_priority(priority as u32);
                let open = spawn(async move {
                    let reader = fragment.open(&projection, config).await?;
                    reader.read_all(batch_size).await
                });
                async move { open.await? }
            }
        })
        .buffered(fragment_readahead)
        .map_ok(|tasks| tasks.map(Ok))
        .try_flatten()
        .try_buffered(get_num_compute_intensive_cpus().max(1))
        .try_filter(|batch| std::future::ready(batch.num_rows() > 0));

    let reads = reads
        .and_then(move |batch| {
            std::future::ready(match batch_size_bytes {
                Some(budget) => split_batch_by_bytes(batch, (budget * 2) as usize),
                None => Ok(vec![batch]),
            })
        })
        .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
        .try_flatten();
    let batches = Box::pin(RecordBatchStreamAdapter::new(schema, reads));
    let (batches, row_ids) = if capture_row_ids {
        let (batches, row_ids) =
            make_rowid_capture_stream(batches, table.manifest.uses_stable_row_ids())?;
        (batches, Some(row_ids))
    } else {
        (batches as SendableRecordBatchStream, None)
    };
    Ok(RewriteReader {
        batches,
        row_ids,
        has_blob_v2,
    })
}
