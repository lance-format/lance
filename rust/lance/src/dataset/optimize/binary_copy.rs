// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::Result;
use crate::dataset::DATA_DIR;
use crate::dataset::WriteParams;
use crate::dataset::fragment::write::generate_random_filename;
use crate::datatypes::Schema;
use lance_core::Error;
use lance_encoding::decoder::{ColumnInfo, PageInfo as DecPageInfo};
use lance_file::reader::FileReader as LFReader;
use lance_file::version::ConcreteFileVersion;
use lance_file::versions as file_versions;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, Fragment};
use prost::Message;
use prost_types::Any;
use std::ops::Range;
use std::sync::Arc;

async fn init_writer_if_necessary(
    dataset: &Dataset,
    version: ConcreteFileVersion,
    current_writer: &mut Option<FileWriter>,
    current_filename: &mut Option<String>,
) -> Result<bool> {
    if current_writer.is_none() {
        let filename = format!("{}.lance", generate_random_filename());
        let path = dataset.base.clone().join(DATA_DIR).join(filename.as_str());
        let object_writer = dataset.object_store.create(&path).await?;
        *current_writer = Some(file_versions::create_lazy_writer(
            version,
            object_writer,
            FileWriterOptions::default(),
        )?);
        *current_filename = Some(filename);
        return Ok(true);
    }
    Ok(false)
}

/// Finalize the current output file and return it as a single [Fragment].
/// - Ensures an output writer / filename is present (creates a new file if needed).
/// - Converts the in-memory `col_pages` / `col_buffers` into `ColumnInfo` metadata, draining them.
/// - Lets the exact file version normalize copied column metadata.
/// - Writes the Lance footer via [flush_footer] and registers the resulting [DataFile] in a [Fragment].
///
/// PAY ATTENTION current function will:
/// - Takes (`Option::take`) the current writer and filename.
/// - Drains `col_pages` and `col_buffers` for all columns.
#[allow(clippy::too_many_arguments)]
async fn finalize_current_output_file(
    schema: &Schema,
    version: ConcreteFileVersion,
    current_writer: &mut Option<FileWriter>,
    current_filename: &mut Option<String>,
    current_page_table: &[ColumnInfo],
    col_pages: &mut [Vec<DecPageInfo>],
    col_buffers: &mut [Vec<(u64, u64)>],
    total_rows_in_current: u64,
) -> Result<Fragment> {
    let mut final_cols: Vec<Arc<ColumnInfo>> = Vec::with_capacity(current_page_table.len());
    for (i, column_info) in current_page_table.iter().enumerate() {
        let mut pages_vec = std::mem::take(&mut col_pages[i]);
        file_versions::finalize_external_metadata_column(
            version,
            schema,
            i,
            &mut pages_vec,
            total_rows_in_current,
        )?;
        let pages_arc = Arc::from(pages_vec.into_boxed_slice());
        let buffers_vec = std::mem::take(&mut col_buffers[i]);
        final_cols.push(Arc::new(ColumnInfo::new(
            column_info.index,
            pages_arc,
            buffers_vec,
            column_info.encoding.clone(),
        )));
    }
    let mut writer = current_writer
        .take()
        .ok_or_else(|| Error::internal("binary copy output writer was not initialized"))?;
    flush_footer(&mut writer, schema, &final_cols, total_rows_in_current).await?;

    // Register the newly closed output file as a fragment data file
    let mut fragment = Fragment::new(0);
    let (field_ids, field_column_indices) = file_versions::data_file_columns(version, schema);
    let filename = current_filename
        .take()
        .ok_or_else(|| Error::internal("binary copy output filename was not initialized"))?;
    let mut data_file = DataFile::new_unstarted(filename, version);
    data_file.fields = field_ids.into();
    data_file.column_indices = field_column_indices.into();
    fragment.files.push(data_file);
    fragment.physical_rows = Some(total_rows_in_current as usize);
    Ok(fragment)
}

/// Rewrite the files in a single task using binary copy semantics.
///
/// Flow overview (per task):
/// fragments
///   └── data files
///         └── columns
///               └── pages (batched reads) -> aligned writes -> page metadata
///         └── column buffers -> aligned writes -> buffer metadata
///   └── flush when target rows reached -> write footer -> fragment metadata
///   └── final flush for remaining rows
///
/// Behavior highlights:
/// - Assumes all input files share the same Lance file version.
/// - Preserves stable row ids by concatenating row-id sequences when enabled.
/// - Delegates physical-column mapping and copied metadata normalization to the exact file version.
/// - Flushes an output file once `max_rows_per_file` rows are accumulated, then repeats.
///
/// Parameters:
/// - `dataset`: target dataset (for storage/config and schema).
/// - `fragments`: fragments to merge via binary copy (assumed consistent versions).
/// - `params`: write parameters (uses `max_rows_per_file`).
/// - `read_batch_bytes_opt`: optional I/O batch size when coalescing page reads.
pub async fn rewrite_files_binary_copy(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    fragments: &[Fragment],
    params: &WriteParams,
    read_batch_bytes_opt: Option<usize>,
) -> Result<Vec<Fragment>> {
    if fragments.is_empty() || fragments.iter().any(|fragment| fragment.files.is_empty()) {
        return Err(Error::invalid_input(
            "binary copy requires at least one data file",
        ));
    }

    // Binary copy algorithm overview:
    // - Reads page and buffer regions directly from source files in bounded batches
    // - Appends them to a new output file with alignment, updating offsets
    // - Recomputes page priorities by adding the cumulative row count to preserve order
    // - Writes a new footer (schema descriptor, column metadata, offset tables, version)
    // - Optionally carries forward stable row ids and persists them inline in fragment metadata
    // Merge small Lance files into larger ones by page-level binary copy.
    let schema = dataset.schema().clone();
    let column_count = schema
        .fields
        .iter()
        .map(|field| file_versions::physical_column_count(version, field))
        .sum();

    let mut out: Vec<Fragment> = Vec::new();
    let mut current_writer: Option<FileWriter> = None;
    let mut current_filename: Option<String> = None;
    let mut current_page_table: Vec<ColumnInfo> = Vec::new();
    // Baseline column encodings captured from the first source file; all subsequent
    // files must match per-column to safely concatenate column-level buffers.
    let mut baseline_col_encoding_bytes: Vec<Vec<u8>> = Vec::new();

    // Column-list<Page-List<DecPageInfo>>
    let mut col_pages: Vec<Vec<DecPageInfo>> = std::iter::repeat_with(Vec::<DecPageInfo>::new)
        .take(column_count)
        .collect();
    let mut col_buffers: Vec<Vec<(u64, u64)>> = vec![Vec::new(); column_count];
    let mut total_rows_in_current: u64 = 0;
    let max_rows_per_file = params.max_rows_per_file as u64;

    // Visit each fragment and all of its data files (a fragment may contain multiple files)
    for frag in fragments.iter() {
        for df in frag.files.iter() {
            let object_store = if let Some(base_id) = df.base_id {
                dataset.object_store(Some(base_id)).await?
            } else {
                dataset.object_store.clone()
            };
            let full_path = dataset.data_file_dir(df)?.clone().join(df.path.as_str());
            let scan_scheduler = ScanScheduler::new(
                object_store.clone(),
                SchedulerConfig::max_bandwidth(&object_store),
            );
            let file_scheduler = scan_scheduler
                .open_file_with_priority(&full_path, 0, &df.file_size_bytes)
                .await?;
            let file_meta = LFReader::read_all_metadata(&file_scheduler).await?;
            let src_column_infos = file_meta.column_infos.clone();
            // Initialize current_page_table
            if current_page_table.is_empty() {
                current_page_table = src_column_infos
                    .iter()
                    .map(|column_index| ColumnInfo {
                        index: column_index.index,
                        buffer_offsets_and_sizes: Arc::from(
                            Vec::<(u64, u64)>::new().into_boxed_slice(),
                        ),
                        page_infos: Arc::from(Vec::<DecPageInfo>::new().into_boxed_slice()),
                        encoding: column_index.encoding.clone(),
                    })
                    .collect();
                baseline_col_encoding_bytes = src_column_infos
                    .iter()
                    .map(|ci| Ok(Any::from_msg(&ci.encoding)?.encode_to_vec()))
                    .collect::<Result<Vec<_>>>()?;
            }

            // Iterate through each column of the current data file of the current fragment
            for (col_idx, src_column_info) in src_column_infos.iter().enumerate() {
                let has_existing_pages = !col_pages[col_idx].is_empty();
                file_versions::copy_external_metadata_column(
                    version,
                    &schema,
                    col_idx,
                    has_existing_pages,
                    || async {
                        init_writer_if_necessary(
                            dataset,
                            version,
                            &mut current_writer,
                            &mut current_filename,
                        )
                        .await?;

                        let read_batch_bytes: u64 =
                            read_batch_bytes_opt.unwrap_or(16 * 1024 * 1024) as u64;

                        let mut page_index = 0;

                        // Iterate through each page of the current column in the current data file of the current fragment
                        while page_index < src_column_info.page_infos.len() {
                    let mut batch_ranges: Vec<Range<u64>> = Vec::new();
                    let mut batch_counts: Vec<usize> = Vec::new();
                    let mut batch_bytes: u64 = 0;
                    let mut batch_pages: usize = 0;
                    // Build a single read batch by coalescing consecutive pages up to
                    // `read_batch_bytes` budget:
                    // - Accumulate total bytes (`batch_bytes`) and page count (`batch_pages`).
                    // - For each page, append its buffer ranges to `batch_ranges` and record
                    //   the number of buffers in `batch_counts` so returned bytes can be
                    //   mapped back to page boundaries.
                    // - Stop when adding the next page would exceed the byte budget, then
                    //   issue one I/O request for the collected ranges.
                    // - Advance `page_index` to reflect pages scheduled in this batch.
                    for current_page in &src_column_info.page_infos[page_index..] {
                        let page_bytes: u64 = current_page
                            .buffer_offsets_and_sizes
                            .iter()
                            .map(|(_, size)| *size)
                            .sum();
                        let would_exceed =
                            batch_pages > 0 && (batch_bytes + page_bytes > read_batch_bytes);
                        if would_exceed {
                            break;
                        }
                        batch_counts.push(current_page.buffer_offsets_and_sizes.len());
                        for (offset, size) in current_page.buffer_offsets_and_sizes.iter() {
                            if *size > 0 {
                                batch_ranges.push((*offset)..(*offset + *size));
                            }
                        }
                        batch_bytes += page_bytes;
                        batch_pages += 1;
                        page_index += 1;
                    }

                    let bytes_vec = if batch_ranges.is_empty() {
                        Vec::new()
                    } else {
                        // read many buffers at once
                        file_scheduler.submit_request(batch_ranges, 0).await?
                    };
                    let mut bytes_iter = bytes_vec.into_iter();

                    for (local_idx, buffer_count) in batch_counts.iter().enumerate() {
                        // Reconstruct the absolute page index within the source column:
                        // - `page_index` now points to the page position
                        // - `batch_pages` is how many pages we included in this batch
                        // - `local_idx` enumerates pages inside the batch [0..batch_pages)
                        // Therefore `page_index - batch_pages + local_idx` yields the exact
                        // source page we are currently materializing, allowing us to access
                        // its metadata (encoding, row count, buffers) for the new page entry.
                        let page_idx = page_index - batch_pages + local_idx;
                        let page = &src_column_info.page_infos[page_idx];
                        let mut new_offsets = Vec::with_capacity(*buffer_count);
                        for (buffer_idx, (_, size)) in
                            page.buffer_offsets_and_sizes.iter().enumerate()
                        {
                            let writer = current_writer.as_mut().ok_or_else(|| {
                                Error::internal("binary copy output writer was not initialized")
                            })?;
                            let bytes = if *size == 0 {
                                None
                            } else {
                                Some(bytes_iter.next().ok_or_else(|| {
                                    Error::execution(format!(
                                        "binary copy: missing page buffer bytes while rewriting data file \
                                         (column {col_idx}, page {page_idx}, buffer {buffer_idx}, expected size {size})",
                                    ))
                                })?)
                            };
                            let (start, written) = writer
                                .write_external_buffer(bytes.as_deref().unwrap_or_default())
                                .await?;
                            new_offsets.push((start, written));
                        }

                        // `priority` acts as the global row offset for this page, ensuring
                        // downstream iterators maintain the correct logical order across
                        // merged inputs.
                        let new_page_info = DecPageInfo {
                            num_rows: page.num_rows,
                            priority: page.priority + total_rows_in_current,
                            encoding: page.encoding.clone(),
                            buffer_offsets_and_sizes: Arc::from(new_offsets.into_boxed_slice()),
                        };
                        col_pages[col_idx].push(new_page_info);
                    }
                        } // finished scheduling & copying pages for this column in the current source file

                        if !src_column_info.buffer_offsets_and_sizes.is_empty() {
                    // Validate column-level encoding compatibility before copying buffers
                    let src_col_encoding_bytes =
                        Any::from_msg(&src_column_info.encoding)?.encode_to_vec();
                    let baseline_bytes = &baseline_col_encoding_bytes[col_idx];
                    if src_col_encoding_bytes != *baseline_bytes {
                        return Err(Error::execution(format!(
                            "binary copy: The ColumnEncoding of column {} is incompatible with the first file, \
                            making it impossible to safely concatenate buffers",
                            col_idx
                        )));
                    }
                    let ranges: Vec<Range<u64>> = src_column_info
                        .buffer_offsets_and_sizes
                        .iter()
                        .filter(|(_, size)| *size > 0)
                        .map(|(offset, size)| (*offset)..(*offset + *size))
                        .collect();
                    let bytes_vec = if ranges.is_empty() {
                        Vec::new()
                    } else {
                        file_scheduler.submit_request(ranges, 0).await?
                    };
                    let mut bytes_iter = bytes_vec.into_iter();
                    for (buffer_idx, (_, size)) in
                        src_column_info.buffer_offsets_and_sizes.iter().enumerate()
                    {
                        let writer = current_writer.as_mut().ok_or_else(|| {
                            Error::internal("binary copy output writer was not initialized")
                        })?;
                        let bytes = if *size == 0 {
                            None
                        } else {
                            Some(bytes_iter.next().ok_or_else(|| {
                                Error::execution(format!(
                                    "binary copy: missing column buffer bytes while rewriting data file \
                                     (column {col_idx}, buffer {buffer_idx}, expected size {size})",
                                ))
                            })?)
                        };
                        let (start, written) = writer
                            .write_external_buffer(bytes.as_deref().unwrap_or_default())
                            .await?;
                        col_buffers[col_idx].push((start, written));
                    }
                        }
                        Ok(())
                    },
                )
                .await?;
            } // finished all columns in the current source file

            // Accumulate rows for the current output file and flush when reaching the threshold
            total_rows_in_current += file_meta.num_rows;
            if total_rows_in_current >= max_rows_per_file {
                let fragment_out = finalize_current_output_file(
                    &schema,
                    version,
                    &mut current_writer,
                    &mut current_filename,
                    &current_page_table,
                    &mut col_pages,
                    &mut col_buffers,
                    total_rows_in_current,
                )
                .await?;

                // Reset state for next output file
                current_writer = None;
                current_page_table.clear();
                for v in col_pages.iter_mut() {
                    v.clear();
                }
                for v in col_buffers.iter_mut() {
                    v.clear();
                }
                out.push(fragment_out);
                total_rows_in_current = 0;
            }
        }
    } // Finished writing all fragments; any remaining data in memory will be flushed below

    if total_rows_in_current > 0 {
        // Flush remaining rows as a final output file
        init_writer_if_necessary(dataset, version, &mut current_writer, &mut current_filename)
            .await?;
        let frag = finalize_current_output_file(
            &schema,
            version,
            &mut current_writer,
            &mut current_filename,
            &current_page_table,
            &mut col_pages,
            &mut col_buffers,
            total_rows_in_current,
        )
        .await?;
        out.push(frag);
    }
    Ok(out)
}

/// Finalizes a compacted data file by writing the Lance footer via `FileWriter`.
///
/// This function does not manually craft the footer. Instead it:
/// - Pads the current `ObjectWriter` position to a 64‑byte boundary (required for v2_1+ readers).
/// - Initializes the active `FileWriter` from the collected column metadata.
/// - Calls `FileWriter::finish()` to emit column metadata, offset tables, global buffers
///   (schema descriptor), version, and to close the writer.
///
/// Preconditions:
/// - All page data and column‑level buffers referenced by `final_cols` have already been written
///   to `writer`; otherwise offsets in the footer will be invalid.
///
async fn flush_footer(
    writer: &mut FileWriter,
    schema: &Schema,
    final_cols: &[Arc<ColumnInfo>],
    total_rows_in_current: u64,
) -> Result<()> {
    writer.write_external_buffer(&[]).await?;
    writer.initialize_with_external_columns(schema.clone(), final_cols, total_rows_in_current)?;
    writer.finish().await?;
    Ok(())
}
