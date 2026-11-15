// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded fixture writer and ordinary/admitted reader comparison.
//! Run performance builds and measurements on a managed VM, not locally.
//! See benches/ADMITTED_FULLZIP.md for provenance and interpretation limits.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{
    Array, LargeBinaryArray, ListArray, RecordBatch, StringArray,
    builder::{LargeBinaryBuilder, ListBuilder, StringBuilder},
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_core::{Error, Result, cache::LanceCache, utils::allocation::AllocationBudget};
use lance_encoding::{
    constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY},
    decoder::{DecoderPlugins, FilterExpression},
};
use lance_file::{
    reader::{FileReader, FileReaderOptions, FullZipScanOutcome},
    version::ConcreteFileVersion,
    versions,
    writer::FileWriterOptions,
};
use lance_io::{
    ReadBatchParams,
    object_store::ObjectStore,
    scheduler::{IoStats, ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};

fn invalid(message: &str) -> Error {
    Error::invalid_input(message)
}

fn word(row: u64, frame: usize, index: usize) -> u64 {
    let mut value = row.wrapping_mul(0x9e3779b97f4a7c15)
        ^ (frame as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ index as u64;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn schema() -> Arc<Schema> {
    let fields = [
        Field::new("row", DataType::Utf8, false),
        Field::new(
            "payload",
            DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
            true,
        ),
    ];
    Arc::new(Schema::new(
        fields
            .into_iter()
            .map(|field| {
                field.with_metadata(HashMap::from([(
                    STRUCTURAL_ENCODING_META_KEY.to_string(),
                    STRUCTURAL_ENCODING_FULLZIP.to_string(),
                )]))
            })
            .collect::<Vec<_>>(),
    ))
}

fn batch(start: u64, rows: usize, bytes: usize) -> Result<RecordBatch> {
    let mut ids = StringBuilder::new();
    let mut lists = ListBuilder::new(LargeBinaryBuilder::new());
    let mut value = vec![0; bytes];
    for row in start..start + rows as u64 {
        ids.append_value(format!("{row:016x}"));
        if row % 11 == 1 {
            lists.append(false);
            continue;
        }
        if row % 13 == 2 {
            lists.append(true);
            continue;
        }
        for frame in 0..2 {
            if row % 17 == 3 && frame == 1 {
                lists.values().append_null();
                continue;
            }
            if row % 19 == 4 && frame == 0 {
                lists.values().append_value([]);
                continue;
            }
            for (i, chunk) in value.chunks_exact_mut(8).enumerate() {
                chunk.copy_from_slice(&word(row, frame, i).to_le_bytes());
            }
            lists.values().append_value(&value);
        }
        lists.append(true);
    }
    Ok(RecordBatch::try_new(
        schema(),
        vec![Arc::new(ids.finish()), Arc::new(lists.finish())],
    )?)
}

async fn write(uri: &str, rows: u64, bytes: usize, page_rows: usize) -> Result<()> {
    if rows == 0
        || bytes == 0
        || !bytes.is_multiple_of(8)
        || page_rows < 4
        || !rows.is_multiple_of(page_rows as u64)
    {
        return Err(invalid(
            "use positive rows divisible by page_rows >= 4 and positive value bytes divisible by 8",
        ));
    }
    if page_rows
        .checked_mul(bytes)
        .and_then(|n| n.checked_mul(2))
        .is_none_or(|n| n > 64 * 1024 * 1024)
    {
        return Err(invalid(
            "fixture writer batch must fit 64 MiB; reduce page_rows",
        ));
    }
    let (store, path) = ObjectStore::from_uri(uri).await?;
    let mut writer = versions::create_writer(
        ConcreteFileVersion::V2_2,
        store.create(&path).await?,
        schema().as_ref().try_into()?,
        FileWriterOptions {
            data_cache_bytes: Some(1),
            ..Default::default()
        },
    )?;
    for start in (0..rows).step_by(page_rows) {
        writer.write_batch(&batch(start, page_rows, bytes)?).await?;
    }
    writer.finish().await?;
    Ok(())
}

#[derive(Clone, Copy)]
struct ReadOptions {
    admitted: bool,
    expected_rows: u64,
    max_rows: u32,
    workspace: usize,
    value_bytes: usize,
    full_verify: bool,
    delay_ms: u64,
    lite: bool,
}

#[derive(Debug, Default)]
struct Sink {
    rows: u64,
    payload_bytes: u64,
    fingerprint: u64,
}

impl Sink {
    fn consume(&mut self, batch: &RecordBatch, bytes: usize, full: bool) -> Result<()> {
        let ids = batch
            .column_by_name("row")
            .and_then(|a| a.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| invalid("missing string row column"))?;
        let lists = batch
            .column_by_name("payload")
            .and_then(|a| a.as_any().downcast_ref::<ListArray>())
            .ok_or_else(|| invalid("missing list payload column"))?;
        for i in 0..batch.num_rows() {
            let row = self.rows;
            if ids.is_null(i)
                || ids.value(i) != format!("{row:016x}")
                || lists.is_null(i) != (row % 11 == 1)
            {
                return Err(invalid("row order or list validity mismatch"));
            }
            if lists.is_valid(i) {
                let array = lists.value(i);
                let values = array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .ok_or_else(|| invalid("payload leaf must be LargeBinary"))?;
                let count = if row % 13 == 2 { 0 } else { 2 };
                if values.len() != count {
                    return Err(invalid("list length mismatch"));
                }
                for frame in 0..count {
                    let null = row % 17 == 3 && frame == 1;
                    if values.is_null(frame) != null {
                        return Err(invalid("leaf validity mismatch"));
                    }
                    if null {
                        continue;
                    }
                    let value = values.value(frame);
                    let len = if row % 19 == 4 && frame == 0 {
                        0
                    } else {
                        bytes
                    };
                    if value.len() != len {
                        return Err(invalid("payload length mismatch"));
                    }
                    self.payload_bytes += value.len() as u64;
                    if full {
                        for (j, chunk) in value.chunks_exact(8).enumerate() {
                            let actual = u64::from_le_bytes(chunk.try_into().unwrap());
                            if actual != word(row, frame, j) {
                                return Err(invalid("payload content mismatch"));
                            }
                            self.fingerprint = self.fingerprint.rotate_left(7) ^ actual;
                        }
                    }
                }
            }
            self.rows += 1;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Report {
    sink: Sink,
    batches: u64,
    first_batch_ms: f64,
    wall_ms: f64,
    sink_ms: f64,
    iops: u64,
    read_bytes: u64,
}

async fn read(uri: &str, options: ReadOptions) -> Result<Report> {
    if options.max_rows == 0
        || options.workspace == 0
        || options.value_bytes == 0
        || !options.value_bytes.is_multiple_of(8)
    {
        return Err(invalid(
            "positive max_rows/workspace and value bytes divisible by 8 required",
        ));
    }
    let started = Instant::now();
    let (store, path) = ObjectStore::from_uri(uri).await?;
    let scheduler = ScanScheduler::new(
        store,
        SchedulerConfig {
            io_buffer_size_bytes: 64 * 1024 * 1024,
            use_lite_scheduler: Some(options.lite),
        },
    );
    let file = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await?;
    let stats = IoStats::new();
    let reader = FileReader::try_open(
        file,
        None,
        Arc::<DecoderPlugins>::default(),
        &Arc::new(LanceCache::no_cache()),
        FileReaderOptions::default(),
    )
    .await?
    .with_io_stats(stats.recorder());
    if reader.num_rows() != options.expected_rows {
        return Err(invalid(
            "fixture row count differs from requested validation",
        ));
    }
    let budget = AllocationBudget::new(options.workspace);
    let mut sink = Sink::default();
    let mut batches = 0;
    let mut first_batch_ms = 0.0;
    let mut sink_time = Duration::ZERO;
    let mut consume = |batch: RecordBatch| -> Result<()> {
        if batches == 0 {
            first_batch_ms = started.elapsed().as_secs_f64() * 1000.0;
        }
        let start = Instant::now();
        sink.consume(&batch, options.value_bytes, options.full_verify)?;
        sink_time += start.elapsed();
        batches += 1;
        Ok(())
    };
    if options.admitted {
        let mut scan = reader
            .open_admitted_fullzip_scan(
                0..reader.num_rows(),
                budget.clone(),
                options.workspace,
                u64::from(options.max_rows),
            )
            .await?;
        loop {
            match scan.try_next().await? {
                FullZipScanOutcome::Batch { batch, .. } => consume(batch)?,
                FullZipScanOutcome::NeedsCapacity => {
                    return Err(invalid(
                        "fixture row did not fit; incomplete read is not a benchmark success",
                    ));
                }
                FullZipScanOutcome::Finished => break,
            }
            if options.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(options.delay_ms)).await;
            }
        }
    } else {
        let mut stream = reader
            .read_stream(
                ReadBatchParams::RangeFull,
                options.max_rows,
                1,
                FilterExpression::no_filter(),
            )
            .await?;
        while let Some(batch) = stream.try_next().await? {
            consume(batch)?;
            if options.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(options.delay_ms)).await;
            }
        }
    }
    if sink.rows != reader.num_rows() || budget.reserved() != 0 {
        return Err(invalid("incomplete row count or leaked allocation credit"));
    }
    let io = stats.snapshot();
    Ok(Report {
        sink,
        batches,
        first_batch_ms,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        sink_ms: sink_time.as_secs_f64() * 1000.0,
        iops: io.iops,
        read_bytes: io.bytes_read,
    })
}

fn number<T: std::str::FromStr>(value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| invalid("invalid numeric argument"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.get(1).map(String::as_str) {
        Some("write") if args.len() == 6 => {
            write(
                &args[2],
                number(&args[3])?,
                number(&args[4])?,
                number(&args[5])?,
            )
            .await?;
            println!("{{\"write_complete\":true}}");
        }
        Some("read") if args.len() == 11 => {
            let admitted = match args[3].as_str() {
                "ordinary" => false,
                "admitted" => true,
                _ => return Err(invalid("reader must be ordinary or admitted")),
            };
            let full = match args[8].as_str() {
                "full" => true,
                "rows" => false,
                _ => return Err(invalid("sink must be full or rows")),
            };
            let lite = match args[10].as_str() {
                "standard" => false,
                "lite" => true,
                _ => return Err(invalid("queue must be standard or lite")),
            };
            let report = read(
                &args[2],
                ReadOptions {
                    admitted,
                    expected_rows: number(&args[4])?,
                    max_rows: number(&args[5])?,
                    workspace: number(&args[6])?,
                    value_bytes: number(&args[7])?,
                    full_verify: full,
                    delay_ms: number(&args[9])?,
                    lite,
                },
            )
            .await?;
            println!(
                "{{\"rows\":{},\"payload_bytes\":{},\"fingerprint\":{},\"full_verify\":{},\"batches\":{},\"first_batch_ms\":{},\"wall_ms\":{},\"sink_ms\":{},\"iops\":{},\"read_bytes\":{}}}",
                report.sink.rows,
                report.sink.payload_bytes,
                report.sink.fingerprint,
                full,
                report.batches,
                report.first_batch_ms,
                report.wall_ms,
                report.sink_ms,
                report.iops,
                report.read_bytes
            );
        }
        _ => {
            return Err(invalid(
                "usage: admitted_fullzip write URI ROWS VALUE_BYTES PAGE_ROWS | read URI ordinary|admitted EXPECTED_ROWS MAX_ROWS WORKSPACE_BYTES VALUE_BYTES rows|full DELAY_MS standard|lite",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_core::utils::tempfile::TempDir;
    use rstest::rstest;

    #[test]
    fn full_sink_rejects_same_length_corruption() {
        let mut lists = ListBuilder::new(LargeBinaryBuilder::new());
        for frame in 0..2 {
            let mut value = (0..8)
                .flat_map(|i| word(0, frame, i).to_le_bytes())
                .collect::<Vec<_>>();
            if frame == 0 {
                value[0] ^= 1;
            }
            lists.values().append_value(value);
        }
        lists.append(true);
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from(vec!["0000000000000000"])),
                Arc::new(lists.finish()),
            ],
        )
        .unwrap();
        Sink::default().consume(&batch, 64, false).unwrap();
        let error = Sink::default().consume(&batch, 64, true).unwrap_err();
        assert!(error.to_string().contains("content mismatch"));
    }

    #[rstest]
    #[tokio::test]
    async fn writer_and_both_readers_validate_all_payloads(
        #[values(64, 8192)] bytes: usize,
        #[values(false, true)] lite: bool,
    ) {
        let tmp = TempDir::default();
        let uri = format!("{}/fixture.lance", tmp.path_str());
        write(&uri, 32, bytes, 8).await.unwrap();
        let options = ReadOptions {
            admitted: false,
            expected_rows: 32,
            max_rows: 8,
            workspace: 512 * 1024,
            value_bytes: bytes,
            full_verify: true,
            delay_ms: 0,
            lite,
        };
        let ordinary = read(&uri, options).await.unwrap();
        let admitted = read(
            &uri,
            ReadOptions {
                admitted: true,
                ..options
            },
        )
        .await
        .unwrap();
        assert_eq!(ordinary.sink.rows, 32);
        assert_eq!(ordinary.sink.rows, admitted.sink.rows);
        assert!(ordinary.sink.payload_bytes > 0);
        assert_eq!(ordinary.sink.payload_bytes, admitted.sink.payload_bytes);
        assert_eq!(ordinary.sink.fingerprint, admitted.sink.fingerprint);
        assert!(
            read(
                &uri,
                ReadOptions {
                    value_bytes: bytes + 8,
                    ..options
                }
            )
            .await
            .is_err()
        );
    }
}
