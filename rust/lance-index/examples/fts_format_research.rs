// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Research-only driver. No production defaults or persistent format changes.

use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

use arrow_array::cast::AsArray;
use arrow_array::{Array, LargeStringArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, stream};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{ROW_ID, cache::LanceCache};
use lance_file::{reader::FileReader, version::ConcreteFileVersion};
use lance_index::scalar::inverted::{
    COMPRESSED_POSITION_COL, InvertedIndex, InvertedIndexBuilder, InvertedIndexParams,
    POSITION_BLOCK_OFFSET_COL, POSTING_COL, PostingListBuilder,
    builder::PositionRecorder,
    document_tokenizer::DocType,
    query::{FtsSearchParams, Operator, Tokens},
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::{Index, metrics::NoOpMetricsCollector, prefilter::NoFilter};
use lance_io::object_store::ObjectStore;
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use object_store::path::Path;
use serde_json::json;

fn text_for_row(row: usize) -> Option<String> {
    if row.is_multiple_of(997) {
        return None;
    }
    if row.is_multiple_of(991) {
        return Some(String::new());
    }
    let mut text = String::from("alpha beta ");
    for _ in 0..row % 5 {
        text.push_str("alpha ");
    }
    for pos in 0..32 {
        let term = (row.wrapping_mul(104729) ^ (pos * 13007)) % 65536;
        text.push_str(&format!("term{term:05} "));
    }
    if row.is_multiple_of(10007) {
        text.push_str("needle alpha beta ");
    }
    Some(text)
}

fn process_snapshot() -> serde_json::Value {
    json!({"status": fs::read_to_string("/proc/self/status").ok(),
           "io": fs::read_to_string("/proc/self/io").ok()})
}

fn emit(value: serde_json::Value) {
    println!("{value}");
}

fn export_blocks(root: &std::path::Path, rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(root)?;
    let started = Instant::now();
    let mut builder = PostingListBuilder::new(true);
    for doc in 0..rows {
        let frequency = 1 + ((doc.wrapping_mul(1664525) >> 7) & 7);
        builder.add(
            doc as u32,
            PositionRecorder::Position(
                (0..frequency)
                    .map(|p| (p * 3 + doc % 3) as u32)
                    .collect::<Vec<_>>()
                    .into(),
            ),
        );
    }
    let batch = builder.to_batch(vec![1.0; rows.div_ceil(128)])?;
    let array = batch[POSTING_COL].as_list::<i32>().value(0);
    let blocks = array.as_binary::<i64>();
    let mut data = fs::File::create(root.join("posting.blocks"))?;
    let mut offsets = vec![0u64];
    for block in blocks.iter() {
        let block = block.ok_or("null posting block")?;
        data.write_all(block)?;
        offsets.push(offsets.last().copied().unwrap() + block.len() as u64);
    }
    data.sync_all()?;
    fs::write(
        root.join("posting.offsets.json"),
        serde_json::to_vec(&offsets)?,
    )?;
    let positions = batch[COMPRESSED_POSITION_COL].as_binary::<i64>().value(0);
    fs::write(root.join("positions.blocks"), positions)?;
    let pos_offsets = batch[POSITION_BLOCK_OFFSET_COL].as_list::<i32>().value(0);
    let mut pos_offsets: Vec<u64> = pos_offsets
        .as_primitive::<arrow_array::types::UInt32Type>()
        .values()
        .iter()
        .map(|x| u64::from(*x))
        .collect();
    if pos_offsets.last().copied() != Some(positions.len() as u64) {
        pos_offsets.push(positions.len() as u64);
    }
    fs::write(
        root.join("positions.offsets.json"),
        serde_json::to_vec(&pos_offsets)?,
    )?;
    emit(json!({"phase":"export", "rows":rows, "blocks":blocks.len(),
        "posting_bytes":offsets.last(), "position_bytes":positions.len(),
        "posting_offset_bytes":offsets.len()*8, "position_offset_bytes":pos_offsets.len()*8,
        "elapsed_s":started.elapsed().as_secs_f64(), "process":process_snapshot()}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_index::scalar::inverted::PostingList;
    use std::collections::BTreeMap;

    #[test]
    fn exported_codec_matches_independent_inputs() {
        for rows in [1usize, 127, 128, 129, 256, 257] {
            let mut builder = PostingListBuilder::new(true);
            for row in 0..rows {
                builder.add(
                    (row * 7) as u32,
                    PositionRecorder::Position(vec![row as u32 % 3, row as u32 % 3 + 4].into()),
                );
            }
            let batch = builder.to_batch(vec![1.0; rows.div_ceil(128)]).unwrap();
            let posting = PostingList::from_batch(&batch, Some(1.0), Some(rows as u32)).unwrap();
            let actual: Vec<_> = posting
                .iter()
                .map(|(doc, freq, pos)| (doc, freq as u32, pos.unwrap().collect::<Vec<_>>()))
                .collect();
            let expected: Vec<_> = (0..rows)
                .map(|row| {
                    (
                        (row * 7) as u64,
                        2,
                        vec![row as u32 % 3, row as u32 % 3 + 4],
                    )
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[tokio::test]
    async fn native_query_matches_raw_text_reference() {
        for container in [ConcreteFileVersion::V2_0, ConcreteFileVersion::V2_2] {
            assert_native_contract(container).await;
        }
    }

    async fn assert_native_contract(container: ConcreteFileVersion) {
        let texts = [
            None,
            Some(""),
            Some("alpha beta"),
            Some("alpha alpha beta"),
            Some("beta alpha"),
            Some("needle alpha beta"),
            Some("alpha gap beta"),
            Some("gamma"),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::LargeUtf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(LargeStringArray::from(texts.to_vec())),
                    Arc::new(UInt64Array::from(
                        (0..texts.len() as u64).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap(),
        ];
        let temporary = tempfile::tempdir().unwrap();
        let store = Arc::new(LanceIndexStore::with_format_version(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(temporary.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
            container,
        ));
        let mut builder = InvertedIndexBuilder::new(
            InvertedIndexParams::default()
                .with_position(true)
                .stem(false)
                .remove_stop_words(false)
                .num_workers(1),
        );
        builder
            .update(
                Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::iter(batches.into_iter().map(Ok)),
                )),
                store.as_ref(),
                None,
            )
            .await
            .unwrap();
        let index = InvertedIndex::load(store, None, &LanceCache::with_capacity(1024 * 1024))
            .await
            .unwrap();
        let docs: Vec<Vec<&str>> = texts
            .iter()
            .map(|text| text.unwrap_or("").split_whitespace().collect())
            .collect();
        let n = docs.iter().filter(|doc| !doc.is_empty()).count() as f64;
        let avgdl = docs.iter().map(Vec::len).sum::<usize>() as f64 / n;
        for (query, and, phrase) in [
            ("alpha", false, false),
            ("alpha gamma", false, false),
            ("alpha needle", true, false),
            ("missing", false, false),
            ("alpha beta", true, true),
            ("alpha alpha", true, true),
            ("needle alpha beta", true, true),
        ] {
            let terms: Vec<_> = query.split_whitespace().collect();
            let mut expected = BTreeMap::new();
            for (row, doc) in docs.iter().enumerate() {
                let matched = if phrase {
                    doc.windows(terms.len())
                        .any(|window| window == terms.as_slice())
                } else if and {
                    terms.iter().all(|term| doc.contains(term))
                } else {
                    terms.iter().any(|term| doc.contains(term))
                };
                if !matched {
                    continue;
                }
                let mut score = 0.0f64;
                for term in &terms {
                    let tf = doc.iter().filter(|token| *token == term).count() as f64;
                    let df = docs.iter().filter(|tokens| tokens.contains(term)).count() as f64;
                    if tf > 0.0 {
                        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                        score +=
                            idf * 2.2 * tf / (tf + 1.2 * (0.25 + 0.75 * doc.len() as f64 / avgdl));
                    }
                }
                expected.insert(row as u64, score);
            }
            for _ in 0..2 {
                let mut params = FtsSearchParams::new().with_limit(None);
                if phrase {
                    params = params.with_phrase_slop(Some(0));
                }
                let (ids, scores) = index
                    .bm25_search(
                        Arc::new(Tokens::new(
                            terms.iter().map(|s| s.to_string()).collect(),
                            DocType::Text,
                        )),
                        params.into(),
                        if and { Operator::And } else { Operator::Or },
                        Arc::new(NoFilter),
                        Arc::new(NoOpMetricsCollector),
                        None,
                    )
                    .await
                    .unwrap();
                assert_eq!(ids.len(), expected.len(), "{query}");
                let actual: BTreeMap<_, _> = ids.into_iter().zip(scores).collect();
                assert_eq!(actual.len(), expected.len(), "duplicate IDs: {query}");
                for (id, score) in actual {
                    let reference = expected.get(&id).unwrap();
                    assert!(score.is_finite() && score > 0.0);
                    if !phrase {
                        assert!(
                            (f64::from(score) - reference).abs() < 1e-5,
                            "{query} row {id}: {score} vs {reference}"
                        );
                    }
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .get(1)
        .ok_or("mode: build/build-ipc/query/export/inspect/stats")?;
    let root = PathBuf::from(args.get(2).ok_or("index directory")?);
    let rows: usize = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("200000")
        .parse()?;
    if mode == "export" {
        return export_blocks(&root, rows);
    }
    let object_store = Arc::new(ObjectStore::local());
    let metadata_cache = Arc::new(LanceCache::with_capacity(64 * 1024 * 1024));
    let index_cache = LanceCache::with_capacity(64 * 1024 * 1024);
    fs::create_dir_all(&root)?;
    let container = match std::env::var("FTS_RESEARCH_CONTAINER_VERSION")
        .unwrap_or_else(|_| "2.2".to_owned())
        .as_str()
    {
        "2.0" => ConcreteFileVersion::V2_0,
        "2.1" => ConcreteFileVersion::V2_1,
        "2.2" => ConcreteFileVersion::V2_2,
        _ => return Err("research container must be 2.0, 2.1 or 2.2".into()),
    };
    if mode == "inspect" {
        let output = PathBuf::from(args.get(4).ok_or("inspect output directory")?);
        fs::create_dir_all(&output)?;
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let mut files: Vec<_> = fs::read_dir(&root)?.collect::<std::io::Result<Vec<_>>>()?;
        files.sort_by_key(|entry| entry.file_name());
        for entry in files {
            if entry.path().extension().is_none_or(|ext| ext != "lance") {
                continue;
            }
            let path = Path::from_filesystem_path(fs::canonicalize(entry.path())?)?;
            let file = scheduler
                .open_file_with_priority(&path, 0, &CachedFileSize::unknown())
                .await?;
            let metadata = FileReader::read_all_metadata(&file).await?;
            let name = entry.file_name().to_string_lossy().into_owned();
            fs::write(
                output.join(format!("{name}.metadata.txt")),
                format!("{metadata:#?}"),
            )?;
            emit(
                json!({"phase":"file_metadata","file":name,"version":format!("{:?}",metadata.version),
                "rows":metadata.num_rows,"file_bytes":metadata.file_size_bytes,
                "data_bytes":metadata.num_data_bytes,"column_metadata_bytes":metadata.num_column_metadata_bytes,
                "global_buffer_bytes":metadata.num_global_buffer_bytes,"footer_bytes":metadata.num_footer_bytes,
                "heap_estimate_bytes":metadata.deep_size_of()}),
            );
        }
        return Ok(());
    }
    let store = Arc::new(LanceIndexStore::with_format_version(
        object_store.clone(),
        Path::from_filesystem_path(fs::canonicalize(&root)?)?,
        metadata_cache.clone(),
        container,
    ));
    if mode == "build" || mode == "build-ipc" {
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::LargeUtf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let batch_schema = schema.clone();
        let synthetic_batches = (0..rows.div_ceil(8192)).map(move |batch| {
            let start = batch * 8192;
            let end = (start + 8192).min(rows);
            RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(LargeStringArray::from(
                        (start..end).map(text_for_row).collect::<Vec<_>>(),
                    )),
                    Arc::new(UInt64Array::from(
                        (start..end).map(|x| x as u64).collect::<Vec<_>>(),
                    )),
                ],
            )
            .map_err(Into::into)
        });
        let batches: Box<dyn Iterator<Item = lance_core::Result<RecordBatch>> + Send> =
            if mode == "build-ipc" {
                let path = args.get(4).ok_or("build-ipc requires an Arrow IPC file")?;
                let reader = arrow_ipc::reader::FileReader::try_new(fs::File::open(path)?, None)?;
                if reader.schema().fields() != schema.fields() {
                    return Err("IPC input must contain text: LargeUtf8 and _rowid: UInt64".into());
                }
                Box::new(reader.map(|batch| batch.map_err(Into::into)))
            } else {
                Box::new(synthetic_batches)
            };
        let params = InvertedIndexParams::default()
            .with_position(true)
            .stem(false)
            .remove_stop_words(false)
            .num_workers(2)
            .memory_limit_mb(512);
        let mut builder = InvertedIndexBuilder::new(params);
        let start = Instant::now();
        builder
            .update(
                Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches))),
                store.as_ref(),
                None,
            )
            .await?;
        let io = object_store.io_stats_incremental();
        emit(
            json!({"phase":"build", "rows":rows, "container":format!("{container:?}"), "elapsed_s":start.elapsed().as_secs_f64(),
            "read_bytes":io.read_bytes, "write_bytes":io.written_bytes, "process":process_snapshot()}),
        );
        return Ok(());
    }
    if mode != "query" && mode != "stats" {
        return Err("unknown mode".into());
    }
    let start = Instant::now();
    let index = Arc::new(InvertedIndex::load(store, None, &index_cache).await?);
    let io = object_store.io_stats_incremental();
    emit(
        json!({"phase":"open", "elapsed_s":start.elapsed().as_secs_f64(),
        "partitions":index.partition_count(), "read_bytes":io.read_bytes,"read_iops":io.read_iops,
        "index_cache_bytes":index_cache.size_bytes().await,
        "metadata_cache_bytes":metadata_cache.size_bytes().await,"index_heap_estimate_bytes":index.deep_size_of(),"process":process_snapshot()}),
    );
    if mode == "stats" {
        let path = args.get(4).ok_or("stats query JSON file")?;
        let queries: Vec<(String, bool, bool)> = serde_json::from_slice(&fs::read(path)?)?;
        let terms: Vec<_> = queries
            .iter()
            .flat_map(|(q, _, _)| q.split_whitespace().map(str::to_owned))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let start = Instant::now();
        let (total_tokens, documents, df) = index.bm25_stats_for_terms(&terms, None).await?;
        let io = object_store.io_stats_incremental();
        emit(
            json!({"phase":"stats","elapsed_s":start.elapsed().as_secs_f64(),"terms":terms,
            "df":df,"documents":documents,"total_tokens":total_tokens,
            "read_bytes":io.read_bytes,"read_iops":io.read_iops,
            "index_heap_estimate_bytes":index.deep_size_of(),"index_cache_bytes":index_cache.size_bytes().await,
            "metadata_cache_bytes":metadata_cache.size_bytes().await,"process":process_snapshot()}),
        );
        return Ok(());
    }
    if args.get(6).is_some_and(|s| s == "prewarm") {
        let start = Instant::now();
        index.prewarm().await?;
        let io = object_store.io_stats_incremental();
        emit(
            json!({"phase":"prewarm", "elapsed_s":start.elapsed().as_secs_f64(),
            "read_bytes":io.read_bytes, "read_iops":io.read_iops,
            "index_cache_bytes":index_cache.size_bytes().await,
            "metadata_cache_bytes":metadata_cache.size_bytes().await,"process":process_snapshot()}),
        );
    }
    let concurrency: usize = args.get(4).map(String::as_str).unwrap_or("1").parse()?;
    let sweeps: usize = args.get(5).map(String::as_str).unwrap_or("10").parse()?;
    let default_queries = [
        ("missing", false, false),
        ("needle", false, false),
        ("alpha", false, false),
        ("alpha term00007", false, false),
        ("alpha term00007", true, false),
        ("alpha beta", true, true),
        ("needle alpha beta", true, true),
        ("term00007 term10000 term20000", false, false),
    ];
    let queries: Vec<(String, bool, bool)> = if let Some(path) = args.get(7) {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        default_queries
            .into_iter()
            .map(|(q, and, phrase)| (q.to_owned(), and, phrase))
            .collect()
    };
    for sweep in 0..sweeps {
        let start = Instant::now();
        let index_ref = &index;
        let results: Vec<_> = stream::iter(queries.iter().enumerate())
            .map(|(query_id, (text, and, phrase))| async move {
                let start = Instant::now();
                let tokens = Arc::new(Tokens::new(
                    text.split_whitespace().map(str::to_owned).collect(),
                    DocType::Text,
                ));
                let mut params = FtsSearchParams::new().with_limit(Some(10));
                if *phrase {
                    params = params.with_phrase_slop(Some(0));
                }
                let (ids, scores) = index_ref
                    .bm25_search(
                        tokens,
                        params.into(),
                        if *and { Operator::And } else { Operator::Or },
                        Arc::new(NoFilter),
                        Arc::new(NoOpMetricsCollector),
                        None,
                    )
                    .await?;
                Ok::<_, lance_core::Error>(
                    json!({"query_id":query_id, "query":text, "and":and,"phrase":phrase,
                "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"ids":ids,"scores":scores}),
                )
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;
        let mut output = Vec::new();
        for result in results {
            output.push(result?);
        }
        let io = object_store.io_stats_incremental();
        emit(
            json!({"phase":"query_sweep","sweep":sweep,"concurrency":concurrency,
            "elapsed_s":start.elapsed().as_secs_f64(),"queries":output,
            "read_bytes":io.read_bytes,"read_iops":io.read_iops,
            "index_cache_bytes":index_cache.size_bytes().await,
            "metadata_cache_bytes":metadata_cache.size_bytes().await,"process":process_snapshot()}),
        );
    }
    Ok(())
}
