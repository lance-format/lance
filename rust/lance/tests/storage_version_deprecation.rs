// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The legacy (v1) storage version is deprecated for writing. Creating a dataset on it
//! must warn; every other write must stay quiet.
//!
//! This lives in its own integration binary because it installs a process-global logger.

use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_file::version::LanceFileVersion;

static LOGS: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();

struct Capture;
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if record.level() == log::Level::Warn {
            LOGS.get()
                .unwrap()
                .lock()
                .unwrap()
                .push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap()
}

fn reader() -> RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>
{
    let b = batch();
    let schema = b.schema();
    RecordBatchIterator::new(vec![Ok(b)].into_iter(), schema)
}

#[tokio::test]
async fn legacy_create_warns_append_does_not() {
    LOGS.set(Arc::new(Mutex::new(Vec::new()))).unwrap();
    log::set_boxed_logger(Box::new(Capture)).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();

    Dataset::write(
        reader(),
        uri,
        Some(WriteParams::with_storage_version(LanceFileVersion::Legacy)),
    )
    .await
    .unwrap();

    let after_create = LOGS.get().unwrap().lock().unwrap().clone();
    assert_eq!(
        after_create
            .iter()
            .filter(|m| m.contains("legacy storage version"))
            .count(),
        1,
        "expected exactly one legacy warning on create, got {after_create:?}"
    );
    println!("WARNING TEXT: {}", after_create[0]);

    LOGS.get().unwrap().lock().unwrap().clear();
    Dataset::write(
        reader(),
        uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let after_append = LOGS.get().unwrap().lock().unwrap().clone();
    assert!(
        !after_append
            .iter()
            .any(|m| m.contains("legacy storage version")),
        "append should not warn, got {after_append:?}"
    );

    // A modern create should stay quiet too.
    let dir2 = tempfile::tempdir().unwrap();
    LOGS.get().unwrap().lock().unwrap().clear();
    Dataset::write(reader(), dir2.path().to_str().unwrap(), None)
        .await
        .unwrap();
    let modern = LOGS.get().unwrap().lock().unwrap().clone();
    assert!(
        !modern.iter().any(|m| m.contains("legacy storage version")),
        "modern create should not warn, got {modern:?}"
    );
}
