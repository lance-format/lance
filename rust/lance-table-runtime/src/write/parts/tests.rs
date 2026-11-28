// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::stream;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::object_store::ObjectStoreParams;
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use rstest::rstest;

use super::{DataFilePart, DataFileTarget};
use crate::{InsertBuilder, WriteParams};

#[rstest]
#[case::small_parts(1024 * 1024)]
#[case::large_parts(64)]
#[tokio::test]
async fn opens_parts_concurrently_with_bounded_fanout_and_caller_order(#[case] block_size: usize) {
    let batch = arrow_array::record_batch!(("id", Int32, [0])).unwrap();
    let params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        store_params: Some(ObjectStoreParams {
            block_size: Some(block_size),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&params)
        .execute(vec![batch])
        .await
        .unwrap();
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        ConcreteFileVersion::V2_2,
    )
    .unwrap();
    let mut parts = Vec::new();
    for row_count in (1..=24).rev() {
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::from(dataset.schema())),
            vec![Arc::new(arrow_array::Int32Array::from(vec![0; row_count]))],
        )
        .unwrap();
        let part = dataset
            .write_data_file_part(&target, None, stream::iter([Ok(batch)]))
            .await
            .unwrap();
        parts.push(part);
    }

    assert_eq!(dataset.state.object_store.block_size(), block_size);
    assert!(parts.iter().all(|part| if block_size == 64 {
        part.size_bytes() > block_size as u64
    } else {
        part.size_bytes() <= block_size as u64
    }));
    let active_opens = Arc::new(AtomicUsize::new(0));
    let peak_opens = Arc::new(AtomicUsize::new(0));
    let head_requests = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let completed = Mutex::new(HashSet::new());
    let active = active_opens.clone();
    let mut completed_policy = ProxyObjectStorePolicy::new();
    completed_policy.set_before_policy(
        "complete_part_open_request",
        Arc::new(move |method, path| {
            // This proxy sits after the delay, immediately before the in-memory
            // store completes the first request (GET for small parts, HEAD otherwise).
            if method == "get_opts" && completed.lock().unwrap().insert(path.clone()) {
                active.fetch_sub(1, Ordering::SeqCst);
            }
            Ok(())
        }),
    );
    let heads = head_requests.clone();
    completed_policy.set_obj_meta_policy(
        "observe_part_head",
        Arc::new(move |method, metadata| {
            if method == "head" {
                heads.fetch_add(1, Ordering::SeqCst);
            }
            Ok(metadata)
        }),
    );
    let delayed = Arc::new(ThrottledStore::new(
        Arc::new(ProxyObjectStore::new(
            dataset.state.object_store.inner.clone(),
            Arc::new(Mutex::new(completed_policy)),
        )),
        ThrottleConfig::default(),
    ));
    let first_part = parts[0].file_name.clone();
    let started = Mutex::new(HashSet::new());
    let mut policy = ProxyObjectStorePolicy::new();
    let active = active_opens.clone();
    let peak = peak_opens.clone();
    let request_count = requests.clone();
    let throttle = delayed.clone();
    policy.set_before_policy(
        "observe_part_open_requests",
        Arc::new(move |method, path| {
            if method == "get_opts" {
                request_count.fetch_add(1, Ordering::SeqCst);
                // Force later parts to finish opening before the first input.
                throttle.config_mut(|config| {
                    config.wait_get_per_call = if path.as_ref().ends_with(&first_part) {
                        Duration::from_millis(15)
                    } else {
                        Duration::from_millis(1)
                    };
                });
                // Observe the first size-validation request for each part.
                if started.lock().unwrap().insert(path.clone()) {
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                }
            }
            Ok(())
        }),
    );
    Arc::make_mut(&mut dataset.state.object_store).inner =
        Arc::new(ProxyObjectStore::new(delayed, Arc::new(Mutex::new(policy))));

    let opened = DataFilePart::open_all(&dataset, &target, &parts)
        .await
        .unwrap();
    assert_eq!(
        opened
            .iter()
            .map(|part| part.num_rows())
            .collect::<Vec<_>>(),
        (1..=24).rev().collect::<Vec<_>>()
    );
    assert_eq!(active_opens.load(Ordering::SeqCst), 0);
    if block_size == 64 {
        assert_eq!(head_requests.load(Ordering::SeqCst), parts.len());
    } else {
        assert_eq!(head_requests.load(Ordering::SeqCst), 0);
        assert_eq!(requests.load(Ordering::SeqCst), parts.len());
    }
    let peak = peak_opens.load(Ordering::SeqCst);
    assert!(
        peak > 1 || dataset.state.object_store.io_parallelism() == 1,
        "independent part opens stayed serial"
    );
    assert!(
        peak <= 8,
        "part open fanout exceeded its memory bound: {peak}"
    );
}

#[tokio::test]
async fn target_uses_an_ordinary_generated_data_file_name() {
    let original = arrow_array::record_batch!(("id", Int32, [0, 1])).unwrap();
    let dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        })
        .execute(vec![original])
        .await
        .unwrap();
    let first = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let second = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();

    assert_ne!(first.file_name.as_str(), second.file_name.as_str());
    assert_eq!(first.file_name.len(), 56);
    assert!(first.file_name.ends_with(".lance"));
    assert!(!first.file_name.contains('/'));

    let restored =
        serde_json::from_slice::<DataFileTarget>(&serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(restored.file_name.as_str(), first.file_name.as_str());
    assert_eq!(restored.base_id, first.base_id);
    assert_eq!(restored.schema, first.schema);
    assert_eq!(restored.version, first.version);
}

#[test]
fn target_serialization_preserves_schema_and_rejects_invalid_identity() {
    let arrow_schema = ArrowSchema::new(vec![
        Field::new(
            "nested",
            DataType::Struct(vec![Field::new("value", DataType::Int32, true)].into()),
            true,
        ),
        Field::new(
            "category",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        ),
    ]);
    let mut schema = lance_core::datatypes::Schema::try_from(&arrow_schema).unwrap();
    schema
        .metadata
        .insert("owner".to_string(), "test".to_string());
    schema.fields[0].id = 10;
    schema.fields[0].children[0].id = 11;
    schema.fields[0].children[0].parent_id = 10;
    schema.fields[0].children[0]
        .metadata
        .insert("unit".to_string(), "count".to_string());
    schema.fields[1].id = 20;
    schema.fields[1].dictionary = Some(lance_core::datatypes::Dictionary {
        offset: 12,
        length: 2,
        values: Some(Arc::new(StringArray::from(vec!["a", "b"]))),
    });
    let target = DataFileTarget::new(Some(7), Arc::new(schema), ConcreteFileVersion::V2_2).unwrap();
    let value = serde_json::to_value(&target).unwrap();
    let restored: DataFileTarget = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored.schema, target.schema);
    assert_eq!(restored.file_name.as_str(), target.file_name.as_str());
    assert_eq!(restored.base_id, target.base_id);
    let dictionary = restored.schema.fields[1].dictionary.as_ref().unwrap();
    assert_eq!((dictionary.offset, dictionary.length), (12, 2));
    for (key, replacement, message) in [
        (
            "format_version",
            serde_json::json!(99),
            "checkpoint version",
        ),
        ("file_version", serde_json::json!("stable"), "version"),
        (
            "file_name",
            serde_json::json!("../outside.lance"),
            "target identity",
        ),
        ("file_name", serde_json::json!(""), "target identity"),
        ("file_name", serde_json::json!(".lance"), "target identity"),
        ("file_name", serde_json::json!(".."), "target identity"),
        ("base_id", serde_json::json!(0), "base_id"),
    ] {
        let mut invalid = value.clone();
        invalid[key] = replacement;
        let error = serde_json::from_value::<DataFileTarget>(invalid).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
}
