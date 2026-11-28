// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JSON target-type recovery, query-domain and scan-fallback tests.

use super::json_filter_ids;
use super::*;
use crate::index::CreateIndexBuilder;

/// [`json_batch`] plus an `id` column that identifies a row across
/// `max_rows_per_file = 1`, so tests can compare ids instead of whole rows.
fn json_id_batch(values: Vec<&str>) -> RecordBatch {
    let mut metadata = HashMap::new();
    metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::UInt64, false),
        ArrowField::new("json", DataType::Utf8, false).with_metadata(metadata),
    ]));
    let ids = UInt64Array::from_iter_values(0..values.len() as u64);
    RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(StringArray::from(values))],
    )
    .unwrap()
}

/// Rewrite every segment of a JSON index to the details a writer that predates
/// `target_data_type` left behind: `Unspecified` and nothing else changed, so
/// the target index files still hold the type it was trained on.
pub(super) async fn forget_json_index_target_type(dataset: &mut Dataset, name: &str) {
    use lance_index::pb::{JsonIndexDetails, JsonTargetDataType};
    use lance_table::format::IndexMetadata;
    use prost::Message;

    let forgotten = dataset
        .load_indices_by_name(name)
        .await
        .unwrap()
        .into_iter()
        .map(|segment| {
            let details =
                JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice())
                    .unwrap();
            assert_ne!(
                details.target_data_type,
                JsonTargetDataType::Unspecified as i32,
                "the current writer must record the type this helper then removes"
            );
            let details = JsonIndexDetails {
                target_data_type: JsonTargetDataType::Unspecified as i32,
                ..details
            };
            IndexMetadata {
                index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
                ..segment
            }
        })
        .collect::<Vec<_>>();
    dataset
        .commit_existing_index_segments(name, "json", forgotten)
        .await
        .unwrap();
}

/// One JSON index over path `p` whose details record `target_data_type` while
/// its target index files stay exactly as the writer built them.
///
/// `Unspecified` reproduces an index written before the field existed, and any
/// value this build does not know reproduces one written by a later build. Both
/// are indistinguishable in the details alone, so they are worth testing
/// separately.
async fn json_index_with_recorded_type(
    test_uri: &str,
    data_storage_version: LanceFileVersion,
    target_index_type: &str,
    target_data_type: i32,
) -> Dataset {
    json_index_with_recorded_docs(
        test_uri,
        data_storage_version,
        target_index_type,
        target_data_type,
        // `json_get_int` makes "many" null, `json_get_string` does not.
        &[r#"{"p":"many"}"#, r#"{"p":null}"#, r#"{}"#, r#"{"p":"7"}"#],
        1,
    )
    .await
}

/// [`json_index_with_recorded_type`] over caller-chosen rows and fragment
/// layout, so a test can give a target one zone that holds both matching and
/// non-matching rows.
async fn json_index_with_recorded_docs(
    test_uri: &str,
    data_storage_version: LanceFileVersion,
    target_index_type: &str,
    target_data_type: i32,
    docs: &[&str],
    rows_per_file: usize,
) -> Dataset {
    use lance_index::pb::JsonIndexDetails;
    use lance_table::format::IndexMetadata;
    use prost::Message;

    let mut metadata = HashMap::new();
    metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::UInt64, false),
        ArrowField::new("json", DataType::Utf8, false).with_metadata(metadata),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..docs.len() as u64)),
            Arc::new(StringArray::from(docs.to_vec())),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: rows_per_file,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::new("json".to_string()).with_params(&serde_json::json!({
        "target_index_type": target_index_type,
        "target_data_type": "Utf8",
        "path": "p",
    }));
    let segment = dataset
        .create_index_builder(&["json"], IndexType::Scalar, &params)
        .name("json_p_idx".to_string())
        .execute_uncommitted()
        .await
        .unwrap();

    let mut details =
        JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice()).unwrap();
    details.target_data_type = target_data_type;
    let rewritten = IndexMetadata {
        index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
        ..segment
    };
    dataset
        .commit_existing_index_segments("json_p_idx", "json", vec![rewritten])
        .await
        .unwrap();
    dataset
}

/// A JSON index whose details omit the target type still answers the accessors
/// its target was trained for, because the target index's own files record that
/// type. The strict accessor/target type match is unchanged: the type that
/// routing compares is the recovered one, not the query literal's.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_unrecorded_btree_target_type_recovers(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;

    let test_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &test_uri,
        data_storage_version,
        "btree",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;

    for (filter, expected_ids, used_index) in [
        // The recovered Utf8 type is exactly what this accessor reads, so the
        // index answers it again.
        ("json_get_string(json, 'p') IS NULL", vec![1, 2], true),
        // Answering from the index's Utf8 keys would drop every row whose value
        // the accessor cannot read as an integer: the recovered type declines
        // this accessor just like a recorded Utf8 type would.
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2], false),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        if used_index {
            assert!(
                plan.contains("json_p_idx"),
                "a recoverable target type must be routed: {plan}"
            );
        } else {
            assert!(
                !plan.contains("json_p_idx"),
                "a mismatched accessor must not be routed to a recovered Utf8 target: {plan}"
            );
        }
    }
}

/// A genuine 0.36 index is the oldest layout the compatibility matrix still
/// creates, and it stores its key type only in the lookup file's schema. Reading
/// that schema is enough to answer a wrapper whose details omit the type, and it
/// must not read the index's page data.
#[tokio::test]
async fn test_legacy_btree_index_reports_key_type_from_metadata() {
    use crate::dataset::index::LanceIndexStoreExt;
    use crate::index::scalar::IndexDetails;
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_io::assert_io_eq;

    let test_dir = copy_test_data_to_tmp("0.36.0/btree_in_index_pkg.lance").unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let index = indices.last().unwrap().clone();
    let details = index.index_details.clone().unwrap();
    let index_details = IndexDetails(details.clone());
    let plugin = index_details.get_plugin().unwrap();
    assert_eq!(plugin.name(), "BTree");

    let io = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();
    let store = Arc::new(
        LanceIndexStore::from_dataset_for_existing(&dataset, &index)
            .await
            .unwrap(),
    );
    io.incremental_stats();
    let recovered = plugin
        .recover_query_domain("x_idx".to_string(), &details, 0, store)
        .await
        .unwrap();
    let stats = io.incremental_stats();

    let (_, domain) = recovered.expect("a 0.36 BTree lookup records its key type");
    assert_eq!(domain.data_type, DataType::Int64);
    // This lookup file is under one kilobyte, so its footer and its rows share a
    // block and a byte assertion would not separate them. What must not happen is
    // a read of the index's page data or a load of the index itself.
    assert!(
        stats
            .requests
            .iter()
            .all(|request| !request.path.as_ref().ends_with("page_data.lance")),
        "reading the key type must not touch page data: {:?}",
        stats.requests
    );
    // A 0.36 index records no file size, so opening it probes the size with a
    // HEAD before reading. The read itself covers the whole file only because
    // all 991 bytes fit in one block; that is a property of this fixture, not of
    // recovery, which asks for the schema and never for a data range.
    assert_io_eq!(
        stats,
        read_iops,
        2,
        "one size probe and one lookup file read, nothing else: {:?}",
        stats.requests
    );
    assert_eq!(
        stats
            .requests
            .iter()
            .filter(|request| request.method == "head")
            .count(),
        1,
        "an index with no recorded file size probes it once: {:?}",
        stats.requests
    );
}

/// How one fixture builds the segment covering one fragment.
struct JsonSegmentSpec<'a> {
    target_index_type: &'a str,
    target_data_type: &'a str,
    path: &'a str,
    target_index_parameters: Option<serde_json::Value>,
    recorded_target_data_type: Option<i32>,
}

impl<'a> JsonSegmentSpec<'a> {
    fn new(target_index_type: &'a str, target_data_type: &'a str, path: &'a str) -> Self {
        Self {
            target_index_type,
            target_data_type,
            path,
            target_index_parameters: None,
            recorded_target_data_type: None,
        }
    }

    fn with_target_index_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.target_index_parameters = Some(parameters);
        self
    }

    /// Record `recorded` in the wrapper details instead of the type the writer
    /// derived, leaving the target's files untouched.
    fn with_recorded_target_data_type(mut self, recorded: i32) -> Self {
        self.recorded_target_data_type = Some(recorded);
        self
    }
}

/// One JSON index segment per fragment, so a name has several segments whose
/// coverage does not overlap. Execution opens all of them for a planned search,
/// which is why planning decides per name.
///
/// Each spec controls its segment's target, path and recorded details, so a
/// test can build the mixes `commit_existing_index_segments` still accepts.
async fn json_index_segment_per_fragment(
    dataset: &mut Dataset,
    name: &str,
    specs: &[JsonSegmentSpec<'_>],
) -> Vec<lance_table::format::IndexMetadata> {
    use lance_index::pb::JsonIndexDetails;
    use lance_table::format::IndexMetadata;
    use prost::Message;

    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    assert_eq!(fragment_ids.len(), specs.len());
    let mut segments = Vec::new();
    for (fragment_id, spec) in fragment_ids.iter().zip(specs) {
        let mut params = serde_json::json!({
            "target_index_type": spec.target_index_type,
            "target_data_type": spec.target_data_type,
            "path": spec.path,
        });
        if let Some(target_index_parameters) = &spec.target_index_parameters {
            params["target_index_parameters"] =
                serde_json::Value::String(serde_json::to_string(target_index_parameters).unwrap());
        }
        let params = ScalarIndexParams::new("json".to_string()).with_params(&params);
        let segment = CreateIndexBuilder::new(dataset, &["json"], IndexType::Scalar, &params)
            .name(name.to_string())
            .fragments(vec![*fragment_id])
            .execute_uncommitted()
            .await
            .unwrap();
        let Some(recorded) = spec.recorded_target_data_type else {
            segments.push(segment);
            continue;
        };
        let details =
            JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice())
                .unwrap();
        let details = JsonIndexDetails {
            target_data_type: recorded,
            ..details
        };
        segments.push(IndexMetadata {
            index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
            ..segment
        });
    }
    segments
}

/// Recovering a type is an optimization, so a target whose files cannot be read
/// must not fail the query. Planning keeps working and the predicate is left to
/// the scan, which is how this index behaved before recovery existed.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_unreadable_target_type_falls_back_to_scan(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;
    use object_store::ObjectStoreExt;

    let test_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &test_uri,
        data_storage_version,
        "btree",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;
    let index = dataset
        .load_indices_by_name("json_p_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let lookup_path = dataset
        .indice_files_dir(&index)
        .unwrap()
        .join(index.uuid.to_string())
        .join("page_lookup.lance");
    dataset
        .table_state
        .object_store
        .inner
        .delete(&lookup_path)
        .await
        .unwrap();

    // A fresh session, so nothing cached from the build hides the missing file.
    let dataset = Dataset::open(test_uri.as_str()).await.unwrap();
    for (filter, expected_ids) in [
        ("json_get_string(json, 'p') IS NULL", vec![1, 2]),
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_p_idx"),
            "an unreadable target must leave the predicate to the scan: {plan}"
        );
    }
}

/// A planned search opens every segment registered under the index name it
/// reports, so a name is only routable when all of its segments hold the same
/// value domain. That has to hold across recorded and recovered types, and a
/// name whose segments disagree has to be declined as a whole rather than
/// segment by segment.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_multi_segment_routing_requires_one_domain(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;

    // Agreeing segments: two recorded Int64, two recovered from the target.
    let test_uri = TempStrDir::default();
    let batch = json_id_batch(vec![
        r#"{"v": 0}"#,
        r#"{"v": 1}"#,
        r#"{"v": 2}"#,
        r#"{"v": 3}"#,
    ]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let unspecified = JsonTargetDataType::Unspecified as i32;
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_v_idx",
        &[
            JsonSegmentSpec::new("btree", "Int64", "v"),
            JsonSegmentSpec::new("btree", "Int64", "v").with_recorded_target_data_type(unspecified),
            JsonSegmentSpec::new("btree", "Int64", "v").with_recorded_target_data_type(unspecified),
            JsonSegmentSpec::new("btree", "Int64", "v"),
        ],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_v_idx", "json", segments)
        .await
        .unwrap();
    assert_eq!(
        dataset
            .load_indices_by_name("json_v_idx")
            .await
            .unwrap()
            .len(),
        4,
        "the fixture must exercise several segments under one name"
    );

    // Rows 1 and 2 come from recovered segments and row 3 from a recorded one,
    // so answering all three requires both to have established the same domain.
    for (filter, expected_ids) in [
        ("json_get_int(json, 'v') >= 1", vec![1, 2, 3]),
        ("json_get_int(json, 'v') IS NULL", Vec::new()),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            plan.contains("json_v_idx"),
            "segments that agree on the value domain must be routed: {plan}"
        );
    }

    // Disagreeing segments: the writer accepts one Int64 and one Utf8 target
    // under one name, and each is valid for its own fragment. A search would
    // open both, so neither may answer.
    let conflict_uri = TempStrDir::default();
    let batch = json_id_batch(vec![r#"{"v": 1}"#, r#"{"v": "one"}"#]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &conflict_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_v_idx",
        &[
            JsonSegmentSpec::new("btree", "Int64", "v"),
            JsonSegmentSpec::new("btree", "Utf8", "v"),
        ],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_v_idx", "json", segments)
        .await
        .unwrap();

    for (filter, expected_ids) in [
        // "one" is unreadable as an integer, so this row is the only null one.
        ("json_get_int(json, 'v') IS NULL", vec![1]),
        // The accessor renders `1` as text, so it matches the integer row.
        ("json_get_string(json, 'v') = '1'", vec![0]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_v_idx"),
            "a name whose segments hold different domains must be declined: {plan}"
        );
    }
}

/// A planned search opens every segment registered under the name it reports,
/// so a name whose segments extract the same value type from different JSON
/// paths cannot answer one query: the predicate planned from one segment's path
/// is handed to the others' keys. That holds whether the conflicting segment
/// recorded its type or the planner recovered it from the target.
#[rstest]
#[case::legacy_recorded(LanceFileVersion::Legacy, false, false)]
#[case::legacy_mixed(LanceFileVersion::Legacy, true, false)]
#[case::stable_recorded(LanceFileVersion::Stable, false, false)]
#[case::stable_mixed(LanceFileVersion::Stable, true, false)]
#[case::stable_row_ids_mixed(LanceFileVersion::Stable, true, true)]
#[tokio::test]
async fn test_json_multi_segment_same_type_different_path_is_not_routed(
    #[case] data_storage_version: LanceFileVersion,
    #[case] recover_second_path: bool,
    #[case] enable_stable_row_ids: bool,
) {
    use lance_index::pb::JsonIndexDetails;
    use lance_index::pb::JsonTargetDataType;
    use prost::Message;

    let test_uri = TempStrDir::default();
    // Two fragments of two rows. Segment 0 indexes `a` over fragment 0 and
    // segment 1 indexes `b` over fragment 1, so a query on `a` that reaches
    // segment 1 answers from the wrong keys (and misses rows in fragment 1).
    let batch = json_id_batch(vec![
        r#"{"a": 1, "b": 10}"#,
        r#"{"a": 2, "b": 1}"#,
        r#"{"a": 3}"#,
        r#"{"b": 1}"#,
    ]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 2,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let second = JsonSegmentSpec::new("btree", "Int64", "b");
    let second = if recover_second_path {
        second.with_recorded_target_data_type(JsonTargetDataType::Unspecified as i32)
    } else {
        second
    };
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_mixed_path_idx",
        &[JsonSegmentSpec::new("btree", "Int64", "a"), second],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_mixed_path_idx", "json", segments)
        .await
        .unwrap();

    let committed = dataset
        .load_indices_by_name("json_mixed_path_idx")
        .await
        .unwrap();
    assert_eq!(committed.len(), 2);
    let paths = committed
        .iter()
        .map(|segment| {
            JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice())
                .unwrap()
                .path
        })
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["a", "b"]);

    // Rows 1 and 3 are the ones a name-wide search gets wrong: `a = 1` matches
    // `b = 1` in segment 1's fragment, `a = 3` has no key in either segment's
    // first fragment, and the AND/OR cases combine the two bad answers.
    for (filter, expected_ids) in [
        ("json_get_int(json, 'a') = 1", vec![0]),
        ("json_get_int(json, 'a') = 3", vec![2]),
        ("json_get_int(json, 'a') IS NULL", vec![3]),
        (
            "json_get_int(json, 'a') = 1 AND json_get_int(json, 'b') = 1",
            vec![],
        ),
        (
            "json_get_int(json, 'a') = 1 OR json_get_int(json, 'b') = 40",
            vec![0],
        ),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_mixed_path_idx"),
            "segments that extract different paths must not answer together: {plan}"
        );
    }
}

/// One value type is not enough when the segments were built by different
/// targets either: an exact target's planner can answer a predicate the other
/// target only narrows to a zone, and both policies are baked into the query
/// before execution opens every segment.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_multi_segment_requires_one_target_kind_and_config(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonIndexDetails;
    use prost::Message;

    // Different target kinds with the same Utf8 domain: a B-tree over fragment
    // 0 and a zone map over fragment 1. The planned equality needs no recheck
    // from the B-tree, while the zone map returns all of its zone, so routing
    // both would add row 3.
    let test_uri = TempStrDir::default();
    let batch = json_id_batch(vec![
        r#"{"v": "needle"}"#,
        r#"{"v": "hay"}"#,
        r#"{"v": "needle"}"#,
        r#"{"v": "hay"}"#,
    ]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let unspecified = lance_index::pb::JsonTargetDataType::Unspecified as i32;
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_v_idx",
        &[
            JsonSegmentSpec::new("btree", "Utf8", "v").with_recorded_target_data_type(unspecified),
            JsonSegmentSpec::new("zonemap", "Utf8", "v")
                .with_recorded_target_data_type(unspecified),
        ],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_v_idx", "json", segments)
        .await
        .unwrap();
    let committed = dataset.load_indices_by_name("json_v_idx").await.unwrap();
    let target_types = committed
        .iter()
        .map(|segment| {
            JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice())
                .unwrap()
                .target_details
                .unwrap()
                .type_url
        })
        .collect::<Vec<_>>();
    assert_eq!(target_types.len(), 2);
    assert_ne!(target_types[0], target_types[1]);
    for (filter, expected_ids) in [
        ("json_get_string(json, 'v') = 'needle'", vec![0, 2]),
        ("json_get_string(json, 'v') IS NULL", vec![]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_v_idx"),
            "segments built by different targets must not answer together: {plan}"
        );
    }
}

/// The same target kind with one configuration difference is not one logical
/// query domain either. Here fragment 0's zone map keeps the exact null bitmap
/// and fragment 1's does not: a query planned from fragment 0 is exact for
/// `IS NULL`, while fragment 1 can only return its whole zone.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_multi_segment_requires_one_target_config(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonIndexDetails;
    use lance_index::pb::JsonTargetDataType;
    use prost::Message;

    let unspecified = JsonTargetDataType::Unspecified as i32;
    let config_uri = TempStrDir::default();
    let batch = json_id_batch(vec![
        r#"{"v": null}"#,
        r#"{"v": "a"}"#,
        r#"{"v": null}"#,
        r#"{"v": "b"}"#,
    ]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &config_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_v_idx",
        &[
            JsonSegmentSpec::new("zonemap", "Utf8", "v")
                .with_recorded_target_data_type(unspecified),
            JsonSegmentSpec::new("zonemap", "Utf8", "v")
                .with_recorded_target_data_type(unspecified)
                .with_target_index_parameters(serde_json::json!({"store_null_bitmap": false})),
        ],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_v_idx", "json", segments)
        .await
        .unwrap();
    let null_policies = dataset
        .load_indices_by_name("json_v_idx")
        .await
        .unwrap()
        .iter()
        .map(|segment| {
            let details =
                JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice())
                    .unwrap();
            details
                .target_details
                .unwrap()
                .to_msg::<lance_index::pbold::ZoneMapIndexDetails>()
                .unwrap()
                .has_null_bitmap
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        null_policies,
        vec![true, false],
        "the fixture needs one exact and one approximate null policy"
    );

    let filter = "json_get_string(json, 'v') IS NULL";
    let indexed = json_filter_ids(&dataset, filter, true).await;
    assert_eq!(indexed, vec![0, 2], "indexed scan: {filter}");
    let full = json_filter_ids(&dataset, filter, false).await;
    assert_eq!(full, vec![0, 2], "full-scan oracle: {filter}");
    let plan = dataset
        .scan()
        .filter(filter)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(
        !plan.contains("json_v_idx"),
        "segments with different null-tracking policies must not answer together: {plan}"
    );
}

/// A zone map persists the type of its min/max columns, so a JSON wrapper that
/// omitted the target type can still recover it from the index file itself.
/// The target stays inexact: the index can only narrow a predicate to a zone,
/// and the original predicate has to recheck the rows in it.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_unrecorded_zonemap_target_type_recovers_and_rechecks(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;

    let test_uri = TempStrDir::default();
    // One fragment, so the zone map's one zone holds a matching row and three
    // non-matching ones. Without the recheck the whole zone would be returned.
    let dataset = json_index_with_recorded_docs(
        &test_uri,
        data_storage_version,
        "zonemap",
        JsonTargetDataType::Unspecified as i32,
        &[
            r#"{"p":"hay"}"#,
            r#"{"p":"needle"}"#,
            r#"{"p":"hay"}"#,
            r#"{}"#,
        ],
        4,
    )
    .await;

    for (filter, expected_ids, needs_recheck) in [
        // The zone map only proves the zone may contain this value, so the
        // original predicate has to reject the other rows in it.
        ("json_get_string(json, 'p') = 'needle'", vec![1], true),
        // The persisted null bitmap makes this one exact: no recheck needed.
        ("json_get_string(json, 'p') IS NULL", vec![3], false),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            plan.contains("json_p_idx"),
            "a recoverable zone map type must be routed: {plan}"
        );
        if needs_recheck {
            assert!(
                plan.contains("json_get_string"),
                "the original predicate must recheck the inexact zone map: {plan}"
            );
        }
    }
}

/// The NGram postings store trigrams, not the values' type, but the wrapper's
/// own training path fixes that type: it decodes a JSON string to `Utf8`, and
/// NGram's trainer accepts only `Utf8` or `LargeUtf8`. Recovery must still open
/// the postings file so an unreadable target keeps the scan fallback.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_unrecorded_ngram_target_type_recovers(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;
    use object_store::ObjectStoreExt;

    let test_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &test_uri,
        data_storage_version,
        "ngram",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;
    let filters = [
        ("contains(json_get_string(json, 'p'), 'man')", vec![0], true),
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2], false),
    ];
    for (filter, expected_ids, uses_index) in &filters {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, *expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, *expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        if *uses_index {
            assert!(
                plan.contains("json_p_idx"),
                "a recoverable ngram type must be routed: {plan}"
            );
        } else {
            assert!(
                !plan.contains("json_p_idx"),
                "a mismatched accessor must not reach a recovered Utf8 ngram target: {plan}"
            );
        }
    }

    // Remove the target after the successful checks: recovery's postings probe
    // must turn the same predicate back into a scan instead of a query failure.
    let index = dataset
        .load_indices_by_name("json_p_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let postings_path = dataset
        .indice_files_dir(&index)
        .unwrap()
        .join(index.uuid.to_string())
        .join(lance_index::scalar::ngram::POSTINGS_FILENAME);
    dataset
        .table_state
        .object_store
        .inner
        .delete(&postings_path)
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri.as_str()).await.unwrap();
    for (filter, expected_ids, _) in &filters {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, *expected_ids, "indexed scan: {filter}");
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_p_idx"),
            "an unreadable ngram target must leave the predicate to the scan: {plan}"
        );
    }
}

/// Details that cannot be decoded, a missing target and a target this build has
/// no plugin for all reach `new_query_parser` before recovery runs. None may
/// panic, and none may stop the scan from answering the predicate.
#[rstest]
#[case::legacy_undecodable(LanceFileVersion::Legacy, "undecodable")]
#[case::legacy_missing_target(LanceFileVersion::Legacy, "missing_target")]
#[case::legacy_unknown_plugin(LanceFileVersion::Legacy, "unknown_plugin")]
#[case::stable_undecodable(LanceFileVersion::Stable, "undecodable")]
#[case::stable_missing_target(LanceFileVersion::Stable, "missing_target")]
#[case::stable_unknown_plugin(LanceFileVersion::Stable, "unknown_plugin")]
#[tokio::test]
async fn test_json_malformed_details_fall_back_to_scan_without_panicking(
    #[case] data_storage_version: LanceFileVersion,
    #[case] variant: &str,
) {
    use lance_index::pb::{JsonIndexDetails, JsonTargetDataType};
    use lance_table::format::IndexMetadata;
    use prost::Message;

    let test_uri = TempStrDir::default();
    let mut dataset = json_index_with_recorded_type(
        &test_uri,
        data_storage_version,
        "btree",
        JsonTargetDataType::Utf8 as i32,
    )
    .await;
    let mut segment = dataset
        .load_indices_by_name("json_p_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let details =
        JsonIndexDetails::decode(segment.index_details.as_ref().unwrap().value.as_slice()).unwrap();
    let rewritten = match variant {
        "undecodable" => prost_types::Any {
            type_url: segment.index_details.as_ref().unwrap().type_url.clone(),
            value: b"not a protobuf message".to_vec(),
        },
        "missing_target" => {
            let details = JsonIndexDetails {
                target_details: None,
                ..details
            };
            prost_types::Any::from_msg(&details).unwrap()
        }
        "unknown_plugin" => {
            let mut target_details = details.target_details.unwrap();
            target_details.type_url = "/lance.index.pb.NoSuchTargetDetails".to_string();
            let details = JsonIndexDetails {
                target_details: Some(target_details),
                ..details
            };
            prost_types::Any::from_msg(&details).unwrap()
        }
        other => panic!("unknown fixture variant {other}"),
    };
    segment = IndexMetadata {
        index_details: Some(Arc::new(rewritten)),
        ..segment
    };
    dataset
        .commit_existing_index_segments("json_p_idx", "json", vec![segment])
        .await
        .unwrap();

    // A fresh session, so no cached parser hides the rewritten details.
    let dataset = Dataset::open(test_uri.as_str()).await.unwrap();
    let filter = "json_get_string(json, 'p') IS NULL";
    let indexed = json_filter_ids(&dataset, filter, true).await;
    assert_eq!(indexed, vec![1, 2], "indexed scan: {filter}");
    let full = json_filter_ids(&dataset, filter, false).await;
    assert_eq!(full, vec![1, 2], "full-scan oracle: {filter}");
    let plan = dataset
        .scan()
        .filter(filter)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(
        !plan.contains("json_p_idx"),
        "unusable details must leave the predicate to the scan: {plan}"
    );
}

/// Recovery makes a formerly inert segment routable, so a segment that fails
/// recovery beside a recorded sibling must still decline the whole name: the
/// recorded segment alone would answer only its own fragments and silently drop
/// the other's rows.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_missing_segment_with_recorded_sibling_declines_name(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;
    use object_store::ObjectStoreExt;

    let test_uri = TempStrDir::default();
    let batch = json_id_batch(vec![r#"{"v": 1}"#, r#"{"v": 2}"#]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_v_idx",
        &[
            JsonSegmentSpec::new("btree", "Int64", "v"),
            JsonSegmentSpec::new("btree", "Int64", "v")
                .with_recorded_target_data_type(JsonTargetDataType::Unspecified as i32),
        ],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_v_idx", "json", segments)
        .await
        .unwrap();
    let unrecoverable = dataset
        .load_indices_by_name("json_v_idx")
        .await
        .unwrap()
        .into_iter()
        .find(|segment| segment.fragment_bitmap.as_ref().unwrap().contains(1))
        .unwrap();
    let lookup_path = dataset
        .indice_files_dir(&unrecoverable)
        .unwrap()
        .join(unrecoverable.uuid.to_string())
        .join("page_lookup.lance");
    dataset
        .table_state
        .object_store
        .inner
        .delete(&lookup_path)
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri.as_str()).await.unwrap();
    for (filter, expected_ids) in [
        ("json_get_int(json, 'v') = 1", vec![0]),
        ("json_get_int(json, 'v') = 2", vec![1]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_v_idx"),
            "one unreadable segment must decline the whole name: {plan}"
        );
    }
}

/// Recovering a target type must cost index metadata, not index data.
///
/// The same dataset with a recorded type is the baseline: planning reads nothing
/// for it. With the type removed, cold planning opens each affected target's
/// lookup file, a second planning call on the same dataset reads nothing, and
/// only executing the query pays for the lookup rows.
///
/// On this fixture (2000 pages) cold planning is one request for one 4 KiB
/// block, against a lookup file of 48,736 bytes with the legacy index file
/// format and 9,195 bytes with the stable one, whose footer shares that block
/// with the last of the data. Executing the query reads 48,000 bytes of lookup
/// rows in the first case. That gap is the whole point: the same bytes are read
/// either way, but only when a query actually runs.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_type_recovery_planning_reads_only_index_metadata(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_io::assert_io_eq;
    use lance_io::utils::tracking_store::IoStats;
    use object_store::ObjectStoreExt;

    const ROWS: i64 = 2000;

    let test_uri = TempStrDir::default();
    let docs = (0..ROWS)
        .map(|v| format!(r#"{{"v": {v}}}"#))
        .collect::<Vec<_>>();
    let batch = json_id_batch(docs.iter().map(String::as_str).collect());
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // A zone per row keeps the lookup file large enough that reading its rows
    // is distinguishable from reading its metadata.
    let params = ScalarIndexParams::new("json".to_string()).with_params(&serde_json::json!({
        "target_index_type": "btree",
        "target_data_type": "Int64",
        "target_index_parameters": serde_json::to_string(&serde_json::json!({
            "zone_size": 1,
        })).unwrap(),
        "path": "v",
    }));
    dataset
        .create_index(
            &["json"],
            IndexType::Scalar,
            Some("json_v_idx".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    // The baseline: the same index with its type recorded reads nothing at
    // planning time.
    let recorded = Dataset::open(test_uri.as_str()).await.unwrap();
    let io = recorded
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();
    io.incremental_stats();
    recorded.scalar_index_info().await.unwrap();
    let recorded_planning = io.incremental_stats();
    assert_io_eq!(
        recorded_planning,
        read_iops,
        0,
        "a recorded type must not read any index file, got {:?}",
        recorded_planning.requests
    );

    forget_json_index_target_type(&mut dataset, "json_v_idx").await;

    let index = dataset
        .load_indices_by_name("json_v_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let lookup_path = dataset
        .indice_files_dir(&index)
        .unwrap()
        .join(index.uuid.to_string())
        .join("page_lookup.lance");
    let lookup_size = dataset
        .table_state
        .object_store
        .inner
        .head(&lookup_path)
        .await
        .unwrap()
        .size;
    assert!(
        lookup_size > 8 * 1024,
        "the fixture needs a lookup file whose rows are distinguishable from its metadata, \
         got {lookup_size} bytes at {lookup_path} for index {}",
        index.uuid
    );

    // A fresh session, so planning starts with no cached index metadata.
    let dataset = Dataset::open(test_uri.as_str()).await.unwrap();
    let io = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();
    io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let cold = io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let warm = io.incremental_stats();

    let ids = json_filter_ids(&dataset, "json_get_int(json, 'v') = 7", true).await;
    assert_eq!(ids, vec![7]);
    let execute = io.incremental_stats();

    let lookup_bytes = |stats: &IoStats| -> u64 {
        stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().ends_with("page_lookup.lance"))
            .map(|request| {
                request
                    .range
                    .as_ref()
                    .map_or(0, |range| range.end - range.start)
            })
            .sum()
    };

    assert_eq!(
        cold.read_iops, 1,
        "cold planning opens the target's lookup file exactly once, got {:?}",
        cold.requests
    );
    assert!(
        cold.read_bytes < lookup_size,
        "cold planning must read part of the lookup file, not all of it: {} of {} bytes",
        cold.read_bytes,
        lookup_size
    );
    assert!(
        cold.requests
            .iter()
            .all(|request| request.path.as_ref().ends_with("page_lookup.lance")),
        "cold planning must not touch the target's page data: {:?}",
        cold.requests
    );
    assert_eq!(
        warm.read_iops, 0,
        "a second planning call on the same dataset reads nothing, got {:?}",
        warm.requests
    );
    assert!(
        lookup_bytes(&execute) > cold.read_bytes,
        "execution has to read the lookup rows planning skipped: {} vs {} bytes",
        lookup_bytes(&execute),
        cold.read_bytes
    );

    // Each segment that needs a type pays one open, so the cold cost is linear
    // in the number of legacy segments rather than in their size.
    let multi_uri = TempStrDir::default();
    let batch = json_id_batch(vec![r#"{"v": 1}"#, r#"{"v": 2}"#]);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &multi_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let unspecified = lance_index::pb::JsonTargetDataType::Unspecified as i32;
    let segments = json_index_segment_per_fragment(
        &mut dataset,
        "json_v_idx",
        &[
            JsonSegmentSpec::new("btree", "Int64", "v").with_recorded_target_data_type(unspecified),
            JsonSegmentSpec::new("btree", "Int64", "v").with_recorded_target_data_type(unspecified),
        ],
    )
    .await;
    dataset
        .commit_existing_index_segments("json_v_idx", "json", segments)
        .await
        .unwrap();
    assert_eq!(
        dataset
            .load_indices_by_name("json_v_idx")
            .await
            .unwrap()
            .len(),
        2
    );

    let dataset = Dataset::open(multi_uri.as_str()).await.unwrap();
    let io = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();
    io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let multi_cold = io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let multi_warm = io.incremental_stats();
    assert_eq!(
        multi_cold.read_iops, 2,
        "one open per segment without a type, got {:?}",
        multi_cold.requests
    );
    assert!(
        multi_cold
            .requests
            .iter()
            .all(|request| request.path.as_ref().ends_with("page_lookup.lance")),
        "recovering several segments still reads no page data: {:?}",
        multi_cold.requests
    );
    assert_eq!(
        multi_warm.read_iops, 0,
        "the second planning call reads nothing, got {:?}",
        multi_warm.requests
    );
}

/// The newly recoverable zone map costs one index-metadata open per segment
/// without a recorded type, like the B-tree path: a recorded type reads
/// nothing, cold planning reads one block of `zonemap.lance`, and repeated
/// planning on the same session reads nothing. Only executing the query reads
/// the zone rows.
///
/// The numbers are this fixture's: 512 single-row zones, with a zonemap file
/// well over one block. They say nothing about a file that fits in one block or
/// about a remote store.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_zonemap_type_recovery_planning_reads_only_metadata(
    #[case] data_storage_version: LanceFileVersion,
) {
    use object_store::ObjectStoreExt;

    const ROWS: i64 = 2048;

    let test_uri = TempStrDir::default();
    let docs = (0..ROWS)
        .map(|v| format!(r#"{{"v": {v}}}"#))
        .collect::<Vec<_>>();
    let batch = json_id_batch(docs.iter().map(String::as_str).collect());
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::new("json".to_string()).with_params(&serde_json::json!({
        "target_index_type": "zonemap",
        "target_data_type": "Int64",
        "target_index_parameters": serde_json::to_string(&serde_json::json!({
            "rows_per_zone": 1,
        })).unwrap(),
        "path": "v",
    }));
    dataset
        .create_index(
            &["json"],
            IndexType::Scalar,
            Some("json_v_idx".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    let recorded = Dataset::open(test_uri.as_str()).await.unwrap();
    let io = recorded
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();
    io.incremental_stats();
    recorded.scalar_index_info().await.unwrap();
    let recorded_planning = io.incremental_stats();
    assert_eq!(
        recorded_planning.read_iops, 0,
        "a recorded type must not read any index file, got {:?}",
        recorded_planning.requests
    );

    forget_json_index_target_type(&mut dataset, "json_v_idx").await;

    let index = dataset
        .load_indices_by_name("json_v_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let zonemap_path = dataset
        .indice_files_dir(&index)
        .unwrap()
        .join(index.uuid.to_string())
        .join("zonemap.lance");
    let zonemap_size = dataset
        .table_state
        .object_store
        .inner
        .head(&zonemap_path)
        .await
        .unwrap()
        .size;
    assert!(
        zonemap_size > 8 * 1024,
        "the fixture needs a zonemap file larger than one read block, got \
         {zonemap_size} bytes at {zonemap_path}"
    );

    let dataset = Dataset::open(test_uri.as_str()).await.unwrap();
    let io = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();
    io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let cold = io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let warm = io.incremental_stats();
    dataset.scalar_index_info().await.unwrap();
    let repeated = io.incremental_stats();

    let ids = json_filter_ids(&dataset, "json_get_int(json, 'v') = 7", true).await;
    assert_eq!(ids, vec![7]);
    let execute = io.incremental_stats();

    let zonemap_bytes = |stats: &lance_io::utils::tracking_store::IoStats| -> u64 {
        stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().ends_with("zonemap.lance"))
            .map(|request| {
                request
                    .range
                    .as_ref()
                    .map_or(0, |range| range.end - range.start)
            })
            .sum()
    };

    assert_eq!(
        cold.read_iops, 1,
        "cold planning opens the target's zonemap file exactly once, got {:?}",
        cold.requests
    );
    assert!(
        cold.read_bytes < zonemap_size,
        "cold planning must read part of the zonemap file, not all of it: {} of {} bytes",
        cold.read_bytes,
        zonemap_size
    );
    assert!(
        cold.requests
            .iter()
            .all(|request| request.path.as_ref().ends_with("zonemap.lance")),
        "cold planning must not touch the target's other files: {:?}",
        cold.requests
    );
    assert_eq!(
        warm.read_iops, 0,
        "a second planning call on the same dataset reads nothing, got {:?}",
        warm.requests
    );
    assert_eq!(
        repeated.read_iops, 0,
        "repeated planning on the same dataset reads nothing, got {:?}",
        repeated.requests
    );
    assert!(
        zonemap_bytes(&execute) > cold.read_bytes,
        "execution has to read the zone rows planning skipped: {} vs {} bytes",
        zonemap_bytes(&execute),
        cold.read_bytes
    );
}

/// A value this build does not know was written by a later build, and its
/// meaning must not be read out of the target's files: only `Unspecified`
/// licenses asking the target. The files here can answer, so routing would
/// succeed if the raw value were reinterpreted.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_unknown_target_data_type_is_not_reinterpreted(
    #[case] data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();
    let dataset =
        json_index_with_recorded_type(&test_uri, data_storage_version, "btree", 999).await;

    for (filter, expected_ids) in [
        ("json_get_string(json, 'p') IS NULL", vec![1, 2]),
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let full = json_filter_ids(&dataset, filter, false).await;
        assert_eq!(full, expected_ids, "full-scan oracle: {filter}");

        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_p_idx"),
            "an unknown target data type must not be routed: {plan}"
        );
    }
}

/// Each JSON target is recovered from what it actually persists: targets with
/// a typed key or bound column report that type from the index file, NGram's
/// domain follows from the wrapper's training path, and a target that records
/// no key type (BloomFilter) leaves the predicate to the scan.
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_json_unrecorded_target_type_boundaries(
    #[case] data_storage_version: LanceFileVersion,
) {
    use lance_index::pb::JsonTargetDataType;

    let test_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &test_uri,
        data_storage_version,
        "bitmap",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;
    // A bitmap target stores its keys in the lookup file's `keys` column with
    // the type it was trained on, so the same recovery applies.
    for (filter, expected_ids) in [
        ("json_get_string(json, 'p') IS NULL", vec![1, 2]),
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        if filter.contains("json_get_string") {
            assert!(
                plan.contains("json_p_idx"),
                "a recoverable bitmap target type must be routed: {plan}"
            );
        } else {
            assert!(
                !plan.contains("json_p_idx"),
                "a mismatched accessor must not reach a recovered Utf8 bitmap target: {plan}"
            );
        }
    }

    // A zone map's `min` column is typed by the values it was trained on, so
    // the same schema-only recovery applies and the inexact target keeps its
    // recheck (covered with multi-row zones in
    // `test_json_unrecorded_zonemap_target_type_recovers_and_rechecks`).
    let zone_map_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &zone_map_uri,
        data_storage_version,
        "zonemap",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;
    for (filter, expected_ids) in [
        ("json_get_string(json, 'p') IS NULL", vec![1, 2]),
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2]),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        if filter.contains("json_get_string") {
            assert!(
                plan.contains("json_p_idx"),
                "a recoverable zone map target type must be routed: {plan}"
            );
        } else {
            assert!(
                !plan.contains("json_p_idx"),
                "a mismatched accessor must not reach a recovered Utf8 zone map target: {plan}"
            );
        }
    }

    // NGram stores trigrams, not values, but the wrapper's training path fixes
    // the domain to Utf8 (see `test_json_unrecorded_ngram_target_type_recovers`).
    let ngram_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &ngram_uri,
        data_storage_version,
        "ngram",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;
    for (filter, expected_ids, uses_index) in [
        ("contains(json_get_string(json, 'p'), 'man')", vec![0], true),
        ("json_get_int(json, 'p') IS NULL", vec![0, 1, 2], false),
    ] {
        let indexed = json_filter_ids(&dataset, filter, true).await;
        assert_eq!(indexed, expected_ids, "indexed scan: {filter}");
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        if uses_index {
            assert!(
                plan.contains("json_p_idx"),
                "a recoverable ngram target type must be routed: {plan}"
            );
        } else {
            assert!(
                !plan.contains("json_p_idx"),
                "a mismatched accessor must not reach a recovered Utf8 ngram target: {plan}"
            );
        }
    }

    // A BloomFilter persists no key type at all, so there is nothing to recover
    // from: its value domain is unknowable from the index alone and the scan
    // has to answer.
    let bloom_uri = TempStrDir::default();
    let dataset = json_index_with_recorded_type(
        &bloom_uri,
        data_storage_version,
        "bloomfilter",
        JsonTargetDataType::Unspecified as i32,
    )
    .await;
    for filter in [
        "json_get_string(json, 'p') = '7'",
        "json_get_string(json, 'p') IS NULL",
    ] {
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            !plan.contains("json_p_idx"),
            "a bloom filter target cannot report a value domain and must not be routed: {plan}"
        );
    }
}
