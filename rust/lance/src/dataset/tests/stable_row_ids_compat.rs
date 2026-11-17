// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::File;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt64Type};
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
use lance_table::format::pb;
use lance_table::format::{Fragment, RowDatasetVersionMeta, RowIdMeta};
use prost::Message;
use serde_json::Value;

use crate::Dataset;
use crate::utils::test::copy_test_data_to_tmp;

fn i64_column(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|value| value.as_i64().expect("expected int64"))
        .collect()
}

fn u64_column(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|value| value.as_u64().expect("expected uint64"))
        .collect()
}

fn row_id_segments(fragment: &Fragment) -> Vec<pb::u64_segment::Segment> {
    let Some(RowIdMeta::Inline(ids)) = &fragment.row_id_meta else {
        panic!("fixture fragment must carry inline row IDs");
    };
    pb::RowIdSequence::decode(ids.bytes().as_ref())
        .expect("row ID sequence must decode")
        .segments
        .into_iter()
        .map(|segment| segment.segment.expect("row ID segment must be set"))
        .collect()
}

fn has_range(segments: &[pb::u64_segment::Segment]) -> bool {
    segments
        .iter()
        .all(|segment| matches!(segment, pb::u64_segment::Segment::Range(_)))
}

fn is_sorted_array(
    segment: &pb::u64_segment::Segment,
    expected: fn(&pb::encoded_u64_array::Array) -> bool,
) -> bool {
    matches!(
        segment,
        pb::u64_segment::Segment::SortedArray(array)
            if array.array.as_ref().is_some_and(expected)
    )
}

fn assert_segment_encoding(name: &str, fragments: &[Fragment]) {
    let segments: Vec<_> = fragments.iter().flat_map(row_id_segments).collect();
    assert!(
        !segments.is_empty(),
        "{name}: expected at least one row ID segment"
    );

    match name {
        "stable_row_id_range" => {
            assert!(has_range(&segments), "{name}: expected only Range segments");
        }
        "stable_row_id_holes" => {
            assert!(
                segments
                    .iter()
                    .any(|s| matches!(s, pb::u64_segment::Segment::RangeWithHoles(_))),
                "{name}: expected RangeWithHoles"
            );
        }
        "stable_row_id_bitmap" => {
            assert!(
                segments
                    .iter()
                    .any(|s| matches!(s, pb::u64_segment::Segment::RangeWithBitmap(_))),
                "{name}: expected RangeWithBitmap"
            );
        }
        "stable_row_id_sorted_array" => {
            assert!(
                segments.iter().any(|segment| {
                    is_sorted_array(segment, |array| {
                        matches!(array, pb::encoded_u64_array::Array::U16Array(_))
                    })
                }),
                "{name}: expected SortedArray with 16-bit offsets"
            );
        }
        "stable_row_id_sorted_array_u32" => {
            assert!(
                segments.iter().any(|segment| {
                    is_sorted_array(segment, |array| {
                        matches!(array, pb::encoded_u64_array::Array::U32Array(_))
                    })
                }),
                "{name}: expected SortedArray with 32-bit offsets"
            );
        }
        "stable_row_id_array" => {
            assert!(
                segments
                    .iter()
                    .any(|s| matches!(s, pb::u64_segment::Segment::Array(_))),
                "{name}: expected unsorted Array"
            );
        }
        "stable_row_id_lineage" => {}
        _ => panic!("unexpected fixture {name}"),
    }
}

fn assert_inline_version_sequences(name: &str, fragment: &Fragment) {
    assert!(
        fragment.created_at_version_meta.is_some(),
        "{name}: created-at version metadata missing"
    );
    assert!(
        fragment.last_updated_at_version_meta.is_some(),
        "{name}: last-updated-at version metadata missing"
    );
    for meta in [
        fragment.created_at_version_meta.as_ref(),
        fragment.last_updated_at_version_meta.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let RowDatasetVersionMeta::Inline(bytes) = meta else {
            panic!("{name}: expected inline version sequence");
        };
        let sequence = pb::RowDatasetVersionSequence::decode(bytes.as_ref())
            .expect("version sequence must decode");
        assert!(
            !sequence.runs.is_empty(),
            "{name}: version sequence must have runs"
        );
    }
}

#[tokio::test]
async fn stable_row_id_compat_fixtures() {
    let dir = copy_test_data_to_tmp("v14.0.2").unwrap();
    let expected: Value =
        serde_json::from_reader(File::open(dir.std_path().join("expected.json")).unwrap()).unwrap();
    let fixtures = expected.as_object().expect("expected object");
    let mut names: Vec<_> = fixtures.keys().cloned().collect();
    names.sort();

    for name in names {
        let fixture = &fixtures[&name];
        let uri = dir.std_path().join(format!("{name}.lance"));
        let dataset = Dataset::open(uri.to_str().unwrap()).await.unwrap();

        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS,
            0,
            "{name}: reader_feature_flags should have FLAG_STABLE_ROW_IDS"
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_STABLE_ROW_IDS,
            0,
            "{name}: writer_feature_flags should have FLAG_STABLE_ROW_IDS"
        );
        let expected_row_ids = u64_column(&fixture[ROW_ID]);
        // Deletes, updates and compaction leave gaps, so the writer's high-water
        // mark is only reachable from the generation steps, not from the live
        // maximum row ID.
        assert_eq!(
            dataset.manifest.next_row_id,
            fixture["next_row_id"]
                .as_u64()
                .expect("expected next_row_id"),
            "{name}: next_row_id"
        );

        let batch = dataset
            .scan()
            .project(&[
                "id",
                ROW_ID,
                ROW_CREATED_AT_VERSION,
                ROW_LAST_UPDATED_AT_VERSION,
            ])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<Int64Type>()
                .values(),
            i64_column(&fixture["id"]).as_slice(),
            "{name}: id column"
        );
        assert_eq!(
            batch
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values(),
            expected_row_ids.as_slice(),
            "{name}: row ID column"
        );
        assert_eq!(
            batch
                .column_by_name(ROW_CREATED_AT_VERSION)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values(),
            u64_column(&fixture[ROW_CREATED_AT_VERSION]).as_slice(),
            "{name}: created-at column"
        );
        assert_eq!(
            batch
                .column_by_name(ROW_LAST_UPDATED_AT_VERSION)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values(),
            u64_column(&fixture[ROW_LAST_UPDATED_AT_VERSION]).as_slice(),
            "{name}: last-updated-at column"
        );

        let expected_ids = i64_column(&fixture["id"]);
        let taken = dataset
            .take_rows(&expected_row_ids, dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            taken
                .column_by_name("id")
                .unwrap()
                .as_primitive::<Int64Type>()
                .values(),
            expected_ids.as_slice(),
            "{name}: take by row ID"
        );

        let fragments = dataset.manifest.fragments.as_ref();
        assert_segment_encoding(&name, fragments);
        for fragment in fragments {
            assert_inline_version_sequences(&name, fragment);
        }
        if name == "stable_row_id_lineage" {
            let max_runs = fragments
                .iter()
                .filter_map(|fragment| fragment.last_updated_at_version_meta.as_ref())
                .filter_map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(bytes) => {
                        pb::RowDatasetVersionSequence::decode(bytes.as_ref()).ok()
                    }
                    RowDatasetVersionMeta::Column => None,
                })
                .map(|sequence| sequence.runs.len())
                .max()
                .unwrap_or(0);
            assert!(
                max_runs > 1,
                "{name}: expected a version sequence with multiple runs, got {max_runs}"
            );
        }

        dataset.validate().await.unwrap();
    }
}
