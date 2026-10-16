// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::as_primitive_array;
use arrow::datatypes::{Int32Type, Int64Type};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float16Array, Float32Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, RecordBatchIterator, StringArray, StructArray,
};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use futures::TryStreamExt;
use futures::stream::StreamExt;
use half::f16;
use lance_arrow::SchemaExt;
use lance_core::ROW_ID;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datagen::{
    ArrayGeneratorExt, BatchCount, ByteCount, Dimension, RowCount, array, gen_batch,
};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::inverted::query::MatchQuery;
use lance_index::scalar::{FullTextSearchQuery, ScalarIndexParams};
use lance_index::vector::ApproxMode;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
use object_store::throttle::ThrottleConfig;
use rstest::rstest;

use crate::blob::{BlobArrayBuilder, blob_field};
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::scanner::test_dataset::{
    TestVectorDataset, assert_scan_slice_matches, find_filtered_read,
};
use crate::dataset::scanner::{ColumnOrdering, DatasetRecordBatchStream, MaterializationStyle};
use crate::dataset::{Dataset, NewColumnTransform, WriteMode, WriteParams};
use crate::datatypes::Schema;
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::io::exec::filtered_read::FilteredReadThreadingMode;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, ThrottledStoreWrapper};
use crate::{Error, Result};

#[tokio::test]
async fn test_batch_size() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, true),
        ArrowField::new("s", DataType::Utf8, true),
    ]));

    let batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                    Arc::new(StringArray::from_iter_values(
                        (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();

    for use_filter in [false, true] {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.clone().into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut builder = dataset.scan();
        builder.batch_size(8);
        if use_filter {
            builder.filter("i IS NOT NULL").unwrap();
        }
        let mut stream = builder.try_into_stream().await.unwrap();
        let mut rows_read = 0;
        while let Some(next) = stream.next().await {
            let next = next.unwrap();
            let expected = 8.min(100 - rows_read);
            assert_eq!(next.num_rows(), expected);
            rows_read += next.num_rows();
        }
    }
}

#[tokio::test]
async fn test_batch_size_bytes_across_data_files() {
    let num_rows = 300;
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..num_rows))],
    )
    .unwrap();
    let test_dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            max_rows_per_file: num_rows as usize + 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let wide_value = "abcdefghij".repeat(6);
    let wide_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "wide",
        DataType::Utf8,
        false,
    )]));
    let wide_batch = RecordBatch::try_new(
        wide_schema.clone(),
        vec![Arc::new(StringArray::from_iter_values(
            (0..num_rows).map(|_| wide_value.as_str()),
        ))],
    )
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::Reader(Box::new(RecordBatchIterator::new(
                [Ok(wide_batch)],
                wide_schema,
            ))),
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(dataset.get_fragment(0).unwrap().num_data_files(), 2);

    let target_bytes = 8 * 1024;
    let mut scan = dataset.scan();
    scan.project(&["id", "wide"])
        .unwrap()
        .batch_size_bytes(target_bytes);
    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        num_rows as usize
    );
    for batch in &batches {
        let wide = batch["wide"].as_string::<i32>();
        let values_bytes = (*wide.value_offsets().last().unwrap()
            - *wide.value_offsets().first().unwrap()) as usize;
        let logical_bytes = batch.num_rows() * std::mem::size_of::<i64>()
            + std::mem::size_of_val(wide.value_offsets())
            + values_bytes;
        assert!(
            logical_bytes <= target_bytes as usize,
            "batch has {logical_bytes} logical bytes, target is {target_bytes}"
        );
    }

    let mut scan = dataset.scan();
    scan.project(&["id", "wide"])
        .unwrap()
        .batch_size(10)
        .batch_size_bytes(target_bytes);
    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert!(batches.iter().all(|batch| batch.num_rows() <= 10));

    let mut scan = dataset.scan();
    scan.project(&["id", "wide"])
        .unwrap()
        .batch_size(200)
        .batch_size_bytes(target_bytes)
        .strict_batch_size(true);
    let error = scan
        .try_into_stream()
        .await
        .err()
        .expect("strict row and byte batch limits should be rejected");
    assert!(
        error
            .to_string()
            .contains("strict_batch_size=true cannot be combined with batch_size_bytes=8192"),
        "unexpected error: {error}"
    );
}

// Builds a genuine `lance.blob.v2` logical array of `rows` 8KiB payloads.
// Payloads vary per row so they do not collapse under compression.
// The legacy `lance-encoding:blob` metadata marker is rejected for file
// version >= 2.2, so fixtures must use the v2 logical array.
fn v2_blob_array(rows: usize, base: usize) -> ArrayRef {
    let mut builder = BlobArrayBuilder::new(rows);
    for r in 0..rows {
        let seed = (base + r).wrapping_mul(2654435761);
        let payload: Vec<u8> = (0usize..8 * 1024)
            .map(|i| (i.wrapping_mul(31).wrapping_add(seed) & 0xff) as u8)
            .collect();
        builder.push_bytes(&payload).unwrap();
    }
    builder.finish().unwrap()
}

#[tokio::test]
async fn test_batch_size_bytes_blob_v2_late_materialization() {
    use lance_core::datatypes::BlobHandling;
    use lance_table::io::commit::RenameCommitHandler;

    let rows_per_batch = 500usize;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("filterme", DataType::Int32, false),
        blob_field("blobs", true),
    ]));
    let batches: Vec<RecordBatch> = (0..8)
        .map(|b| {
            let base = b * rows_per_batch;
            let filterme = Arc::new(Int32Array::from_iter_values(
                (base as i32)..(base as i32 + rows_per_batch as i32),
            ));
            RecordBatch::try_new(
                schema.clone(),
                vec![filterme, v2_blob_array(rows_per_batch, base)],
            )
            .unwrap()
        })
        .collect();
    let data = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let dataset = Dataset::write(
        data,
        "memory://test",
        Some(WriteParams {
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let target_bytes = 8 * 1024;
    let mut scan = dataset.scan();
    scan.project(&["blobs"])
        .unwrap()
        .blob_handling(BlobHandling::AllBinary)
        .filter("filterme < 100")
        .unwrap()
        .batch_size_bytes(target_bytes);
    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 100);
    for batch in &batches {
        assert!(
            batch.get_array_memory_size() <= (target_bytes * 2) as usize,
            "batch has {} bytes, limit is {}",
            batch.get_array_memory_size(),
            target_bytes * 2
        );
    }
}

#[tokio::test]
async fn test_batch_size_bytes_blob_v2_vector_search() {
    use lance_core::datatypes::BlobHandling;
    use lance_table::io::commit::RenameCommitHandler;

    let rows_per_batch = 500usize;
    let item_field = Arc::new(ArrowField::new("item", DataType::Float32, true));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new("vec", DataType::FixedSizeList(item_field.clone(), 32), true),
        blob_field("blobs", true),
    ]));
    let batches: Vec<RecordBatch> = (0..8)
        .map(|b| {
            let base = b * rows_per_batch;
            let i = Arc::new(Int32Array::from_iter_values(
                (base as i32)..(base as i32 + rows_per_batch as i32),
            ));
            let vec = Arc::new(FixedSizeListArray::new(
                item_field.clone(),
                32,
                Arc::new(Float32Array::from_iter_values(
                    (0..rows_per_batch * 32).map(|v| ((base + v) % 1024) as f32),
                )),
                None,
            ));
            RecordBatch::try_new(
                schema.clone(),
                vec![i, vec, v2_blob_array(rows_per_batch, base)],
            )
            .unwrap()
        })
        .collect();
    let data = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let dataset = Dataset::write(
        data,
        "memory://test",
        Some(WriteParams {
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let query = Float32Array::from_iter_values((0..32).map(|v| v as f32));
    let target_bytes = 8 * 1024;
    let k = 20;
    let mut scan = dataset.scan();
    scan.nearest("vec", &query, k)
        .unwrap()
        .use_index(false)
        .project(&["blobs"])
        .unwrap()
        .blob_handling(BlobHandling::AllBinary)
        .batch_size_bytes(target_bytes);
    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), k);
    for batch in &batches {
        assert!(
            batch.get_array_memory_size() <= (target_bytes * 2) as usize,
            "batch has {} bytes, limit is {}",
            batch.get_array_memory_size(),
            target_bytes * 2
        );
    }
}

#[tokio::test]
async fn test_strict_batch_size() {
    let dataset = lance_datagen::gen_batch()
        .col("x", array::step::<Int32Type>())
        .anon_col(array::step::<Int64Type>())
        .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
        .await
        .unwrap();

    let mut scan = dataset.scan();
    scan.batch_size(10)
        .strict_batch_size(true)
        .filter("x % 2 == 0")
        .unwrap();

    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
    assert_eq!(batch_sizes, vec![10, 10, 1]);
}

#[tokio::test]
async fn test_column_not_exist() {
    let dataset = lance_datagen::gen_batch()
        .col("x", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
        .await
        .unwrap();

    let check_err_msg = |r: Result<DatasetRecordBatchStream>| {
        let Err(err) = r else {
            panic!("Expected an error to be raised saying column y is not found but got no error")
        };

        assert!(
            err.to_string().contains("No field named y"),
            "Expected error to contain 'No field named y' but got {}",
            err
        );
    };

    let mut scan = dataset.scan();
    scan.project(&["x", "y"]).unwrap();
    check_err_msg(scan.try_into_stream().await);

    let mut scan = dataset.scan();
    scan.project(&["y"]).unwrap();
    check_err_msg(scan.try_into_stream().await);

    // This represents a query like `SELECT 1 AS foo` which we could _technically_ satisfy
    // but it is not supported today
    let mut scan = dataset.scan();
    scan.project_with_transform(&[("foo", "1")]).unwrap();
    match scan.try_into_stream().await {
        Ok(_) => panic!("Expected an error to be raised saying not supported"),
        Err(e) => {
            assert!(
                e.to_string().contains("Received only dynamic expressions"),
                "Expected error to contain 'Received only dynamic expressions' but got {}",
                e
            );
        }
    }
}

#[cfg(not(windows))]
#[tokio::test]
async fn test_local_object_store() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, true),
        ArrowField::new("s", DataType::Utf8, true),
    ]));

    let batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                    Arc::new(StringArray::from_iter_values(
                        (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.clone().into_iter().map(Ok), schema.clone());
    Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(&format!("file-object-store://{}", test_uri))
        .await
        .unwrap();
    let mut builder = dataset.scan();
    builder.batch_size(8);
    let mut stream = builder.try_into_stream().await.unwrap();
    let mut rows_read = 0;
    while let Some(next) = stream.next().await {
        let next = next.unwrap();
        let expected = 8.min(100 - rows_read);
        assert_eq!(next.num_rows(), expected);
        rows_read += next.num_rows();
    }
}

// Regression for #6580: a scan with `filter` + `project` of a
// `(Large)List<Struct>` column used to panic in `merge_with_schema`
// (called from `TakeStream::map_batch`) because the filtered batch arrived
// as a sliced view of a larger batch and the cloned list offsets did not
// start at zero. The trigger requires (a) a `(Large)List<Struct>`
// projection where the struct is split across `filtered_read` and
// `TakeExec` and (b) a sparse-tail selectivity pattern so the trailing
// filter result lands deep inside the values buffer of its source batch.
// Parametrized over `List`/`LargeList` since the fix touches both offset
// widths in `merge_with_schema`.
#[rstest]
#[tokio::test]
async fn test_filter_project_list_struct_sparse_tail(
    // The panic is specific to v2.x storage; the legacy reader takes a
    // different code path. V2_0 and V2_2 are the versions called out in
    // the original report.
    #[values(
        LanceFileVersion::V2_0,
        LanceFileVersion::Stable,
        LanceFileVersion::V2_2
    )]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] large_list: bool,
) {
    use arrow_array::{LargeListArray, ListArray, UInt16Array};
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};

    let struct_fields = Fields::from(vec![
        Arc::new(ArrowField::new("a", DataType::Int32, true)),
        Arc::new(ArrowField::new("b", DataType::Int32, true)),
    ]);
    let item_field = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(struct_fields.clone()),
        true,
    ));
    let items_dtype = if large_list {
        DataType::LargeList(item_field.clone())
    } else {
        DataType::List(item_field.clone())
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("grp", DataType::UInt16, false),
        ArrowField::new("items", items_dtype, false),
    ]));

    let make_batch = |start: i32, n: usize, group: u16| -> RecordBatch {
        let ids = Int32Array::from_iter_values(start..start + n as i32);
        let groups = UInt16Array::from(vec![group; n]);

        let mut offsets = Vec::with_capacity(n + 1);
        let mut a_vals: Vec<i32> = Vec::new();
        let mut b_vals: Vec<i32> = Vec::new();
        offsets.push(0i64);
        for i in 0..n {
            // Variable-length lists (1..=18) so offsets don't land on
            // batch-row boundaries.
            let len = 1 + (i % 18);
            for j in 0..len {
                a_vals.push(j as i32);
                b_vals.push(-(j as i32));
            }
            offsets.push(a_vals.len() as i64);
        }
        let struct_arr = Arc::new(StructArray::new(
            struct_fields.clone(),
            vec![
                Arc::new(Int32Array::from(a_vals)) as ArrayRef,
                Arc::new(Int32Array::from(b_vals)) as ArrayRef,
            ],
            None,
        ));
        let items: ArrayRef = if large_list {
            Arc::new(LargeListArray::new(
                item_field.clone(),
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                struct_arr,
                None,
            ))
        } else {
            let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
            Arc::new(ListArray::new(
                item_field.clone(),
                OffsetBuffer::new(ScalarBuffer::from(offsets_i32)),
                struct_arr,
                None,
            ))
        };
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids) as ArrayRef,
                Arc::new(groups) as ArrayRef,
                items,
            ],
        )
        .unwrap()
    };

    // Sparse-tail selectivity (matching the original report's shape at a
    // smaller scale): a large leading block of matches, a large gap of
    // non-matches, then a small trailing match. Single fragment.
    let batches = vec![
        make_batch(0, 20_000, 7),
        make_batch(20_000, 80_000, 1),
        make_batch(100_000, 1_500, 7),
    ];

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 1_000_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Force a column split inside the `items` struct by marking `items.b`
    // as a late-materialized field: `filtered_read` returns the batch with
    // `items.a`, and `TakeExec` adds `items.b`. `merge_with_schema` then
    // takes its `List<Struct>` branch, which is where the panic was.
    let items_b_field_id = dataset
        .schema()
        .field("items")
        .unwrap()
        .child("item")
        .unwrap()
        .child("b")
        .unwrap()
        .id as u32;
    let mut scan = dataset.scan();
    scan.filter("grp = 7").unwrap();
    scan.project(&["id", "items"]).unwrap();
    scan.materialization_style(MaterializationStyle::AllEarlyExcept(vec![items_b_field_id]));
    let result = scan.try_into_batch().await.unwrap();
    assert_eq!(result.num_rows(), 21_500);
}

#[tokio::test]
async fn test_scan_regexp_match_and_non_empty_captions() {
    // Build a small dataset with three Utf8 columns and verify the full
    // scan().filter(...) path handles regexp_match combined with non-null/non-empty checks.
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("keywords", DataType::Utf8, true),
        ArrowField::new("natural_caption", DataType::Utf8, true),
        ArrowField::new("poetic_caption", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                Some("Liberty for all"),
                Some("peace"),
                Some("revolution now"),
                Some("Liberty"),
                Some("revolutionary"),
                Some("none"),
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
                Some(""),
                Some("c"),
                Some("d"),
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("x"),
                Some(""),
                Some("y"),
                Some("z"),
                None,
                Some("w"),
            ])) as ArrayRef,
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    let mut scan = dataset.scan();
    scan.filter(
        "regexp_match(keywords, 'Liberty|revolution') AND \
             (natural_caption IS NOT NULL AND natural_caption <> '' AND \
              poetic_caption IS NOT NULL AND poetic_caption <> '')",
    )
    .unwrap();

    let out = scan.try_into_batch().await.unwrap();
    assert_eq!(out.num_rows(), 1);

    let out_keywords = out
        .column_by_name("keywords")
        .unwrap()
        .as_string::<i32>()
        .value(0);
    let out_nat = out
        .column_by_name("natural_caption")
        .unwrap()
        .as_string::<i32>()
        .value(0);
    let out_poetic = out
        .column_by_name("poetic_caption")
        .unwrap()
        .as_string::<i32>()
        .value(0);

    assert_eq!(out_keywords, "Liberty for all");
    assert_eq!(out_nat, "a");
    assert_eq!(out_poetic, "x");
}

#[tokio::test]
async fn test_nested_projection() {
    let point_fields: Fields = vec![
        ArrowField::new("x", DataType::Float32, true),
        ArrowField::new("y", DataType::Float32, true),
    ]
    .into();
    let metadata_fields: Fields = vec![
        ArrowField::new("location", DataType::Struct(point_fields), true),
        ArrowField::new("age", DataType::Int32, true),
    ]
    .into();
    let metadata_field = ArrowField::new("metadata", DataType::Struct(metadata_fields), true);
    let schema = Arc::new(ArrowSchema::new(vec![
        metadata_field,
        ArrowField::new("idx", DataType::Int32, true),
    ]));
    let data = lance_datagen::rand(&schema)
        .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
        .await
        .unwrap();

    let mut scan = data.scan();
    scan.project(&["metadata.location.x", "metadata.age"])
        .unwrap();
    let batch = scan.try_into_batch().await.unwrap();

    assert_eq!(
        batch.schema().as_ref(),
        &ArrowSchema::new(vec![
            ArrowField::new("metadata.location.x", DataType::Float32, true),
            ArrowField::new("metadata.age", DataType::Int32, true),
        ])
    );

    // 0 - metadata
    // 2 - x
    // 4 - age
    let take_schema = data.schema().project_by_ids(&[0, 2, 4], false);

    let taken = data.take_rows(&[0, 5], take_schema).await.unwrap();

    // The expected schema drops y from the location field
    let part_point_fields = Fields::from(vec![ArrowField::new("x", DataType::Float32, true)]);
    let part_metadata_fields = Fields::from(vec![
        ArrowField::new("location", DataType::Struct(part_point_fields), true),
        ArrowField::new("age", DataType::Int32, true),
    ]);
    let part_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "metadata",
        DataType::Struct(part_metadata_fields),
        true,
    )]));

    assert_eq!(taken.schema(), part_schema);
}

#[tokio::test]
async fn test_project_with_schema_row_id_without_row_address() {
    // `_rowid` named in the schema, with no `_rowaddr` alongside it. Only `_rowoffset`
    // needs `AddRowOffsetExec`; asking for that node here would leave it without the
    // address column it reads, and the scan would fail to plan.
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "idx",
        DataType::Int32,
        true,
    )]));
    let data = lance_datagen::rand(&schema)
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(3))
        .await
        .unwrap();

    let requested = Schema::try_from(&ArrowSchema::new(vec![
        ArrowField::new("idx", DataType::Int32, true),
        ArrowField::new(ROW_ID, DataType::UInt64, true),
    ]))
    .unwrap();

    let mut scan = data.scan();
    scan.project_with_schema(&requested).unwrap();
    let batch = scan.try_into_batch().await.unwrap();

    assert_eq!(batch.schema().field_names(), vec!["idx", ROW_ID]);
    assert_eq!(batch.num_rows(), 6);
}

#[rstest]
#[tokio::test]
async fn test_limit(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] stable_row_ids: bool,
) -> Result<()> {
    let test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids).await?;
    let dataset = &test_ds.dataset;

    let full_data = dataset.scan().try_into_batch().await?.slice(19, 2);

    let actual = dataset
        .scan()
        .limit(Some(2), Some(19))?
        .try_into_batch()
        .await?;

    assert_eq!(actual.num_rows(), 2);
    assert_eq!(actual, full_data);

    for filter in [None, Some("i > 2")] {
        let mut scan = dataset.scan();
        if let Some(filter) = filter {
            scan.filter(filter)?;
        }
        let actual = scan.limit(Some(0), None)?.try_into_batch().await?;
        assert_eq!(actual.num_rows(), 0);
    }
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_v2_limit_pushdown_correctness(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] with_deletions: bool,
) -> Result<()> {
    let mut dataset = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset_with_params(
            FragmentCount::from(4),
            FragmentRowCount::from(10),
            Some(WriteParams {
                max_rows_per_file: 10,
                enable_stable_row_ids: stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await?;
    if with_deletions {
        dataset.delete("idx % 10 = 0").await?;
    }

    for (offset, limit) in [
        (0, 5),
        (8, 8),
        (9, 3),
        (17, 10),
        (35, 10),
        (100, 5),
        (15, 0),
    ] {
        assert_scan_slice_matches(&dataset, None, offset, limit).await?;
    }

    let fragments = dataset.fragments();
    let reordered = vec![fragments[3].clone(), fragments[1].clone()];
    assert_scan_slice_matches(&dataset, Some(&reordered), 8, 10).await?;
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_v2_limit_pushdown_prunes_fragments(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] with_deletions: bool,
) -> Result<()> {
    let mut dataset = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset_with_params(
            FragmentCount::from(32),
            FragmentRowCount::from(10),
            Some(WriteParams {
                max_rows_per_file: 10,
                enable_stable_row_ids: stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await?;
    let offset = if with_deletions {
        dataset.delete("idx % 10 = 0").await?;
        25 * 9
    } else {
        25 * 10
    };

    let candidate_fragments = dataset.fragments().len();
    let mut scan = dataset.scan();
    scan.limit(Some(3), Some(offset))?;
    let plan = scan.create_plan().await?;
    let filtered = find_filtered_read(plan.as_ref()).expect("filtered read");
    let planned_fragments = filtered
        .options()
        .fragments
        .as_ref()
        .expect("pruned fragments")
        .len();

    assert_eq!(candidate_fragments, 32);
    assert_eq!(planned_fragments, 1);
    assert_scan_slice_matches(&dataset, None, offset as usize, 3).await?;
    Ok(())
}

#[tokio::test]
async fn test_stable_row_id_limit_pushdown_after_update() -> Result<()> {
    let dataset = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset_with_params(
            FragmentCount::from(3),
            FragmentRowCount::from(10),
            Some(WriteParams {
                max_rows_per_file: 10,
                enable_stable_row_ids: true,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await?;
    let dataset = crate::dataset::UpdateBuilder::new(Arc::new(dataset))
        .update_where("idx IN (1, 11, 21)")?
        .set("idx", "idx + 1000")?
        .build()?
        .execute()
        .await?
        .new_dataset;
    let mut dataset = Arc::try_unwrap(dataset).unwrap_or_else(|dataset| dataset.as_ref().clone());

    for (offset, limit) in [(0, 5), (8, 8), (18, 8), (27, 3)] {
        assert_scan_slice_matches(&dataset, None, offset, limit).await?;
    }

    compact_files(&mut dataset, CompactionOptions::default(), None).await?;
    for (offset, limit) in [(0, 5), (8, 8), (18, 8), (27, 3)] {
        assert_scan_slice_matches(&dataset, None, offset, limit).await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_limit_with_scalar_index_and_refine_filter() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("topic", DataType::Int32, false),
        ArrowField::new("is_night", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..20)),
            Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(1, 20))),
            Arc::new(Int32Array::from_iter_values(
                (0..20).map(|i| if i < 10 { 0 } else { 1 }),
            )),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["topic"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let actual = dataset
        .scan()
        .filter("topic = 1 AND is_night = 1")
        .unwrap()
        .limit(Some(10), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(actual.num_rows(), 10);
    let ids = actual
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values();
    assert_eq!(ids, &(10..20).collect::<Vec<_>>());
}

#[test_log::test(tokio::test)]
async fn test_limit_cancel() {
    // If there is a filter and a limit and we can't use the index to satisfy
    // the filter, then we have to read until we have enough matching rows and
    // then cancel the scan.
    //
    // This test regresses the case where we fail to cancel the scan for whatever
    // reason.

    // Make the store slow so that if we don't cancel the scan, it will take a loooong time.
    let throttled = Arc::new(ThrottledStoreWrapper {
        config: ThrottleConfig {
            wait_get_per_call: Duration::from_millis(100),
            ..Default::default()
        },
    });
    let write_params = WriteParams {
        store_params: Some(ObjectStoreParams {
            object_store_wrapper: Some(throttled.clone()),
            ..Default::default()
        }),
        max_rows_per_file: 1,
        ..Default::default()
    };

    // Make a dataset with lots of tiny fragments, that will make it more obvious if we fail to cancel the scan.
    let dataset = gen_batch()
        .col("i", array::step::<Int32Type>().with_random_nulls(0.1))
        .into_ram_dataset_with_params(
            FragmentCount::from(2000),
            FragmentRowCount::from(1),
            Some(write_params),
        )
        .await
        .unwrap();

    let mut scan = dataset.scan();
    scan.filter("i IS NOT NULL").unwrap();
    scan.limit(Some(10), None).unwrap();

    let start = Instant::now();
    scan.try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let duration = start.elapsed();

    // This test is a timing test, which is unfortunate, as it may be flaky.  I'm hoping
    // we have enough wiggle room here.  The failure case is 30s on my machine and the pass
    // case is a few hundred milliseconds.
    assert!(duration < Duration::from_secs(3));
}

// More rows than one batch, so the row count alone does not keep the
// optimizer from splitting the read across partitions.
async fn scan_order_dataset() -> Dataset {
    gen_batch()
        .col("idx", array::step::<Int32Type>())
        .col(
            "grp",
            array::cycle::<Int32Type>((0..12_288).map(|idx| idx / 1_000).collect()),
        )
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(4_096))
        .await
        .unwrap()
}

#[rstest]
#[case::computed_column(None, &[], &[], false)]
// Row 1 of fragments 2, 1 and 0
#[case::take_with_filter(
    Some("_rowid IN (8589934593, 4294967297, 1) AND idx >= 0"),
    &[],
    &[],
    false
)]
#[case::sort_column_projected_away(None, &["idx"], &[], false)]
// The sorted partitions are merged back in order, so they may run in parallel.
#[case::sort_column_kept(None, &["idx"], &[("idx", "idx")], true)]
#[case::sort_key_partly_projected(None, &["grp", "idx"], &[("grp", "grp")], false)]
// `idx > 0` keeps an ordering on its second column, but a weaker one than `idx`.
#[case::sort_key_weakened(
    None,
    &["grp", "idx"],
    &[("grp", "grp"), ("flag", "idx > 0")],
    false
)]
#[tokio::test(flavor = "multi_thread")]
async fn test_scan_order_survives_parallelism(
    #[case] filter: Option<&str>,
    #[case] descending_by: &[&str],
    #[case] extra_columns: &[(&str, &str)],
    #[case] round_robin: bool,
) {
    let dataset = scan_order_dataset().await;
    let mut scan = dataset.scan();
    scan.target_parallelism(8);
    let mut columns = vec![("foo", "idx * 2")];
    columns.extend_from_slice(extra_columns);
    scan.project_with_transform(&columns).unwrap();
    if let Some(filter) = filter {
        scan.filter(filter).unwrap();
    }
    if !descending_by.is_empty() {
        let ordering = descending_by
            .iter()
            .map(|column| ColumnOrdering::desc_nulls_first(column.to_string()))
            .collect();
        scan.order_by(Some(ordering)).unwrap();
    }

    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(plan.contains("RoundRobinBatch"), round_robin, "{plan}");
    // The plan merges its own partitions back in order.
    assert_eq!(
        plan.contains("SortPreservingMergeExec"),
        round_robin,
        "{plan}"
    );
    let mut expected = match filter {
        Some(_) => vec![1, 4_097, 8_193],
        None => (0..12_288).collect::<Vec<i32>>(),
    };
    if !descending_by.is_empty() {
        expected.reverse();
    }
    let expected = expected.iter().map(|idx| idx * 2).collect::<Vec<_>>();
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(
        batch["foo"].as_primitive::<Int32Type>().values(),
        expected.as_slice()
    );
}

#[tokio::test]
async fn test_unordered_scan_keeps_parallelism() {
    let dataset = scan_order_dataset().await;
    let mut scan = dataset.scan();
    scan.target_parallelism(8);
    scan.scan_in_order(false);
    scan.project_with_transform(&[("foo", "idx * 2")]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("RoundRobinBatch"), "{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    let mut values = batch["foo"].as_primitive::<Int32Type>().values().to_vec();
    values.sort_unstable();
    assert_eq!(values, (0..12_288).map(|idx| idx * 2).collect::<Vec<_>>());
}

#[rstest]
#[tokio::test]
async fn test_ann_prefilter(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] stable_row_ids: bool,
    #[values(ApproxMode::Normal, ApproxMode::Fast)] approx_mode: ApproxMode,
    #[values(
        VectorIndexParams::ivf_pq(2, 4, 2, MetricType::L2, 2),
        VectorIndexParams::ivf_hnsw(
            MetricType::L2,
            IvfBuildParams::new(2),
            HnswBuildParams::default()
                .max_level(2)
                .num_edges(4)
                .ef_construction(16)
        ),
        VectorIndexParams::with_ivf_hnsw_pq_params(
            MetricType::L2,
            IvfBuildParams {
                num_partitions: Some(2),
                max_iters: 2,
                sample_rate: 2,
                ..Default::default()
            },
            HnswBuildParams::default()
                .max_level(2)
                .num_edges(4)
                .ef_construction(16),
            PQBuildParams {
                num_sub_vectors: 2,
                num_bits: 4,
                max_iters: 2,
                sample_rate: 2,
                ..Default::default()
            }
        ),
        VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            HnswBuildParams::default()
                .max_level(2)
                .num_edges(4)
                .ef_construction(16),
            SQBuildParams::default()
        )
    )]
    index_params: VectorIndexParams,
) {
    use lance_arrow::{FixedSizeListArrayExt, fixed_size_list_type};

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("filterable", DataType::Int32, true),
        ArrowField::new("vector", fixed_size_list_type(2, DataType::Float32), true),
    ]));

    let vector_values = Float32Array::from_iter_values((0..64).map(|x| x as f32));

    let batches = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..32)),
                Arc::new(FixedSizeListArray::try_new_from_values(vector_values, 2).unwrap()),
            ],
        )
        .unwrap(),
    ];

    let write_params = WriteParams {
        data_storage_version: Some(data_storage_version),
        max_rows_per_file: 16, // At least two files to make sure stable row ids make a difference
        enable_stable_row_ids: stable_row_ids,
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    dataset
        .create_index(&["vector"], IndexType::Vector, None, &index_params, false)
        .await
        .unwrap();

    let query_key = Arc::new(Float32Array::from_iter_values((0..2).map(|x| x as f32)));
    let mut scan = dataset.scan();
    scan.filter("filterable > 5").unwrap();
    scan.nearest("vector", query_key.as_ref(), 1).unwrap();
    scan.minimum_nprobes(2);
    scan.ef(16);
    scan.approx_mode(approx_mode);
    scan.with_row_id();

    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches.len(), 0);

    scan.prefilter(true);

    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);

    let first_match = batches[0][ROW_ID].as_primitive::<UInt64Type>().values()[0];

    // HNSW+SQ is an approximate index; this test validates *prefiltering*, so
    // every row failing `filterable > 5` (row ids 0..=5) must be excluded.
    // HNSW recall is covered by dedicated vector-index tests elsewhere.
    assert!(
        first_match > 5,
        "prefilter not honored: returned row id {first_match} should satisfy `filterable > 5`"
    );
}

#[rstest]
#[tokio::test]
async fn test_filter_on_large_utf8(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "ls",
        DataType::LargeUtf8,
        true,
    )]));

    let batches = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from_iter_values(
                (0..10).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap(),
    ];

    let write_params = WriteParams {
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    let mut scan = dataset.scan();
    scan.filter("ls = 's-8'").unwrap();

    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let batch = &batches[0];

    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(LargeStringArray::from_iter_values(
            (8..9).map(|v| format!("s-{}", v)),
        ))],
    )
    .unwrap();

    assert_eq!(batch, &expected);
}

#[rstest]
#[tokio::test]
async fn test_filter_with_regex(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "ls",
        DataType::Utf8,
        true,
    )]));

    let batches = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..20).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap(),
    ];

    let write_params = WriteParams {
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    let mut scan = dataset.scan();
    scan.filter("regexp_match(ls, 's-1.')").unwrap();

    let stream = scan.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();
    let batch = &batches[0];

    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from_iter_values(
            (10..=19).map(|v| format!("s-{}", v)),
        ))],
    )
    .unwrap();

    assert_eq!(batch, &expected);
}

#[tokio::test]
async fn test_filter_proj_bug() {
    let struct_i_field = ArrowField::new("i", DataType::Int32, true);
    let struct_o_field = ArrowField::new("o", DataType::Utf8, true);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(
            "struct",
            DataType::Struct(vec![struct_i_field.clone(), struct_o_field.clone()].into()),
            true,
        ),
        ArrowField::new("s", DataType::Utf8, true),
    ]));

    let input_batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            let struct_i_arr: Arc<Int32Array> =
                Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20));
            let struct_o_arr: Arc<StringArray> = Arc::new(StringArray::from_iter_values(
                (i * 20..(i + 1) * 20).map(|v| format!("o-{:02}", v)),
            ));
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StructArray::from(vec![
                        (Arc::new(struct_i_field.clone()), struct_i_arr as ArrayRef),
                        (Arc::new(struct_o_field.clone()), struct_o_arr as ArrayRef),
                    ])),
                    Arc::new(StringArray::from_iter_values(
                        (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    let batches =
        RecordBatchIterator::new(input_batches.clone().into_iter().map(Ok), schema.clone());
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(LanceFileVersion::Legacy),
        ..Default::default()
    };
    Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    let batches = dataset
        .scan()
        .filter("struct.i >= 20")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let batch = concat_batches(&batches[0].schema(), &batches).unwrap();

    let expected_batch = concat_batches(&schema, &input_batches.as_slice()[1..]).unwrap();
    assert_eq!(batch, expected_batch);

    // different order
    let batches = dataset
        .scan()
        .filter("struct.o >= 'o-20'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch, expected_batch);

    // other reported bug with nested top level column access
    let batches = dataset
        .scan()
        .project(vec!["struct"].as_slice())
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    concat_batches(&batches[0].schema(), &batches).unwrap();
}

#[rstest]
#[tokio::test]
async fn test_ann_with_deletion(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] stable_row_ids: bool,
) {
    const PARTITIONS: usize = 4;
    // A partition is trained only when the data can give it a codebook's
    // worth of vectors, so the fixture has to cover every partition it asks
    // for; the cached-entry count below counts per partition.
    const ROWS: usize = PARTITIONS * 256;

    let vec_params = vec![
        // TODO: re-enable diskann test when we can tune to get reproducible results.
        // VectorIndexParams::with_diskann_params(MetricType::L2, DiskANNParams::new(10, 1.5, 10)),
        VectorIndexParams::ivf_pq(PARTITIONS, 8, 2, MetricType::L2, 2),
    ];
    for params in vec_params {
        use lance_arrow::FixedSizeListArrayExt;

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // make dataset
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    32,
                ),
                true,
            ),
        ]));

        // vectors are [1, 1, 1, ...] [2, 2, 2, ...]
        let vector_values: Float32Array = (0..32 * ROWS).map(|v| (v / 32) as f32 + 1.0).collect();
        let vectors = FixedSizeListArray::try_new_from_values(vector_values, 32).unwrap();

        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..ROWS as i32)),
                    Arc::new(vectors.clone()),
                ],
            )
            .unwrap(),
        ];

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.index_cache_entry_count().await, 0);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let mut scan = dataset.scan();
        // closest be i = 0..5
        let key: Float32Array = (0..32).map(|_v| 1.0_f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(100);
        scan.minimum_nprobes(100);

        assert_eq!(
            dataset.index_cache_entry_count().await,
            2, // 2 for index metadata at version 1 and 2.
        );
        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            dataset.index_cache_entry_count().await,
            5 + dataset.versions().await.unwrap().len()
        );
        assert_eq!(results.len(), 1);
        let batch = &results[0];

        let expected_i = BTreeSet::from_iter(vec![0, 1, 2, 3, 4]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);

        // DELETE top result and search again

        dataset.delete("i = 1").await.unwrap();
        let mut scan = dataset.scan();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(100);
        scan.minimum_nprobes(100);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        // i=1 was deleted, and 5 is the next best, the reset shouldn't change
        let expected_i = BTreeSet::from_iter(vec![0, 2, 3, 4, 5]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);

        // Add a second fragment and test the case where there are no deletion
        // files but there are missing fragments.
        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(ROWS as i32..2 * ROWS as i32)),
                    Arc::new(vectors),
                ],
            )
            .unwrap(),
        ];

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        dataset.delete(&format!("i < {ROWS}")).await.unwrap();

        let mut scan = dataset.scan();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(100);
        scan.minimum_nprobes(100);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        // It should not pick up any results from the first fragment
        let first = ROWS as i32;
        let expected_i =
            BTreeSet::from_iter(vec![first, first + 1, first + 2, first + 3, first + 4]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);
    }
}

#[tokio::test]
async fn test_projection_order() {
    let vec_params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 2);
    let mut data = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .col("text", array::rand_utf8(ByteCount::from(10), false))
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(100))
        .await
        .unwrap();
    data.create_index(&["vec"], IndexType::Vector, None, &vec_params, true)
        .await
        .unwrap();

    let mut scan = data.scan();
    scan.nearest("vec", &Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]), 5)
        .unwrap();
    scan.with_row_id().project(&["text"]).unwrap();

    let results = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(
        results[0].schema().field_names(),
        vec!["text", "_distance", "_rowid"]
    );
}

#[rstest]
#[tokio::test]
async fn test_count_rows_with_filter(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let mut data_gen = BatchGenerator::new().col(Box::new(
        IncrementingInt32::new().named("Filter_me".to_owned()),
    ));
    Dataset::write(
        data_gen.batch(32),
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    assert_eq!(32, dataset.count_rows(None).await.unwrap());
    assert_eq!(
        16,
        dataset
            .count_rows(Some("`Filter_me` > 15".to_string()))
            .await
            .unwrap()
    );
}

#[rstest]
#[tokio::test]
async fn test_dynamic_projection(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
    Dataset::write(
        data_gen.batch(32),
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 32);

    let mut scanner = dataset.scan();

    let scan_res = scanner
        .project_with_transform(&[("bool", "i > 15")])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(1, scan_res.num_columns());

    let bool_col = scan_res
        .column_by_name("bool")
        .expect("bool column should exist");
    let bool_arr = bool_col.as_boolean();
    for i in 0..32 {
        if i > 15 {
            assert!(bool_arr.value(i));
        } else {
            assert!(!bool_arr.value(i));
        }
    }
}

#[rstest]
#[tokio::test]
async fn test_column_casting_function(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let mut data_gen =
        BatchGenerator::new().col(Box::new(RandomVector::new().named("vec".to_owned())));
    Dataset::write(
        data_gen.batch(32),
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 32);

    let mut scanner = dataset.scan();

    let scan_res = scanner
        .project_with_transform(&[("f16", "_cast_list_f16(vec)")])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(1, scan_res.num_columns());
    assert_eq!(32, scan_res.num_rows());
    assert_eq!("f16", scan_res.schema().field(0).name());

    let mut scanner = dataset.scan();
    let scan_res_original = scanner
        .project(&["vec"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let f32_col: &Float32Array = scan_res_original
        .column_by_name("vec")
        .unwrap()
        .as_fixed_size_list()
        .values()
        .as_primitive();
    let f16_col: &Float16Array = scan_res
        .column_by_name("f16")
        .unwrap()
        .as_fixed_size_list()
        .values()
        .as_primitive();

    for (f32_val, f16_val) in f32_col.iter().zip(f16_col.iter()) {
        let f32_val = f32_val.unwrap();
        let f16_val = f16_val.unwrap();
        assert_eq!(f16::from_f32(f32_val), f16_val);
    }
}

#[tokio::test]
async fn test_ivf_pq_query_parallelism_returns_same_results() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();

    let query_vector = Float32Array::from(vec![0.0; 32]);

    let mut sequential = test_ds.dataset.scan();
    sequential.nearest("vec", &query_vector, 50).unwrap();
    let sequential_results = sequential.try_into_batch().await.unwrap();

    let mut parallel = test_ds.dataset.scan();
    parallel
        .nearest("vec", &query_vector, 50)
        .unwrap()
        .query_parallelism(4);
    let parallel_results = parallel.try_into_batch().await.unwrap();

    assert_eq!(sequential_results, parallel_results);
}

#[tokio::test]
async fn test_fts_fast_search_excludes_unindexed_rows() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_fts_index().await.unwrap();
    // Append rows after index build so they stay unindexed.
    test_ds.append_data_with_range(10, 20).await.unwrap();

    let mut scanner = test_ds.dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new_query(
            MatchQuery::new("15".to_owned())
                .with_column(Some("s".to_owned()))
                .into(),
        ))
        .unwrap();
    let normal_rows = scanner.try_into_batch().await.unwrap().num_rows();

    let mut scanner = test_ds.dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new_query(
            MatchQuery::new("15".to_owned())
                .with_column(Some("s".to_owned()))
                .into(),
        ))
        .unwrap()
        .fast_search();
    let fast_rows = scanner.try_into_batch().await.unwrap().num_rows();

    assert_eq!(normal_rows, 2);
    assert_eq!(fast_rows, 1);
}

#[test_log::test(test)]
fn test_scan_finishes_all_tasks() {
    // Need to use multi-threaded runtime otherwise tasks don't run unless someone is polling somewhere
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_time()
        .build()
        .unwrap();

    runtime.block_on(async move {
        let ds = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1000), FragmentRowCount::from(10))
            .await
            .unwrap();

        // This scan with has a small I/O buffer size and batch size to mimic a real-world situation
        // that required a lot of data.  Many fragments will be scheduled at low priority and the data
        // buffer will fill up with data reads.  When the scan is abandoned, the tasks to read the fragment
        // metadata were left behind and would never finish because the data was never decoded to drain the
        // backpressure queue.
        //
        // The fix (that this test verifies) is to ensure we close the I/O scheduler when the scan is abandoned.
        let mut stream = ds
            .scan()
            .fragment_readahead(1000)
            .batch_size(1)
            .io_buffer_size(1)
            .batch_readahead(1)
            .try_into_stream()
            .await
            .unwrap();
        stream.next().await.unwrap().unwrap();
    });

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        if runtime.handle().metrics().num_alive_tasks() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    assert!(
        runtime.handle().metrics().num_alive_tasks() == 0,
        "Tasks should have finished within 10 seconds but there are still {} tasks running",
        runtime.handle().metrics().num_alive_tasks()
    );
}

#[tokio::test]
async fn test_io_buffer_size_explicit_propagated() {
    // Sanity check: an explicit .io_buffer_size(N) call must reach the
    // FilteredReadExec options unchanged, and the absence of one must leave
    // io_buffer_size_bytes as None so FilteredReadExec can pick its own
    // fallback (env var or max_bandwidth).
    let data = lance_datagen::gen_batch()
        .col("x", lance_datagen::array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(1));
    let dataset = Dataset::write(data, "memory://test_io_buffer_explicit", None)
        .await
        .unwrap();

    let plan = dataset.scan().create_plan().await.unwrap();
    let filtered =
        find_filtered_read(plan.as_ref()).expect("expected a FilteredReadExec in the scan plan");
    assert_eq!(filtered.options().io_buffer_size_bytes, None);

    let mut scanner = dataset.scan();
    scanner.io_buffer_size(7777);
    let plan = scanner.create_plan().await.unwrap();
    let filtered =
        find_filtered_read(plan.as_ref()).expect("expected a FilteredReadExec in the scan plan");
    assert_eq!(filtered.options().io_buffer_size_bytes, Some(7777));
}

#[tokio::test]
async fn test_materialization_readahead_bytes_propagated() {
    let data = lance_datagen::gen_batch()
        .col("x", lance_datagen::array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(1));
    let dataset = Dataset::write(data, "memory://test_materialization_readahead_bytes", None)
        .await
        .unwrap();

    let mut scanner = dataset.scan();
    scanner.materialization_readahead_bytes(7777);
    let plan = scanner.create_plan().await.unwrap();
    let filtered =
        find_filtered_read(plan.as_ref()).expect("expected a FilteredReadExec in the scan plan");
    assert_eq!(
        filtered.options().materialization_readahead_bytes,
        Some(7777)
    );
}

#[tokio::test]
async fn test_zero_materialization_readahead_bytes_rejected() {
    let data = lance_datagen::gen_batch()
        .col("x", lance_datagen::array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(1));
    let dataset = Dataset::write(data, "memory://test_zero_materialization_budget", None)
        .await
        .unwrap();
    let mut scanner = dataset.scan();
    scanner.materialization_readahead_bytes(0);
    let err = scanner.create_plan().await.unwrap_err();
    assert!(
        err.to_string()
            .contains("materialization_readahead_bytes must be greater than 0")
    );
}

#[tokio::test]
async fn test_batch_readahead_bounds_decode_concurrency() {
    let data = lance_datagen::gen_batch()
        .col("x", lance_datagen::array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(1));
    let dataset = Dataset::write(data, "memory://test_batch_readahead_concurrency", None)
        .await
        .unwrap();

    // Default: threading mode falls back to get_num_compute_intensive_cpus().
    let plan = dataset.scan().create_plan().await.unwrap();
    let filtered =
        find_filtered_read(plan.as_ref()).expect("expected a FilteredReadExec in the scan plan");
    assert_eq!(
        filtered.options().threading_mode,
        FilteredReadThreadingMode::OnePartitionMultipleThreads(get_num_compute_intensive_cpus()),
    );

    // Explicit batch_readahead(N) bounds the decode fan-out to N.
    let mut scanner = dataset.scan();
    scanner.batch_readahead(3);
    let plan = scanner.create_plan().await.unwrap();
    let filtered =
        find_filtered_read(plan.as_ref()).expect("expected a FilteredReadExec in the scan plan");
    assert_eq!(
        filtered.options().threading_mode,
        FilteredReadThreadingMode::OnePartitionMultipleThreads(3),
    );

    let mut scanner = dataset.scan();
    scanner.batch_readahead(0);
    let Err(Error::InvalidInput { source, .. }) = scanner.create_plan().await else {
        panic!("expected batch_readahead=0 to be rejected");
    };
    assert!(
        source
            .to_string()
            .contains("batch_readahead must be greater than 0")
    );
}
