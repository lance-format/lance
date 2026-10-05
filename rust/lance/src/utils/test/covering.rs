// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures for tests about covering indexes.
//!
//! No index builder writes carried values yet, so there is no API that
//! produces a covering index. Every covering test has to build a plain index
//! and then re-commit its metadata with the declaration attached, which is
//! what [`declare_covering`] does. Once a creation API exists these fixtures
//! collapse into calls to it.

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_linalg::distance::MetricType;
use lance_testing::datagen::generate_random_array;

use crate::Dataset;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;

/// Rows written per fragment by the fixtures in this module.
pub const ROWS_PER_FRAGMENT: i32 = 512;

/// Vector width used by the fixtures in this module.
pub const DIMENSION: i32 = 16;

/// Partitions used by [`create_ivf_pq_index`].
///
/// Four, not one: with a single partition the probe path is never reached, so
/// a selection rule that wrongly picks a covered index still looks correct.
pub const NUM_PARTITIONS: u32 = 4;

fn vector_field(name: &str) -> ArrowField {
    ArrowField::new(
        name,
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            DIMENSION,
        ),
        false,
    )
}

fn random_vectors(rows: i32) -> Arc<FixedSizeListArray> {
    Arc::new(
        FixedSizeListArray::try_new_from_values(
            generate_random_array(rows as usize * DIMENSION as usize),
            DIMENSION,
        )
        .unwrap(),
    )
}

/// A one-fragment dataset with a `vec` column to key an index on and an
/// `Int32` `payload` column to declare as carried.
pub async fn write_vector_payload_dataset(uri: &str) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        vector_field("vec"),
        ArrowField::new("payload", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            random_vectors(ROWS_PER_FRAGMENT),
            Arc::new(Int32Array::from_iter_values(0..ROWS_PER_FRAGMENT)),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(reader, uri, None).await.unwrap()
}

/// A one-fragment dataset whose carried column is itself a vector, for tests
/// about which column an index may be *selected* for.
pub async fn write_two_vector_column_dataset(uri: &str) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        vector_field("vec"),
        vector_field("payload_vec"),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            random_vectors(ROWS_PER_FRAGMENT),
            random_vectors(ROWS_PER_FRAGMENT),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(reader, uri, None).await.unwrap()
}

/// Append a fragment to a [`write_vector_payload_dataset`] dataset, leaving
/// every index stale.
///
/// Tests about maintenance need this so the index group really would be
/// rebuilt; without it they assert against a group the operation had no work
/// for, and keep passing if the behavior under test moves behind a no-work
/// check.
pub async fn append_vector_payload_rows(dataset: &mut Dataset, rows: i32) {
    let schema = Arc::new(ArrowSchema::new(vec![
        vector_field("vec"),
        ArrowField::new("payload", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            random_vectors(rows),
            Arc::new(Int32Array::from_iter_values(
                ROWS_PER_FRAGMENT..ROWS_PER_FRAGMENT + rows,
            )),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    dataset.append(reader, None).await.unwrap();
}

/// Build an IVF_PQ index on `column`, with [`NUM_PARTITIONS`] partitions so
/// the probe path is genuinely reached.
pub async fn create_ivf_pq_index(dataset: &mut Dataset, column: &str) {
    let params = VectorIndexParams::ivf_pq(NUM_PARTITIONS as usize, 8, 2, MetricType::L2, 50);
    dataset
        .create_index(&[column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
}

/// Build a BTree index on `column`.
pub async fn create_btree_index(dataset: &mut Dataset, column: &str, name: Option<&str>) {
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index(
            &[column],
            IndexType::BTree,
            name.map(str::to_string),
            &params,
            true,
        )
        .await
        .unwrap();
}

/// A one-fragment dataset of three `Int32` columns, for tests about covered
/// *scalar* indexes: `a` and `b` are keyed, `carried` plays the covered column.
pub async fn write_three_int_column_dataset(uri: &str) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, false),
        ArrowField::new("b", DataType::Int32, false),
        ArrowField::new("carried", DataType::Int32, false),
    ]));
    let column = || Arc::new(Int32Array::from_iter_values(0..64)) as _;
    let batch = RecordBatch::try_new(schema.clone(), vec![column(), column(), column()]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(reader, uri, None).await.unwrap()
}

/// Commit a covering declaration for `keyed`/`carried` under `name`, with no
/// index files behind it.
///
/// Unlike [`declare_covering`], this does not need a real index to exist:
/// guards that only look up an entry in the manifest are reached just as well
/// by a synthetic one, and building a real index first would cost more than it
/// proves. Returns the two field ids.
pub async fn commit_synthetic_covered_index(
    dataset: &mut Dataset,
    name: &str,
    keyed: &str,
    carried: &str,
) -> (i32, i32) {
    let keyed_id = dataset.schema().field_id(keyed).unwrap();
    let carried_id = dataset.schema().field_id(carried).unwrap();

    let covered = lance_table::format::IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: name.to_string(),
        fields: vec![keyed_id, carried_id],
        covering_fields: vec![carried_id],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(dataset.fragment_bitmap.as_ref().clone()),
        index_details: None,
        index_version: 0,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: None,
    };

    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![covered],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    (keyed_id, carried_id)
}

/// Append a fragment to a [`write_three_int_column_dataset`] dataset, leaving
/// every index stale.
pub async fn append_three_int_column_rows(dataset: &mut Dataset, rows: i32) {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, false),
        ArrowField::new("b", DataType::Int32, false),
        ArrowField::new("carried", DataType::Int32, false),
    ]));
    let column = || Arc::new(Int32Array::from_iter_values(64..64 + rows)) as _;
    let batch = RecordBatch::try_new(schema.clone(), vec![column(), column(), column()]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    dataset.append(reader, None).await.unwrap();
}

/// Re-commit the index keyed on `keyed` so that it declares `carried` as a
/// covering column, and return the two field ids.
///
/// Only that index is replaced; any other index on the table is left in place,
/// which is what tests about one covered index not blocking the others rely
/// on.
pub async fn declare_covering(dataset: &mut Dataset, keyed: &str, carried: &str) -> (i32, i32) {
    let keyed_id = dataset.schema().field_id(keyed).unwrap();
    let carried_id = dataset.schema().field_id(carried).unwrap();

    let current = dataset.load_indices().await.unwrap();
    let plain = current
        .iter()
        .find(|idx| idx.fields == vec![keyed_id])
        .cloned()
        .unwrap_or_else(|| panic!("no index keyed on '{keyed}' to declare covering on"));

    let covered = lance_table::format::IndexMetadata {
        fields: vec![keyed_id, carried_id],
        covering_fields: vec![carried_id],
        ..plain.clone()
    };

    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![covered],
                    removed_indices: vec![plain],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    (keyed_id, carried_id)
}
