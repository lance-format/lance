// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZero;

use arrow_array::RecordBatch;
use lance_core::datatypes::Schema;
use lance_file::{version::ConcreteFileVersion, versions};
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::apply_feature_flags;
use lance_table::format::{DataFile, Manifest};
use lance_table::io::commit::{ManifestNamingScheme, write_manifest_file_to_path};
use object_store::path::Path;

use crate::paths::DATA_DIR;

pub async fn write_file(
    store: &ObjectStore,
    base: &Path,
    name: &str,
    batch: &RecordBatch,
    version: ConcreteFileVersion,
) -> DataFile {
    let schema = Schema::try_from(batch.schema_ref().as_ref()).unwrap();
    let field_ids = schema.field_ids();
    let path = base.clone().join(DATA_DIR).join(name);
    let file = store.create(&path).await.unwrap();
    let mut writer = versions::create_writer(version, file, schema, Default::default()).unwrap();
    writer.write_batch(batch).await.unwrap();
    let summary = writer.finish().await.unwrap();

    DataFile::new(
        name,
        field_ids,
        (0..batch.num_columns() as i32).collect(),
        version,
        NonZero::new(summary.size_bytes),
        None,
    )
}

pub async fn publish_manifest(store: &ObjectStore, base: &Path, manifest: &mut Manifest) {
    apply_feature_flags(manifest, false, false).unwrap();
    let path = ManifestNamingScheme::V2.manifest_path(base, manifest.version);
    write_manifest_file_to_path(store, manifest, None, &path, None)
        .await
        .unwrap();
}

/// Build a committed in-memory table using the standalone writer entry point.
pub async fn test_commit_table(num_rows: usize, num_fragments: usize) -> crate::Table {
    let uri = "memory://";
    let session = std::sync::Arc::new(crate::TableSession::default());
    let (store, base) =
        ObjectStore::from_uri_and_params(session.store_registry(), uri, &Default::default())
            .await
            .unwrap();
    let mut fragments = Vec::with_capacity(num_fragments);
    for slot in 0..num_fragments {
        let rows_per_fragment = num_rows / num_fragments;
        let start = slot * rows_per_fragment;
        let a = arrow_array::Int32Array::from_iter_values(
            (start..start + rows_per_fragment).map(|row| row as i32),
        );
        let b = arrow_array::Int32Array::from(vec![Some(0); rows_per_fragment]);
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("b", arrow_schema::DataType::Int32, true),
        ]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(a), std::sync::Arc::new(b)],
        )
        .unwrap();
        let file = write_file(
            &store,
            &base,
            &format!("{slot}.lance"),
            &batch,
            ConcreteFileVersion::V2_1,
        )
        .await;
        fragments.push(lance_table::format::Fragment {
            files: vec![file],
            physical_rows: Some(rows_per_fragment),
            ..lance_table::format::Fragment::new(slot as u64)
        });
    }
    let schema = Schema::try_from(&arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
        arrow_schema::Field::new("b", arrow_schema::DataType::Int32, true),
    ]))
    .unwrap();
    let operation = lance_table::transaction::Operation::Overwrite {
        schema,
        fragments,
        config_upsert_values: None,
        initial_bases: None,
    };
    crate::CommitBuilder::new(uri)
        .with_session(session)
        .with_object_store(store)
        .execute(lance_table::transaction::Transaction::new(
            0, operation, None,
        ))
        .await
        .unwrap()
}
