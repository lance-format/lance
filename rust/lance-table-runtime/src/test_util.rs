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
