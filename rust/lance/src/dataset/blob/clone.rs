// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Copy Managed payloads and rebind descriptors for an independent deep clone.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, GenericListArray, RecordBatch, StringArray, StructArray, UInt32Array,
};
use arrow_schema::DataType;
use lance_core::datatypes::{BlobHandling, BlobKind, Field};
use lance_core::{Error, ROW_ADDR, Result};
use lance_io::object_store::ObjectStore;
use lance_table::format::{Fragment, Manifest};
use object_store::path::Path;
use uuid::Uuid;

use super::{
    BlobV2DescriptorColumns, field_contains_blob, join_base_and_relative_path, managed_references,
};
use crate::dataset::Dataset;
use crate::dataset::optimize::BlobV2BatchRewritePlan;

type BlobCopies = HashMap<(u32, String), (u32, String)>;

fn replace_array(copies: &BlobCopies, field: &Field, array: ArrayRef) -> Result<ArrayRef> {
    if !field_contains_blob(field) {
        return Ok(array);
    }
    if field.is_blob_v2() {
        let values = array.as_struct();
        let column = |name: &str| {
            values.column_by_name(name).ok_or_else(|| {
                Error::internal(format!("Prepared Managed rewrite is missing {name}"))
            })
        };
        let columns = BlobV2DescriptorColumns {
            descriptions: values,
            kinds: column("kind")?.as_primitive(),
            positions: column("position")?.as_primitive(),
            sizes: column("blob_size")?.as_primitive(),
            blob_ids: column("blob_id")?.as_primitive(),
            blob_uris: column("uri")?.as_string(),
        };
        let mut ids = columns.blob_ids.values().to_vec();
        let mut uris = Vec::with_capacity(values.len());
        for (row, id) in ids.iter_mut().enumerate() {
            let uri = columns.blob_uris.value(row);
            if !columns.is_null_blob(row)
                && columns.kinds.value(row) == BlobKind::Managed as u8
                && let Some(replacement) = copies.get(&(*id, uri.to_string()))
            {
                *id = replacement.0;
                uris.push(Some(replacement.1.as_str()));
            } else {
                uris.push(columns.blob_uris.is_valid(row).then_some(uri));
            }
        }
        let arrays = values
            .fields()
            .iter()
            .zip(values.columns())
            .map(|(field, array)| -> ArrayRef {
                match field.name().as_str() {
                    "blob_id" => Arc::new(UInt32Array::new(
                        ids.clone().into(),
                        columns.blob_ids.nulls().cloned(),
                    )),
                    "uri" => Arc::new(StringArray::from(uris.clone())),
                    _ => array.clone(),
                }
            })
            .collect();
        return Ok(Arc::new(StructArray::try_new(
            values.fields().clone(),
            arrays,
            values.nulls().cloned(),
        )?));
    }
    match array.data_type() {
        DataType::Struct(_) => {
            let values = array.as_struct();
            let arrays = field
                .children
                .iter()
                .zip(values.columns())
                .map(|(child, array)| replace_array(copies, child, array.clone()))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(StructArray::try_new(
                values.fields().clone(),
                arrays,
                values.nulls().cloned(),
            )?))
        }
        DataType::List(child) => {
            let values = array.as_list::<i32>();
            Ok(Arc::new(GenericListArray::<i32>::try_new(
                child.clone(),
                values.offsets().clone(),
                replace_array(copies, &field.children[0], values.values().clone())?,
                values.nulls().cloned(),
            )?))
        }
        DataType::LargeList(child) => {
            let values = array.as_list::<i64>();
            Ok(Arc::new(GenericListArray::<i64>::try_new(
                child.clone(),
                values.offsets().clone(),
                replace_array(copies, &field.children[0], values.values().clone())?,
                values.nulls().cloned(),
            )?))
        }
        datatype => Err(Error::not_supported(format!(
            "Managed replacement does not support {datatype}"
        ))),
    }
}

/// Rewrite only top-level columns that contain selected references. Fragment
/// identity, row IDs, deletion vectors, and unrelated column files survive.
async fn rewrite_blob_columns(
    copies: &BlobCopies,
    source: Arc<Dataset>,
    target: Arc<Dataset>,
) -> Result<Vec<Fragment>> {
    let fields = source
        .schema()
        .fields
        .iter()
        .filter(|field| field_contains_blob(field))
        .collect::<Vec<_>>();
    let mut updated = Vec::new();
    if copies.is_empty() || fields.is_empty() {
        return Ok(updated);
    }
    for fragment in source.get_fragments() {
        let columns = fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>();
        let write_schema = source.schema().project(&columns)?;
        let mut projection = columns.clone();
        projection.push(ROW_ADDR);
        let destination_fragment = target
            .get_fragment(fragment.id())
            .ok_or_else(|| Error::internal("Managed rewrite target fragment is missing"))?;
        let mut updater = destination_fragment
            .updater(
                Some(&projection),
                Some((write_schema, target.schema().clone())),
                Some(64),
                Some(BlobHandling::BlobsDescriptions),
            )
            .await?;
        updater.allow_external_blob_outside_bases();
        while let Some(batch) = updater.next().await? {
            let plan = BlobV2BatchRewritePlan::try_new(
                source.schema(),
                batch.schema().as_ref(),
                false,
                true,
            )?;
            let prepared = plan.transform_batch(&source, batch.clone()).await?;
            let arrays = prepared
                .schema()
                .fields()
                .iter()
                .zip(prepared.columns())
                .map(|(field, array)| {
                    let field = source
                        .schema()
                        .field(field.name())
                        .ok_or_else(|| Error::internal("Managed rewrite field is missing"))?;
                    replace_array(copies, field, array.clone())
                })
                .collect::<Result<Vec<_>>>()?;
            updater
                .update(RecordBatch::try_new(prepared.schema(), arrays)?)
                .await?;
        }
        let mut fragment = updater.finish().await?;
        let replaced = fragment
            .files
            .last()
            .ok_or_else(|| Error::internal("Managed rewrite produced no data file"))?
            .fields
            .clone();
        for file in fragment.files.iter_mut().rev().skip(1) {
            file.fields = file
                .fields
                .iter()
                .map(|id| if replaced.contains(id) { -2 } else { *id })
                .collect::<Vec<_>>()
                .into();
        }
        fragment
            .files
            .retain(|file| file.fields.iter().any(|id| *id != -2));
        updated.push(fragment);
    }
    Ok(updated)
}

pub async fn copy_blob_columns(
    source: Arc<Dataset>,
    store: Arc<ObjectStore>,
    base: Path,
    uri: &str,
    manifest: &mut Manifest,
) -> Result<()> {
    let mut target = source.as_ref().clone();
    target.object_store = store;
    target.base = base;
    target.uri = uri.to_string();
    target.base_object_stores = Default::default();
    target.manifest = Arc::new(manifest.clone());
    manifest.bind_managed_base(target.managed_default_base()?)?;
    target.manifest = Arc::new(manifest.clone());
    let target = Arc::new(target);
    let target_base = target.managed_default_base()?;
    let mut copies = HashMap::new();
    for (id, uri) in managed_references(&source, source.scan()).await? {
        let source_base = source.manifest.base_paths.get(&id).ok_or_else(|| {
            Error::invalid_input(format!("Managed clone references unknown base ID {id}"))
        })?;
        let path = join_base_and_relative_path(
            &source_base.extract_path(source.session.store_registry())?,
            &uri,
        )?;
        let target_uri = format!("_blobs/{}.blob", Uuid::new_v4());
        let target_path = join_base_and_relative_path(&target.base, &target_uri)?;
        source
            .object_store(Some(id))
            .await?
            .copy_bulk(&path, &target.object_store, &target_path)
            .await?;
        copies.insert((id, uri), (target_base.id, target_uri));
    }
    let rewritten = rewrite_blob_columns(&copies, source, target).await?;
    let fragments = Arc::make_mut(&mut manifest.fragments);
    for fragment in rewritten {
        let existing = fragments
            .iter_mut()
            .find(|entry| entry.id == fragment.id)
            .ok_or_else(|| Error::internal("Cloned fragment is missing"))?;
        *existing = fragment;
    }
    Ok(())
}
