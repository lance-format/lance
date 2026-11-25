// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve ordinary sidecars without materializing their payloads. Task-local
//! IDs let destination writers stream live ranges and copy dedicated objects.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::Field;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

use super::locate::{BlobLocator, StoredBlob};
use crate::Table;
use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange};

#[derive(Clone)]
pub(super) struct Source {
    pub(super) store: Arc<ObjectStore>,
    pub(super) path: Path,
}

#[derive(Default)]
struct Sources {
    ids: HashMap<(Option<u32>, Path), u32>,
    objects: HashMap<u32, Source>,
}

#[derive(Default)]
pub struct SidecarRewrite(Mutex<Sources>);

impl SidecarRewrite {
    fn register(&self, base: Option<u32>, source: Source) -> Result<u32> {
        let mut sources = self
            .0
            .lock()
            .map_err(|_| Error::internal("Poisoned sidecar copy map"))?;
        let key = (base, source.path.clone());
        if let Some(id) = sources.ids.get(&key) {
            return Ok(*id);
        }
        let id = sources
            .ids
            .len()
            .checked_add(1)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Error::invalid_input("Too many sidecar objects in compaction task"))?;
        sources.ids.insert(key, id);
        sources.objects.insert(id, source);
        Ok(id)
    }

    pub(super) async fn prepare(
        &self,
        dataset: &Table,
        field_id: u32,
        values: &StructArray,
        rows: &[u64],
        field: &Field,
    ) -> Result<ArrayRef> {
        if values.len() != rows.len() {
            return Err(Error::internal(
                "Blob descriptor and source row counts differ",
            ));
        }
        let mut locator = BlobLocator::new(dataset, field_id);
        let mut builder = BlobDescriptorArrayBuilder::new_with_metadata(
            field.name(),
            field.is_nullable(),
            field.metadata().clone(),
        );
        for (row, row_addr) in rows.iter().copied().enumerate() {
            let Some(blob) = StoredBlob::parse(values, row)? else {
                builder.push_null()?;
                continue;
            };
            match blob {
                StoredBlob::Packed { blob_id, .. } | StoredBlob::Dedicated { blob_id, .. } => {
                    let base_id = locator.sidecar(row_addr, blob_id)?.0;
                    let file = locator.locate(&blob, row_addr).await?;
                    let id = self.register(
                        base_id,
                        Source {
                            store: file.object_store().clone(),
                            path: file.data_path().clone(),
                        },
                    )?;
                    if matches!(blob, StoredBlob::Dedicated { .. }) {
                        builder.push_dedicated(id, file.size())?;
                    } else {
                        builder.push_packed(
                            id,
                            BlobRange {
                                offset: file.position(),
                                size: file.size(),
                            },
                        )?;
                    }
                }
                StoredBlob::Empty | StoredBlob::Inline { .. } => {
                    let file = locator.locate(&blob, row_addr).await?;
                    builder.push_inline(file.read().await?)?;
                }
                StoredBlob::External {
                    base_id,
                    uri,
                    position,
                    size,
                } => builder.push(BlobDescriptor::External {
                    base_id,
                    uri: uri.to_string(),
                    offset: position,
                    size,
                })?,
                StoredBlob::Managed { base_id, .. } => {
                    // A clone can retain Managed descriptors after the schema's
                    // write opt-in is removed. Stream those ranges into ordinary
                    // packs rather than changing the write capability.
                    let base = locator.managed_base(base_id, row_addr)?;
                    let file = locator.locate(&blob, row_addr).await?;
                    let id = self.register(
                        base,
                        Source {
                            store: file.object_store().clone(),
                            path: file.data_path().clone(),
                        },
                    )?;
                    builder.push_packed(
                        id,
                        BlobRange {
                            offset: file.position(),
                            size: file.size(),
                        },
                    )?;
                }
            }
        }
        Ok(builder.finish()?.into_parts().1)
    }

    pub(super) fn source(&self, id: u32) -> Result<Source> {
        self.0
            .lock()
            .map_err(|_| Error::internal("Poisoned sidecar copy map"))?
            .objects
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::internal(format!("Missing source for compacted sidecar {id}")))
    }
}
