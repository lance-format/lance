// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_io::object_store::ObjectStore;
use lance_table::format::Manifest;
use object_store::path::Path;

use super::managed::managed_references;
use crate::dataset::write::seeds::LanceWriteSeedHook;
use crate::{Dataset, Result};

pub async fn copy_blob_columns(
    source: Arc<Dataset>,
    store: Arc<ObjectStore>,
    base: Path,
    uri: &str,
    manifest: &mut Manifest,
) -> Result<()> {
    let references = managed_references(&source, source.scan(), true).await?;
    let hook = Arc::new(LanceWriteSeedHook::new(source.session.clone()));
    Box::pin(lance_table_runtime::blob::copy_blob_columns(
        &source.table_view().to_table(),
        store,
        base,
        uri,
        manifest,
        references,
        Some(hook),
    ))
    .await
}
