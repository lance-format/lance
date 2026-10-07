// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Commit-scoped leaf reads retained on local scratch storage. The materializer
//! reuses these reads, including untouched coalesce neighbors. Native validation
//! can separately retain the complete Fragment set for global operations.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use lance_core::Result;
use lance_core::utils::tempfile::TempDir;
use prost::Message;
use tokio::sync::{Mutex, OnceCell};
use uuid::Uuid;

use super::{LeafKey, NodeStore};
use crate::format::{Fragment, pb};

pub(super) struct ValidationReads {
    directory: TempDir,
    leaves: Mutex<BTreeMap<LeafKey, Arc<OnceCell<PathBuf>>>>,
}

impl ValidationReads {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            directory: TempDir::try_new()?,
            leaves: Mutex::new(BTreeMap::new()),
        })
    }

    /// The retained read of `child` for this commit, empty until a reader loads it.
    pub(super) async fn cell(
        &self,
        store: &NodeStore,
        child: &pb::FragmentTreeChild,
    ) -> Result<Arc<OnceCell<PathBuf>>> {
        Ok(self
            .leaves
            .lock()
            .await
            .entry(LeafKey::new(store, child)?)
            .or_default()
            .clone())
    }

    /// A leaf through its retained read, loading it with `load` only when no
    /// reader of this commit has retained it yet.
    pub(super) async fn read(
        &self,
        cell: &OnceCell<PathBuf>,
        load: impl Future<Output = Result<Vec<Fragment>>>,
    ) -> Result<Vec<Fragment>> {
        let mut loaded = None;
        let path = cell
            .get_or_try_init(|| async {
                let fragments = load.await?;
                let record = pb::Manifest {
                    fragments: fragments.iter().map(Into::into).collect(),
                    ..Default::default()
                };
                let path = self
                    .directory
                    .std_path()
                    .join(format!("{}.pb", Uuid::new_v4()));
                tokio::fs::write(&path, record.encode_to_vec()).await?;
                loaded = Some(fragments);
                Ok::<_, lance_core::Error>(path)
            })
            .await?;
        if let Some(fragments) = loaded {
            return Ok(fragments);
        }
        let bytes = tokio::fs::read(path).await?;
        let record = pb::Manifest::decode(bytes.as_slice())?;
        record
            .fragments
            .into_iter()
            .map(crate::fragment_metadata::validation::fragment)
            .collect()
    }
}
