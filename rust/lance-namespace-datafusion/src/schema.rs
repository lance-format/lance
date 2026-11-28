// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use datafusion::catalog::SchemaProvider;
use datafusion::datasource::TableProvider;
use datafusion::error::Result;

use crate::error::to_datafusion_error;
use crate::namespace_level::NamespaceLevel;
use lance::datafusion::LanceTableProvider;

/// A dynamic [`SchemaProvider`] backed directly by a [`NamespaceLevel`].
///
/// Exposes Lance tables in the namespace as [`LanceTableProvider`] instances,
/// loaded on demand and cached by table name.
#[derive(Debug, Clone)]
pub struct LanceSchemaProvider {
    ns_level: NamespaceLevel,
    tables: DashMap<String, Arc<LanceTableProvider>>,
}

impl LanceSchemaProvider {
    pub async fn try_new(namespace: NamespaceLevel) -> Result<Self> {
        Ok(Self {
            ns_level: namespace,
            tables: DashMap::new(),
        })
    }

    async fn load_and_cache_table(
        &self,
        table_name: &str,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        let dataset = self
            .ns_level
            .load_dataset(table_name)
            .await
            .map_err(to_datafusion_error)?;
        let dataset = Arc::new(dataset);
        let table_provider = Arc::new(LanceTableProvider::new(dataset, false, false));
        self.tables
            .insert(table_name.to_string(), Arc::clone(&table_provider));
        Ok(Some(table_provider as Arc<dyn TableProvider>))
    }
}

#[async_trait]
impl SchemaProvider for LanceSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    async fn table(&self, table_name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        // Only the handle leaves the map here. Checking the version is a remote
        // read and reloading takes the map's write lock, so the read guard must
        // not be held across either.
        let cached = self
            .tables
            .get(table_name)
            .map(|existing| Arc::clone(existing.value()));
        let Some(existing) = cached else {
            return self.load_and_cache_table(table_name).await;
        };
        let ds = existing.dataset();
        let latest = ds.latest_version_id().await.map_err(to_datafusion_error)?;
        if latest == ds.version().version {
            return Ok(Some(existing as Arc<dyn TableProvider>));
        }
        // The cached provider fell behind: replace it.
        self.tables.remove(table_name);
        self.load_and_cache_table(table_name).await
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}
