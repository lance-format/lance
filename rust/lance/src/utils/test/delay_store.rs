// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_io::object_store::WrappingObjectStore;
use lance_io::utils::testing::{DelayConfig, DelayedBackend};

#[derive(Debug, Clone, Default)]
pub struct DelayedStoreWrapper {
    pub config: DelayConfig,
}

impl WrappingObjectStore for DelayedStoreWrapper {
    fn wrap(
        &self,
        _prefix: &str,
        original: Arc<dyn lance_io::object_store::StorageBackend>,
    ) -> Arc<dyn lance_io::object_store::StorageBackend> {
        Arc::new(DelayedBackend::new(original, self.config))
    }
}
