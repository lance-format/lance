// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_io::storage::BackendWrapper;
use lance_io::utils::testing::{DelayConfig, DelayedBackend};

#[derive(Debug, Clone, Default)]
pub struct DelayedStoreWrapper {
    pub config: DelayConfig,
}

impl BackendWrapper for DelayedStoreWrapper {
    fn wrap(
        &self,
        _prefix: &str,
        original: Arc<dyn lance_io::storage::StorageBackend>,
    ) -> Arc<dyn lance_io::storage::StorageBackend> {
        Arc::new(DelayedBackend::new(original, self.config))
    }
}
