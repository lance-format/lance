// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub(super) use lance_table_runtime::commit::retry::{
    RetryConfig, RetryExecutor, execute_with_retry,
};

impl lance_table_runtime::commit::retry::RetrySnapshot for crate::Dataset {
    async fn checkout_latest(&mut self) -> crate::Result<()> {
        self.checkout_latest().await
    }
}
