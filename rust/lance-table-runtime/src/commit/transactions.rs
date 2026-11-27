// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_table::transaction::Transaction;

use crate::Table;
use crate::caches::ManifestTransactionKey;

struct NewTransactionResult<'a> {
    dataset: BoxFuture<'a, Result<Table>>,
    new_transactions: BoxStream<'a, Result<(u64, Arc<Transaction>)>>,
}

#[derive(Clone, Copy)]
pub enum TransactionReadSource {
    SessionCache,
    /// Read the transaction from its manifest or the file it references.
    /// A version-only cache entry may belong to a previous table at the URI.
    Storage,
}

fn load_new_transactions(
    dataset: &Table,
    source: TransactionReadSource,
) -> NewTransactionResult<'_> {
    // Resolve every manifest with version > our current version (the latest plus
    // the ones in between). On non-lexically-ordered stores this uses the version
    // hint to avoid an O(n) listing.
    let io_parallelism = dataset.state.object_store.as_ref().io_parallelism();
    // Both stages must poll their input even when the store permits only one I/O.
    let stage_parallelism = (io_parallelism / 2).max(1);
    let locations = dataset.state.commit_handler.list_manifest_locations_since(
        &dataset.state.base,
        dataset.state.object_store.as_ref(),
        dataset.manifest.version,
    );

    // Will send the latest manifest via a channel.
    let (latest_tx, latest_rx) = tokio::sync::oneshot::channel();
    let mut latest_tx = Some(latest_tx);

    let manifests = locations
        .map_ok(move |location| {
            let latest_tx = latest_tx.take();
            async move {
                let manifest = match source {
                    TransactionReadSource::SessionCache => {
                        Table::get_manifest(
                            dataset.state.object_store.as_ref(),
                            &location,
                            &dataset.state.uri,
                            dataset.state.session.as_ref(),
                        )
                        .await?
                    }
                    TransactionReadSource::Storage => Arc::new(
                        Table::load_manifest(
                            dataset.state.object_store.as_ref(),
                            &location,
                            &dataset.state.uri,
                            dataset.state.session.as_ref(),
                        )
                        .await?,
                    ),
                };

                if let Some(latest_tx) = latest_tx {
                    // We ignore the error, since we don't care if the receiver is dropped.
                    let _ = latest_tx.send((manifest.clone(), location.clone()));
                }

                Ok((manifest, location))
            }
        })
        .try_buffer_unordered(stage_parallelism);
    let transactions = manifests
        .map_ok(move |(manifest, location)| async move {
            let version = manifest.version;
            let missing_transaction = || {
                Error::internal(format!(
                    "Dataset version {version} does not have a transaction file"
                ))
            };
            if let TransactionReadSource::Storage = source {
                let transaction = dataset
                    .state
                    .read_transaction_from_storage(&manifest, &location)
                    .await?
                    .ok_or_else(missing_transaction)?;
                return Ok((version, Arc::new(transaction)));
            }

            let tx_key = ManifestTransactionKey::new(&dataset.state.object_store, &location);
            let cached = if let Some(key) = &tx_key {
                dataset.state.metadata_cache.get_with_key(key).await
            } else {
                None
            };
            let transaction = if let Some(cached) = cached {
                cached
            } else {
                let mut dataset_version = dataset.clone();
                dataset_version.set_manifest(manifest, location.clone());
                let loaded = Arc::new(
                    dataset_version
                        .read_transaction()
                        .await?
                        .ok_or_else(missing_transaction)?,
                );
                if let Some(key) = &tx_key {
                    dataset
                        .state
                        .metadata_cache
                        .insert_with_key(key, loaded.clone())
                        .await;
                }
                loaded
            };
            Ok((version, transaction))
        })
        .try_buffer_unordered(stage_parallelism);

    let dataset = async move {
        if let Ok((latest_manifest, location)) = latest_rx.await {
            // Advancing a retry must preserve the handle's branch incarnation.
            let mut latest = dataset.clone();
            latest.set_manifest(latest_manifest, location);
            Ok(latest)
        } else {
            // If we didn't get the latest manifest, we can still return the dataset
            // with the current manifest.
            Ok(dataset.clone())
        }
    }
    .boxed();

    let new_transactions = transactions.boxed();

    NewTransactionResult {
        dataset,
        new_transactions,
    }
}

pub async fn load_and_sort_new_transactions(
    dataset: &Table,
    source: TransactionReadSource,
) -> Result<(Table, Vec<(u64, Arc<Transaction>)>)> {
    let NewTransactionResult {
        dataset: new_ds,
        new_transactions,
    } = load_new_transactions(dataset, source);
    let new_transactions = new_transactions.try_collect::<Vec<_>>();
    let (new_ds, mut txns) = futures::future::try_join(new_ds, new_transactions).await?;
    txns.sort_by_key(|(version, _)| *version);
    Ok((new_ds, txns))
}
