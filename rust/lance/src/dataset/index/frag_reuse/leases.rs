// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading what the listed retention leases require of the fragment reuse
//! history, for compaction, trimming and the rewrite commit check. The rules
//! themselves live in [`lance_table::system_index::frag_reuse::leases`].

use std::collections::BTreeSet;

use lance_core::{Error, Result};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::leases::{
    LeaseRequirement, LeasedFragments, RecordLineage,
};
use log::warn;
use roaring::RoaringBitmap;

use crate::Dataset;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::frag_reuse::load_frag_reuse_records;
use crate::index::loading::load_all_indices;

/// The oldest version a listed lease retains.
pub fn oldest_leased_version(dataset: &Dataset) -> Option<u64> {
    dataset
        .manifest
        .retention_leases
        .iter()
        .map(|lease| lease.version())
        .min()
}

/// The fragment ids of every version `dataset`'s listed leases retain, read
/// from the leased versions' manifests on the current branch.
///
/// Cleanup retains those manifests while the leases are listed, so failing to
/// read one means the table was cleaned by a writer that broke that contract
/// or damaged externally. The leased fragments are then unknown, and every
/// record is treated as required rather than guessing.
pub async fn leased_fragments(dataset: &Dataset) -> LeasedFragments {
    let versions: BTreeSet<u64> = dataset
        .manifest
        .retention_leases
        .iter()
        .map(|lease| lease.version())
        .collect();
    let mut fragments = RoaringBitmap::new();
    for version in versions {
        let leased = if version == dataset.manifest.version {
            Ok(dataset.fragment_bitmap.as_ref().clone())
        } else {
            version_fragments(dataset, version).await
        };
        match leased {
            Ok(leased) => fragments |= leased,
            Err(error) => {
                warn!(
                    "cannot read leased version {version} of {}; keeping every fragment reuse \
                     record it might need: {error}",
                    dataset.uri
                );
                return LeasedFragments::Unknown;
            }
        }
    }
    LeasedFragments::Known(fragments)
}

async fn version_fragments(dataset: &Dataset, version: u64) -> Result<RoaringBitmap> {
    let location = dataset
        .commit_handler
        .resolve_version_location(&dataset.base, version, &dataset.object_store.inner)
        .await?;
    let manifest = Dataset::get_manifest(
        dataset.object_store.as_ref(),
        &location,
        &dataset.uri,
        dataset.session.as_ref(),
    )
    .await?;
    Ok(manifest
        .fragments
        .iter()
        .filter_map(|fragment| u32::try_from(fragment.id).ok())
        .collect())
}

/// The records of a fragment reuse entry as lineage, in the order the entry
/// stores them: legacy versions first, then transitions.
pub async fn record_lineages(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<Vec<RecordLineage>> {
    let records = load_frag_reuse_records(dataset, entry).await?;
    Ok(records
        .legacy_versions
        .iter()
        .map(RecordLineage::from)
        .chain(records.transitions.iter().map(RecordLineage::from))
        .collect())
}

/// What `dataset`'s listed leases require of its current fragment reuse
/// history, or `None` when no lease is listed.
pub async fn lease_requirement(dataset: &Dataset) -> Result<Option<LeaseRequirement>> {
    if dataset.manifest.retention_leases.is_empty() {
        return Ok(None);
    }
    let leased = leased_fragments(dataset).await;
    let records = match load_all_indices(dataset)
        .await?
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
    {
        Some(entry) => record_lineages(dataset, entry).await?,
        None => Vec::new(),
    };
    Ok(Some(LeaseRequirement::compute(&leased, &records)))
}

/// Refuse a rewrite that replaces fragments a listed lease requires without
/// recording fragment reuse transitions for them, checked against the
/// attempt's predecessor `dataset` so that leases acquired concurrently are
/// honored.
///
/// The conflict is retryable: replanning the compaction against the new head
/// records the transitions.
pub async fn ensure_rewrite_records_lease_transitions(
    dataset: &Dataset,
    transaction: &Transaction,
) -> Result<()> {
    let Operation::Rewrite {
        groups,
        frag_reuse_index,
        ..
    } = &transaction.operation
    else {
        return Ok(());
    };
    let Some(requirement) = lease_requirement(dataset).await? else {
        return Ok(());
    };
    let required: Vec<u64> = groups
        .iter()
        .flat_map(|group| group.old_fragments.iter().map(|fragment| fragment.id))
        .filter(|id| requirement.requires_sources([*id]))
        .collect();
    if required.is_empty() {
        return Ok(());
    }
    let recorded: BTreeSet<u64> = match frag_reuse_index {
        Some(entry) => record_lineages(dataset, entry)
            .await?
            .iter()
            .flat_map(|record| record.sources().collect::<Vec<_>>())
            .collect(),
        None => BTreeSet::new(),
    };
    let missing: Vec<u64> = required
        .into_iter()
        .filter(|id| !recorded.contains(id))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(Error::retryable_commit_conflict_source(
        dataset.manifest.version,
        format!(
            "transaction {} rewrites fragments {missing:?}, which a listed retention lease \
             requires, without recording fragment reuse transitions for them; replan the \
             rewrite against version {} so it records them",
            transaction.uuid, dataset.manifest.version
        )
        .into(),
    ))
}
