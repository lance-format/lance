// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compose several writes into one Transaction V2 commit, each building on
//! what the earlier ones staged.

use std::sync::Arc;

use arrow_array::RecordBatch;
use lance_core::{Error, Result};
use lance_index::{IndexParams, IndexType};
use lance_table::feature_flags::transaction_v2_enabled;
use lance_table::format::{DataFile, IndexMetadata};
use lance_table::io::commit::ManifestLocation;
use lance_table::transaction::action::{
    Action, AddDataFile, AddFragment, AddIndexSegment, CompositeOperation, FieldRef, Ref, UserStep,
    provisional_fragment_id,
};
use uuid::Uuid;

use super::{CommitBuilder, InsertBuilder, ManifestWriteConfig, WriteMode, WriteParams};
use crate::Dataset;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::CreateIndexBuilder;
use crate::index::load_all_indices;
use crate::session::index_caches::IndexMetadataKey;

/// A fragment a [`StagedSnapshot`] added, before it has a committed id.
///
/// Opaque on purpose: the snapshot owns the operation's local tokens, so a
/// caller cannot name a fragment some other way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedFragment {
    token: u32,
}

/// The read version plus a [`CompositeOperation`] still being composed, and a
/// [`Dataset`] over the speculative manifest that operation would produce.
///
/// Each write consumes the snapshot and returns the next one, so a later write
/// reads everything the earlier ones staged -- an index segment can cover
/// fragments appended in the same commit. Fragments the operation adds are
/// numbered by provisional id in the speculative manifest (see
/// [`provisional_fragment_id`]); a segment built over them stores those ids,
/// and the commit records the translation to the ids it allocates.
///
/// Experimental: committing a snapshot writes Transaction V2, with the
/// compatibility consequences described on
/// [`CommitBuilder::with_experimental_composite_operations`].
pub struct StagedSnapshot {
    read: Arc<Dataset>,
    steps: Vec<UserStep>,
    next_fragment_token: u32,
    staged: Dataset,
}

impl StagedSnapshot {
    /// Start composing an operation that reads `read`.
    ///
    /// Fails unless this build permits Transaction V2 (see
    /// [`CommitBuilder::with_experimental_composite_operations`]); calling this
    /// is the caller's opt-in.
    pub async fn try_new_experimental(read: Arc<Dataset>) -> Result<Self> {
        if !transaction_v2_enabled() {
            return Err(Error::not_supported(
                "Transaction V2 is experimental and not enabled in this build, so there is no \
                 operation for a StagedSnapshot to compose",
            ));
        }
        let staged = stage(&read, &[]).await?;
        Ok(Self {
            read,
            steps: Vec::new(),
            next_fragment_token: 0,
            staged,
        })
    }

    /// The speculative dataset: the read version with every staged step
    /// applied. Not a committed version.
    pub fn dataset(&self) -> &Dataset {
        &self.staged
    }

    /// Write `batches` as new fragments.
    pub async fn append(
        mut self,
        description: impl Into<String>,
        batches: Vec<RecordBatch>,
    ) -> Result<(Self, Vec<StagedFragment>)> {
        let written = InsertBuilder::new(Arc::new(self.staged.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(batches)
            .await?;
        let Operation::Append { fragments } = written.operation else {
            return Err(Error::internal(format!(
                "an append staged {} rather than an Append",
                written.operation
            )));
        };

        let mut actions = Vec::new();
        let mut added = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            let token = self.mint_fragment_token()?;
            let physical_rows = fragment.physical_rows.ok_or_else(|| {
                Error::internal("a freshly written fragment must know its row count")
            })?;
            actions.push(Action::AddFragment(AddFragment {
                id: Ref::Local(token),
                physical_rows: physical_rows as u64,
                row_id_meta: fragment.row_id_meta,
                last_updated_at_version_meta: fragment.last_updated_at_version_meta,
                created_at_version_meta: fragment.created_at_version_meta,
                data_change: true,
            }));
            for file in fragment.files {
                let field_ids = file
                    .fields
                    .iter()
                    .copied()
                    .map(FieldRef::Committed)
                    .collect();
                actions.push(Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(token),
                    file: DataFile {
                        fields: Arc::from([]),
                        ..file
                    },
                    field_ids,
                    data_change: true,
                    base: None,
                }));
            }
            added.push(StagedFragment { token });
        }

        let snapshot = self.push(UserStep::new(description, actions)).await?;
        Ok((snapshot, added))
    }

    /// Build a segment of the index `name` on `column` over `fragments`, which
    /// this snapshot added, and add it to that index.
    ///
    /// The segment is built from the speculative dataset, so it stores the
    /// fragments' provisional ids.
    pub async fn add_index_segment(
        self,
        description: impl Into<String>,
        name: &str,
        column: &str,
        index_type: IndexType,
        params: &dyn IndexParams,
        fragments: &[StagedFragment],
    ) -> Result<Self> {
        let provisional_ids = fragments
            .iter()
            .map(|fragment| self.provisional_id(*fragment))
            .collect::<Result<Vec<_>>>()?;

        let mut staged = self.staged.clone();
        let segment = CreateIndexBuilder::new(&mut staged, &[column], index_type, params)
            .name(name.to_string())
            // The name already belongs to the index this extends; the segment
            // is added beside its segments, not in place of them.
            .replace(true)
            .fragments(provisional_ids)
            .execute_uncommitted()
            .await?;

        let action = Action::AddIndexSegment(segment_action(segment, fragments)?);
        self.push(UserStep::new(description, vec![action])).await
    }

    /// The transaction to commit, reading the snapshot's read version.
    pub fn into_transaction(self) -> Transaction {
        transaction(&self.read, self.steps)
    }

    /// Commit the composed operation onto the latest version.
    pub async fn commit(self) -> Result<Dataset> {
        let read = self.read.clone();
        CommitBuilder::new(read)
            .with_experimental_composite_operations(true)
            .execute(self.into_transaction())
            .await
    }

    fn mint_fragment_token(&mut self) -> Result<u32> {
        let token = self.next_fragment_token;
        // Checked here so a handle never names a token with no provisional id.
        provisional_fragment_id(token)?;
        self.next_fragment_token += 1;
        Ok(token)
    }

    fn provisional_id(&self, fragment: StagedFragment) -> Result<u32> {
        if fragment.token >= self.next_fragment_token {
            return Err(Error::invalid_input(format!(
                "{fragment:?} was not added by this snapshot"
            )));
        }
        provisional_fragment_id(fragment.token)
    }

    async fn push(mut self, step: UserStep) -> Result<Self> {
        self.steps.push(step);
        self.staged = stage(&self.read, &self.steps).await?;
        Ok(self)
    }
}

fn transaction(read: &Dataset, steps: Vec<UserStep>) -> Transaction {
    Transaction::new(
        read.manifest.version,
        Operation::CompositeOperation(CompositeOperation::new(steps)),
        None,
    )
}

/// The action that adds `segment`, built over the provisional ids of
/// `fragments`.
fn segment_action(segment: IndexMetadata, fragments: &[StagedFragment]) -> Result<AddIndexSegment> {
    if segment.base_id.is_some() {
        return Err(Error::internal(
            "a segment built over a staged dataset lives in the dataset's own index directory",
        ));
    }
    let keys = segment.keyed_fields().to_vec();
    Ok(AddIndexSegment {
        uuid: segment.uuid,
        name: segment.name,
        fields: keys.into_iter().map(FieldRef::Committed).collect(),
        covering_fields: segment
            .covering_fields
            .into_iter()
            .map(FieldRef::Committed)
            .collect(),
        index_details: segment.index_details,
        index_version: segment.index_version,
        covered_fragments: Some(
            fragments
                .iter()
                .map(|fragment| Ref::Local(fragment.token))
                .collect(),
        ),
        files: segment.files.unwrap_or_default(),
        base: None,
        created_at: segment.created_at,
        // The speculative manifest's version is not one anything can read;
        // the segment reflects the version the operation reads.
        dataset_version: None,
        data_change: false,
        local_fragment_addresses: true,
    })
}

/// A dataset over the speculative manifest of `steps` applied to `read`.
///
/// Its index list lives only in the session's index cache, under a manifest
/// location no committed version has, so nothing tries to read it from
/// storage and nothing committed is served it.
async fn stage(read: &Dataset, steps: &[UserStep]) -> Result<Dataset> {
    let read_indices = load_all_indices(read).await?.as_ref().clone();
    let (manifest, indices) = transaction(read, steps.to_vec()).build_staged_manifest(
        &read.manifest,
        read_indices,
        &ManifestWriteConfig::default().to_build_config(),
    )?;

    let identity = format!("staged-snapshot-{}", Uuid::new_v4());
    let location = ManifestLocation {
        version: manifest.version,
        path: read.base.clone().join("_staged").join(identity.as_str()),
        size: None,
        naming_scheme: read.manifest_location.naming_scheme,
        e_tag: Some(identity.clone()),
        identity: Some(identity),
    };
    let staged = Dataset::checkout_manifest(
        read.object_store.clone(),
        read.base.clone(),
        read.uri().to_string(),
        Arc::new(manifest),
        location,
        read.session.clone(),
        read.commit_handler.clone(),
        read.file_reader_options.clone(),
        read.store_params.as_deref().cloned(),
        read.base_store_params.clone(),
    )?;
    let key = IndexMetadataKey {
        version: staged.manifest.version,
        store_identity: &staged.object_store.store_prefix,
        e_tag: staged.manifest_location.e_tag.as_deref(),
    };
    staged
        .index_cache
        .insert_with_key(&key, Arc::new(indices))
        .await;
    Ok(staged)
}
