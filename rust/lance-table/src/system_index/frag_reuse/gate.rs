// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The write gate of a table carrying a tagged fragment reuse history.
//!
//! Every commit entrance (`CommitBuilder`, `apply_commit`, detached commits,
//! retries) funnels through `Transaction::prepare_indices`, which runs
//! [`classify`] once per attempt before anything is derived from the index
//! list (and the builder runs it again as its own chokepoint). The
//! classifier names why an operation is admitted, so that every `Operation`
//! variant has a stated verdict for every table state, and refuses with a
//! reason that says what to do instead.

use super::FRAG_REUSE_INDEX_NAME;
use super::metadata::{is_tagged, uses_tagged_fri};
use crate::format::{IndexMetadata, Manifest};
use crate::transaction::{FragReuseUpdate, Operation};
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

/// Why the gate lets an operation through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// No tagged entry: the operation is the table's ordinary business.
    Untagged,
    /// A rewrite appending transitions: the history's own maintenance.
    AppendsTransitions,
    /// A bare rewrite of fragments no index, the entry included, covers.
    RewritesUncoveredFragments,
    /// Moves no row and rewrites no column: deletion vectors, new
    /// fragments, the schema, the config, or the whole table.
    MovesNoRows,
    /// Restore copies an earlier manifest whole; it never reaches
    /// `build_manifest`, classified here so the table is complete.
    RestoresManifest,
    /// Creates, replaces or drops user indices; the entry is untouched.
    MaintainsUserIndices,
}

/// Classify `operation` against the table's current state.
///
/// `frag_reuse` is what the commit path settled about the entry for this
/// attempt; `migration_next_row_id` marks the stable row id migration.
/// `Err` refuses the commit: `InvalidInput` for a malformed intent, or
/// `NotSupported` for a shape a tagged table does not admit.
pub fn classify(
    operation: &Operation,
    current_manifest: Option<&Manifest>,
    current_indices: &[IndexMetadata],
    frag_reuse: &FragReuseUpdate,
    migration_next_row_id: Option<u64>,
) -> Result<Admission> {
    // A tagged entry may only reach the manifest through the commit path's
    // assembly, which merges the records the operation's entry adds onto
    // the CURRENT entry at every attempt and validates them; the assembly
    // rides in `FragReuseUpdate::Rewrite`. A tagged `frag_reuse_index`
    // arriving without one is a pre-assembled snapshot: it has bypassed
    // binding, conservation, folding and ledger validation, may be stale,
    // and splicing it would silently drop records a concurrent writer
    // appended.
    let appends_transitions = matches!(operation, Operation::Rewrite { .. })
        && matches!(frag_reuse, FragReuseUpdate::Rewrite(_));
    if let Operation::Rewrite {
        frag_reuse_index: Some(entry),
        ..
    } = operation
        && is_tagged(entry)
        && !appends_transitions
    {
        return Err(Error::invalid_input(
            "tagged fragment reuse entries must be assembled by the commit path, which \
             merges the records the entry adds onto the current entry; a pre-assembled \
             tagged snapshot bypasses validation and may splice away concurrent records",
        ));
    }

    let replaces_entry = matches!(
        operation,
        Operation::Rewrite {
            frag_reuse_index: Some(_),
            ..
        }
    ) && !appends_transitions;
    if !current_indices.iter().any(is_tagged) {
        // The sticky flag is the final authority on the record form
        // (`uses_tagged_fri`): once a manifest carries
        // FLAG_FRAGMENT_REUSE_INDEX every fragment reuse write is tagged,
        // even after a trim removed a fully drained entry. A v0 snapshot
        // would silently downgrade the table back to whole-history
        // replacement; the commit path converts a stale v0 intent into
        // transitions before this point (`finish_rewrite`), so only a
        // writer that skipped that conversion arrives here.
        if replaces_entry
            && current_manifest.is_some_and(|manifest| {
                uses_tagged_fri(
                    manifest,
                    current_indices
                        .iter()
                        .find(|index| index.name == FRAG_REUSE_INDEX_NAME),
                )
            })
        {
            return Err(Error::invalid_input(
                "a v0 fragment reuse snapshot cannot be published onto a table using \
                 the tagged fragment reuse format: the sticky feature flag makes the \
                 tagged format permanent, even after a drained entry is trimmed away, \
                 so the rewrite must append transitions instead",
            ));
        }
        return Ok(Admission::Untagged);
    }

    // The history's own maintenance operation.
    if appends_transitions {
        return Ok(Admission::AppendsTransitions);
    }

    // A rewrite that carries neither an entry nor transition intent is still
    // safe when no current index, the entry included (its bitmap holds the
    // recorded lineage), covers any rewritten fragment: it cannot invalidate
    // stored provenance or the lineage, and bitmap maintenance is a no-op for
    // fragments no bitmap contains. Deferred compaction of never-covered data
    // commits this shape instead of growing the history with untrimmable
    // transitions. Fragment ids beyond the row-address range can never appear
    // in an index bitmap, so they are uncovered by definition. Only histories
    // this writer understands (index_version 1) qualify: a future entry
    // version may change what its bitmap means. Empty rewrites stay refused.
    if let Operation::Rewrite {
        frag_reuse_index: None,
        rewritten_indices,
        groups,
    } = operation
        && rewritten_indices.is_empty()
        && !groups.is_empty()
        && current_indices
            .iter()
            .filter(|index| index.name == FRAG_REUSE_INDEX_NAME)
            .all(|index| matches!(index.index_version, 0 | 1))
    {
        let rewritten: RoaringBitmap = groups
            .iter()
            .flat_map(|group| group.old_fragments.iter())
            .filter_map(|fragment| u32::try_from(fragment.id).ok())
            .collect();
        // Unknown coverage is not empty coverage: without a bitmap the
        // entry could cover anything, so refuse.
        if current_indices.iter().all(|index| {
            index
                .fragment_bitmap
                .as_ref()
                .is_some_and(|bitmap| bitmap.is_disjoint(&rewritten))
        }) {
            return Ok(Admission::RewritesUncoveredFragments);
        }
    }

    // A tagged history translates row ADDRESSES; stable row ids replace
    // them, so the migration (a `Merge` committed with
    // `migration_next_row_id`) is refused while an entry exists. The other
    // direction is refused where a transition is recorded.
    if migration_next_row_id.is_some() {
        return Err(Error::not_supported(
            "stable row id migration is not supported on a table with a tagged fragment \
             reuse history: tagged histories translate row addresses, which stable row ids \
             replace. Trim the history first",
        ));
    }

    // An in-place column rewrite (an update with `fields_modified`, that is
    // `MergeInsertWriteMode::RewriteColumns`; a merge that swaps a column's
    // data file, as `alter_columns` does; a data replacement) is admitted:
    // the commit preparation withdraws every affected segment's
    // contribution (`Transaction::withdraw_in_place_rewrites`, fed by
    // `Transaction::rewritten_physical_columns`), a whole transition's
    // sources for a translating segment, so the reader scans those rows
    // until `optimize_indices` rebuilds them.
    // Creating, replacing or dropping user indices leaves the tagged entry
    // untouched (it is carried through unchanged). An index built on a
    // tagged table covers live fragments directly; one built before a
    // rewrite and rebased across it by the conflict resolver lands with its
    // provenance and translates. Only a `CreateIndex` touching the entry
    // itself has no tagged semantics here.
    if let Operation::CreateIndex {
        new_indices,
        removed_indices,
        ..
    } = operation
        && new_indices
            .iter()
            .chain(removed_indices.iter())
            .all(|idx| idx.name != FRAG_REUSE_INDEX_NAME)
    {
        return Ok(Admission::MaintainsUserIndices);
    }

    match operation {
        // Deletion vectors, new fragments, the config, the schema
        // (`retain_relevant_indices` still drops an index on a removed
        // field) or the whole table (Overwrite discards every index, the
        // entry included; the sticky flag keeps the tagged record form).
        Operation::Append { .. }
        | Operation::ReserveFragments { .. }
        | Operation::Delete { .. }
        | Operation::Update { .. }
        | Operation::UpdateConfig { .. }
        | Operation::Overwrite { .. }
        | Operation::Project { .. }
        | Operation::Merge { .. }
        | Operation::DataReplacement { .. } => Ok(Admission::MovesNoRows),
        Operation::Restore { .. } => Ok(Admission::RestoresManifest),
        // A bare rewrite of covered fragments or a v0 snapshot would
        // misinterpret the history; MemWAL state, overlays, clones and base
        // changes have no tagged semantics yet. A CreateIndex reaching here
        // touches the entry itself.
        Operation::Rewrite { .. }
        | Operation::CreateIndex { .. }
        | Operation::UpdateMemWalState { .. }
        | Operation::DataOverlay { .. }
        | Operation::Clone { .. }
        | Operation::UpdateBases { .. } => Err(Error::not_supported(
            "Tagged FRI history maintenance is not implemented for this operation; upgrade to a writer supporting tagged histories",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    use crate::format::{BasePath, DataFile, Fragment};
    use crate::transaction::test_support::{
        overlay_with_field, sample_index_metadata, sample_manifest,
    };
    use crate::transaction::{DataOverlayGroup, DataReplacementGroup, RewriteGroup};
    use lance_file::version::ConcreteFileVersion;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// What the table looks like: fragment 0 carries field 0 in `a.lance`;
    /// a tagged table also has the entry (lineage {0, 5}: fragment 5 was
    /// rewritten into fragment 0) and, optionally, a user index on field 0.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Table {
        Untagged,
        Tagged,
        /// A segment built from fragment 5, translating to fragment 0.
        TaggedWithTranslatingIndex,
        /// A segment built on fragment 0 directly.
        TaggedWithDirectIndex,
    }

    fn table(state: Table) -> (Manifest, Vec<IndexMetadata>) {
        let mut manifest = sample_manifest();
        manifest.fragments = Arc::new(vec![Fragment::new(0).with_file(
            "a.lance",
            vec![0],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
        )]);
        let mut indices = Vec::new();
        if state != Table::Untagged {
            manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
            manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
            let mut entry = sample_index_metadata(FRAG_REUSE_INDEX_NAME);
            entry.fields.clear();
            entry.fragment_bitmap = Some([0u32, 5].into_iter().collect());
            indices.push(entry);
        }
        match state {
            Table::TaggedWithTranslatingIndex => {
                let mut segment = sample_index_metadata("id_idx");
                segment.fragment_bitmap = Some([5u32].into_iter().collect());
                indices.push(segment);
            }
            Table::TaggedWithDirectIndex => indices.push(sample_index_metadata("id_idx")),
            _ => {}
        }
        (manifest, indices)
    }

    fn update(fields_modified: Vec<u32>) -> Operation {
        let mut update = crate::transaction::test_support::update_txn(vec![]);
        let Operation::Update {
            updated_fragments,
            fields_modified: modified,
            ..
        } = &mut update.operation
        else {
            unreachable!()
        };
        *updated_fragments = vec![Fragment::new(0)];
        *modified = fields_modified;
        update.operation
    }

    fn rewrite(old: u64, frag_reuse_index: Option<IndexMetadata>) -> Operation {
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![Fragment::new(old)],
                new_fragments: vec![Fragment::new(10)],
            }],
            rewritten_indices: vec![],
            frag_reuse_index,
        }
    }

    /// The tagged entry a rewrite appending transitions carries: the same
    /// entry the commit path's assembly puts in the config for that case.
    fn appended_entry() -> IndexMetadata {
        let mut entry = sample_index_metadata(FRAG_REUSE_INDEX_NAME);
        entry.fields.clear();
        entry
    }

    /// What the commit path settled for the attempt:
    /// `rewrite_appends_transitions` is the one shape whose verdict depends
    /// on the commit path having assembled the entry.
    fn frag_reuse_for(kind: &str) -> FragReuseUpdate {
        if kind == "rewrite_appends_transitions" {
            FragReuseUpdate::Rewrite(std::sync::Arc::new(
                crate::transaction::TaggedRewriteAssembly {
                    entry: appended_entry(),
                    base_entry_version: None,
                    reordered_sources: RoaringBitmap::new(),
                },
            ))
        } else {
            FragReuseUpdate::None
        }
    }

    /// One sample per `Operation` variant (several for the variants whose
    /// verdict depends on their payload).
    fn sample(kind: &str, manifest: &Manifest) -> Operation {
        match kind {
            "append" => Operation::Append {
                fragments: vec![Fragment::new(1)],
            },
            "delete" => Operation::Delete {
                updated_fragments: vec![],
                deleted_fragment_ids: vec![0],
                predicate: "true".into(),
            },
            "overwrite" => Operation::Overwrite {
                fragments: vec![Fragment::new(0)],
                schema: manifest.schema.clone(),
                config_upsert_values: None,
                initial_bases: None,
            },
            "create_index" => Operation::CreateIndex {
                new_indices: vec![sample_index_metadata("new_idx")],
                removed_indices: vec![],
            },
            "rewrite_bare_covered" => rewrite(0, None),
            "rewrite_bare_uncovered" => rewrite(7, None),
            "rewrite_appends_transitions" => rewrite(0, Some(appended_entry())),
            "rewrite_v0_snapshot" => {
                let mut entry = appended_entry();
                entry.index_version = 0;
                rewrite(0, Some(entry))
            }
            "rewrite_tagged_snapshot" => rewrite(0, Some(appended_entry())),
            "data_replacement" => Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    0,
                    DataFile::new(
                        "b.lance",
                        vec![0],
                        vec![0],
                        ConcreteFileVersion::V2_0,
                        None,
                        None,
                    ),
                )],
            },
            "data_overlay" => Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id: 0,
                    overlays: vec![overlay_with_field(0, 1)],
                }],
            },
            "merge_new_column" => Operation::Merge {
                fragments: manifest.fragments.as_ref().clone(),
                schema: manifest.schema.clone(),
                preserves_nullability: true,
            },
            "merge_rewrites_column" => Operation::Merge {
                fragments: vec![Fragment::new(0).with_file(
                    "b.lance",
                    vec![0],
                    vec![0],
                    ConcreteFileVersion::V2_0,
                    None,
                )],
                schema: manifest.schema.clone(),
                preserves_nullability: true,
            },
            "restore" => Operation::Restore { version: 1 },
            "reserve_fragments" => Operation::ReserveFragments { num_fragments: 2 },
            "update_rewrites_rows" => update(vec![]),
            "update_rewrites_column" => update(vec![0]),
            "update_rewrites_unindexed_column" => update(vec![1]),
            "project" => Operation::Project {
                schema: manifest.schema.clone(),
                preserves_nullability: true,
            },
            "update_config" => Operation::UpdateConfig {
                config_updates: None,
                table_metadata_updates: None,
                schema_metadata_updates: None,
                field_metadata_updates: HashMap::new(),
            },
            "update_mem_wal_state" => Operation::UpdateMemWalState {
                compacted_sstables: vec![],
            },
            "clone" => Operation::Clone {
                is_shallow: true,
                ref_name: None,
                ref_version: 1,
                ref_path: "memory://source".into(),
                branch_name: None,
            },
            "update_bases" => Operation::UpdateBases {
                new_bases: vec![BasePath {
                    id: 1,
                    name: None,
                    is_dataset_root: false,
                    path: "memory://base".into(),
                }],
            },
            _ => unreachable!("{kind}"),
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Verdict {
        Admit(Admission),
        NotSupported,
        InvalidInput,
    }

    fn verdict(result: Result<Admission>) -> Verdict {
        match result {
            Ok(admission) => Verdict::Admit(admission),
            Err(Error::NotSupported { .. }) => Verdict::NotSupported,
            Err(Error::InvalidInput { .. }) => Verdict::InvalidInput,
            Err(other) => panic!("unexpected error kind: {other}"),
        }
    }

    /// Every `Operation` variant against every table state. A new variant
    /// must add its rows here; a new state must add its column.
    #[rstest::rstest]
    // Untagged tables: everything is ordinary business, except a tagged
    // snapshot, which is malformed anywhere.
    #[case("append", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case("delete", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case("overwrite", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case("create_index", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case(
        "rewrite_bare_covered",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case(
        "rewrite_v0_snapshot",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case("rewrite_tagged_snapshot", Table::Untagged, Verdict::InvalidInput)]
    #[case(
        "data_replacement",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case("data_overlay", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case(
        "merge_rewrites_column",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case("restore", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case(
        "reserve_fragments",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case(
        "update_rewrites_column",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case("project", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case("update_config", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case(
        "update_mem_wal_state",
        Table::Untagged,
        Verdict::Admit(Admission::Untagged)
    )]
    #[case("clone", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    #[case("update_bases", Table::Untagged, Verdict::Admit(Admission::Untagged))]
    // Tagged tables.
    #[case("append", Table::Tagged, Verdict::Admit(Admission::MovesNoRows))]
    #[case("delete", Table::Tagged, Verdict::Admit(Admission::MovesNoRows))]
    #[case("overwrite", Table::Tagged, Verdict::Admit(Admission::MovesNoRows))]
    #[case(
        "create_index",
        Table::Tagged,
        Verdict::Admit(Admission::MaintainsUserIndices)
    )]
    #[case("rewrite_bare_covered", Table::Tagged, Verdict::NotSupported)]
    #[case(
        "rewrite_bare_uncovered",
        Table::Tagged,
        Verdict::Admit(Admission::RewritesUncoveredFragments)
    )]
    #[case(
        "rewrite_appends_transitions",
        Table::Tagged,
        Verdict::Admit(Admission::AppendsTransitions)
    )]
    #[case("rewrite_v0_snapshot", Table::Tagged, Verdict::NotSupported)]
    #[case("rewrite_tagged_snapshot", Table::Tagged, Verdict::InvalidInput)]
    #[case(
        "data_replacement",
        Table::Tagged,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "data_replacement",
        Table::TaggedWithDirectIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "data_replacement",
        Table::TaggedWithTranslatingIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case("data_overlay", Table::Tagged, Verdict::NotSupported)]
    #[case(
        "merge_new_column",
        Table::Tagged,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "merge_rewrites_column",
        Table::TaggedWithDirectIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "merge_rewrites_column",
        Table::TaggedWithTranslatingIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case("restore", Table::Tagged, Verdict::Admit(Admission::RestoresManifest))]
    #[case(
        "reserve_fragments",
        Table::Tagged,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "update_rewrites_rows",
        Table::TaggedWithTranslatingIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "update_rewrites_column",
        Table::TaggedWithDirectIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "update_rewrites_column",
        Table::TaggedWithTranslatingIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case(
        "update_rewrites_unindexed_column",
        Table::TaggedWithTranslatingIndex,
        Verdict::Admit(Admission::MovesNoRows)
    )]
    #[case("project", Table::Tagged, Verdict::Admit(Admission::MovesNoRows))]
    #[case("update_config", Table::Tagged, Verdict::Admit(Admission::MovesNoRows))]
    #[case("update_mem_wal_state", Table::Tagged, Verdict::NotSupported)]
    #[case("clone", Table::Tagged, Verdict::NotSupported)]
    #[case("update_bases", Table::Tagged, Verdict::NotSupported)]
    fn every_operation_has_a_verdict(
        #[case] kind: &str,
        #[case] state: Table,
        #[case] expected: Verdict,
    ) {
        let (manifest, indices) = table(state);
        let operation = sample(kind, &manifest);
        let got = verdict(classify(
            &operation,
            Some(&manifest),
            &indices,
            &frag_reuse_for(kind),
            None,
        ));
        assert_eq!(got, expected, "{kind} on {state:?}");
    }

    /// The stable row id migration is a `Merge` with the activation marker.
    #[test]
    fn migration_is_refused_on_a_tagged_table() {
        let (manifest, indices) = table(Table::Tagged);
        let error = classify(
            &sample("merge_new_column", &manifest),
            Some(&manifest),
            &indices,
            &FragReuseUpdate::None,
            Some(100),
        )
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().contains("stable row id migration"),
            "{error}"
        );
    }
}
