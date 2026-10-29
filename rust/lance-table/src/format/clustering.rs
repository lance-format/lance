// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Runtime form of the manifest's clustering definitions.
//!
//! A definition is immutable once allocated: fragments and index segments
//! reference it by `clustering_id`, and the provider-specific layout
//! description lives under provider-owned keys in `Manifest::table_metadata`.
//! Lance carries the definitions and the references; it never interprets them.

use std::collections::{HashMap, HashSet};

use lance_core::datatypes::Schema;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

use crate::format::pb;
use crate::format::{Fragment, IndexMetadata, Manifest};

/// An immutable clustering definition referenced by fragment and index
/// segment `clustering_id`.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ClusteringDefinition {
    /// Non-zero table-local ID allocated through `Manifest::max_clustering_id`.
    pub id: u32,
    /// Ordered, non-empty list of distinct schema field IDs the layout is keyed on.
    pub columns: Vec<i32>,
    /// Non-empty, case-sensitive clustering implementation name.
    pub provider: String,
    /// Non-zero layout/configuration identifier owned by the provider. It is
    /// neither a dataset version nor a software version.
    pub version: u64,
}

impl ClusteringDefinition {
    /// Check the invariants every definition must hold, wherever it came from.
    pub fn validate(&self) -> Result<()> {
        if self.id == 0 {
            return Err(Error::invalid_input(
                "clustering definition has id 0; ids are allocated from 1",
            ));
        }
        if self.columns.is_empty() {
            return Err(Error::invalid_input(format!(
                "clustering definition {} has no columns",
                self.id
            )));
        }
        let mut seen = HashSet::with_capacity(self.columns.len());
        if let Some(duplicate) = self.columns.iter().find(|column| !seen.insert(**column)) {
            return Err(Error::invalid_input(format!(
                "clustering definition {} repeats field id {} in {:?}",
                self.id, duplicate, self.columns
            )));
        }
        if self.provider.is_empty() {
            return Err(Error::invalid_input(format!(
                "clustering definition {} has an empty provider",
                self.id
            )));
        }
        if self.version == 0 {
            return Err(Error::invalid_input(format!(
                "clustering definition {} (provider {}) has version 0",
                self.id, self.provider
            )));
        }
        Ok(())
    }
}

impl TryFrom<pb::Clustering> for ClusteringDefinition {
    type Error = Error;

    fn try_from(p: pb::Clustering) -> Result<Self> {
        let definition = Self {
            id: p.id,
            columns: p.columns,
            provider: p.provider,
            version: p.version,
        };
        definition
            .validate()
            .map_err(|error| Error::corrupt_file_named("manifest", error.to_string()))?;
        Ok(definition)
    }
}

/// Check the clustering invariants of a manifest and its index section.
///
/// Definitions are well formed with unique ids at or below the high-water
/// mark, and every fragment or index segment reference named in
/// `references` resolves to a definition in the manifest. A definition's key
/// columns are not checked against the current schema: a column may have been
/// dropped after the definition was allocated, and the objects still
/// referencing it are not thereby invalid.
///
/// `references` carries only what the commit newly introduced or changed. A
/// definition can only be removed while nothing references it (see
/// [`apply_complete_clustering_definitions`]), so a reference that resolved
/// against a valid base manifest still resolves against the manifest built
/// from it; nothing else has to be revisited. Callers that have no such base,
/// or that want the whole table audited, use [`validate_clustering`].
pub fn validate_clustering_of(manifest: &Manifest, references: &ChangedReferences) -> Result<()> {
    let ids = definition_ids(manifest)?;
    for (fragment, id) in &references.fragments {
        if !ids.contains(id) {
            return Err(Error::invalid_input(format!(
                "fragment {fragment} references clustering definition {id}, which the manifest does not define"
            )));
        }
    }
    for (segment, name, id) in &references.index_segments {
        if !ids.contains(id) {
            return Err(Error::invalid_input(format!(
                "index segment {segment} ({name}) references clustering definition {id}, which the manifest does not define"
            )));
        }
    }
    Ok(())
}

/// The clustering labels a commit introduced or changed.
///
/// Accumulated where the builder forms the candidate's object lists, so its
/// size is proportional to what the operation touched and not to the table: a
/// metadata-only commit contributes nothing and costs no per-object work.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ChangedReferences {
    fragments: Vec<(u64, u32)>,
    index_segments: Vec<(String, String, u32)>,
    /// Every clustering-specific object visit, including accumulation. A test
    /// asserts on this: a counter that only counted survivors of a filter
    /// could not witness a traversal it was downstream of.
    visits: usize,
}

impl ChangedReferences {
    /// Note a fragment whose label this commit introduced or changed.
    pub fn fragment(&mut self, fragment: &Fragment) {
        self.visits += 1;
        if let Some(id) = fragment.clustering_id {
            self.fragments.push((fragment.id, id));
        }
    }

    /// Note an index segment whose label this commit introduced or changed.
    pub fn index_segment(&mut self, segment: &IndexMetadata) {
        self.visits += 1;
        if let Some(id) = segment.clustering_id {
            self.index_segments
                .push((segment.uuid.to_string(), segment.name.clone(), id));
        }
    }

    #[cfg(test)]
    pub fn visits(&self) -> usize {
        self.visits
    }
}

/// The definition set and watermark checks, which are O(definitions) and run
/// on every commit regardless of scope.
fn definition_ids(manifest: &Manifest) -> Result<HashSet<u32>> {
    let mut ids = HashSet::with_capacity(manifest.clustering.len());
    for definition in &manifest.clustering {
        definition.validate()?;
        if definition.id > manifest.max_clustering_id {
            return Err(Error::invalid_input(format!(
                "clustering definition {} is above max_clustering_id {}",
                definition.id, manifest.max_clustering_id
            )));
        }
        if !ids.insert(definition.id) {
            return Err(Error::invalid_input(format!(
                "clustering definition {} is defined more than once",
                definition.id
            )));
        }
    }
    Ok(ids)
}

/// Audit every fragment and index segment reference in `manifest`.
///
/// This is the opt-in full diagnostic pass. It is deliberately NOT wired into
/// the commit path or table open, which validate only what they changed.
pub fn validate_clustering(manifest: &Manifest, indices: &[IndexMetadata]) -> Result<()> {
    let ids = definition_ids(manifest)?;
    for fragment in manifest.fragments.iter() {
        if let Some(id) = fragment.clustering_id
            && !ids.contains(&id)
        {
            return Err(Error::invalid_input(format!(
                "fragment {} references clustering definition {}, which the manifest does not define",
                fragment.id, id
            )));
        }
    }
    for index in indices {
        if let Some(id) = index.clustering_id
            && !ids.contains(&id)
        {
            return Err(Error::invalid_input(format!(
                "index segment {} ({}) references clustering definition {}, which the manifest does not define",
                index.uuid, index.name, id
            )));
        }
    }
    Ok(())
}

/// Apply a writer's COMPLETE desired definitions list to `manifest`.
///
/// `desired` is the exact definition list the next manifest must carry, not an
/// append list: a current definition the writer omits is removed, and the
/// entries it adds continue `max_clustering_id` in allocation order. The
/// watermark never goes backwards, so removing a definition does not release
/// its id, and an id at or below the mark can never be allocated again.
///
/// `read` is the definition set the writer planned against. It is what
/// separates an intended removal from a lost race: omitting a definition the
/// writer never saw would silently drop a concurrent allocation, so that is a
/// conflict the writer must resolve by reloading and rebuilding the list.
/// Callers that have no separate read snapshot pass the landing base, which is
/// correct precisely because nothing can have intervened.
///
/// A definition is removed only while nothing references it. `indices` is the
/// index section the manifest is published with; when the target removes ids,
/// the manifest's fragments and `indices` are each walked once, and a fragment
/// or index segment still labelled with a removed id refuses the whole update.
/// A target that removes nothing walks neither. The manifest is untouched on
/// every error.
///
/// `commit_version` only labels the conflict error.
pub fn apply_complete_clustering_definitions(
    manifest: &mut Manifest,
    desired: &[ClusteringDefinition],
    read: &[ClusteringDefinition],
    indices: &[IndexMetadata],
    commit_version: u64,
) -> Result<()> {
    let stale = |message: String| Error::commit_conflict_source(commit_version, message.into());
    let mut seen = HashSet::with_capacity(desired.len());
    if let Some(duplicate) = desired.iter().find(|d| !seen.insert(d.id)) {
        return Err(Error::invalid_input(format!(
            "desired clustering definitions list names id {} more than once",
            duplicate.id
        )));
    }
    let by_id: HashMap<u32, &ClusteringDefinition> = desired.iter().map(|d| (d.id, d)).collect();
    let read_ids: HashSet<u32> = read.iter().map(|d| d.id).collect();
    for current in &manifest.clustering {
        match by_id.get(&current.id).copied() {
            Some(d) if d == current => {}
            Some(_) => {
                return Err(stale(format!(
                    "desired clustering definitions list changes definition {}, which is immutable; \
                     reload the manifest and rebuild the list",
                    current.id
                )));
            }
            // Omitted. Intended removal only if the writer planned against a
            // snapshot that had it; otherwise the list predates this
            // definition and publishing it would drop a concurrent allocation.
            None if read_ids.contains(&current.id) => {}
            None => {
                return Err(stale(format!(
                    "desired clustering definitions list omits definition {}, which the writer \
                     never read; reload the manifest and rebuild the list",
                    current.id
                )));
            }
        }
    }
    let current_ids: HashSet<u32> = manifest.clustering.iter().map(|c| c.id).collect();
    let mut next = manifest.max_clustering_id;
    let mut added = Vec::new();
    for definition in desired {
        if current_ids.contains(&definition.id) {
            continue;
        }
        definition.validate()?;
        let Some(expected) = next.checked_add(1) else {
            return Err(Error::invalid_input(
                "clustering definition ids are exhausted",
            ));
        };
        if definition.id != expected {
            return Err(stale(format!(
                "new clustering definition {} does not continue max_clustering_id {} (expected {}); \
                 reload the manifest and rebuild the list",
                definition.id, manifest.max_clustering_id, expected
            )));
        }
        if let Some(missing) = definition
            .columns
            .iter()
            .find(|column| manifest.schema.field_by_id(**column).is_none())
        {
            return Err(Error::invalid_input(format!(
                "clustering definition {} is keyed on field id {missing}, which is not in the schema",
                definition.id
            )));
        }
        next = expected;
        added.push(definition.clone());
    }
    // Retained entries keep their order; additions follow in allocation order.
    // A removal-only list adds nothing, so this cannot early-return on
    // `added.is_empty()`.
    let retained: Vec<ClusteringDefinition> = manifest
        .clustering
        .iter()
        .filter(|current| by_id.contains_key(&current.id))
        .cloned()
        .collect();
    let removed_ids: HashSet<u32> = manifest
        .clustering
        .iter()
        .filter(|current| !by_id.contains_key(&current.id))
        .map(|current| current.id)
        .collect();
    if removed_ids.is_empty() && added.is_empty() {
        #[cfg(test)]
        LAST_REMOVAL_SCAN_VISITS.with(|cell| cell.set(0));
        return Ok(());
    }
    refuse_referenced_removals(manifest, indices, &removed_ids)?;
    manifest.clustering = retained;
    manifest.clustering.extend(added);
    manifest.max_clustering_id = next;
    Ok(())
}

/// Refuse to allocate a definition whose key fields are not the fields the
/// writer planned against.
///
/// A definition keys on field ids. The writer resolved its key columns in the
/// snapshot it read; a rebased attempt lands on a later schema, and on a table
/// with reusable field ids an id may by then name another column (a dropped
/// key replaced by a new column, or an overwrite that numbered a new schema
/// from zero). Allocating would freeze that binding, so each key field of a
/// new definition must be the same field in both schemas: same name, same
/// parent, same type. A field the landing schema lacks is left to the
/// allocation's own schema check. Only definitions new to `manifest` are
/// checked, so a retained list costs nothing here.
pub fn refuse_key_fields_that_changed_identity(
    read_schema: &Schema,
    landing: &Manifest,
    desired: &[ClusteringDefinition],
    commit_version: u64,
) -> Result<()> {
    let current_ids: HashSet<u32> = landing.clustering.iter().map(|c| c.id).collect();
    for definition in desired.iter().filter(|d| !current_ids.contains(&d.id)) {
        for column in &definition.columns {
            let (Some(planned), Some(landed)) = (
                read_schema.field_by_id(*column),
                landing.schema.field_by_id(*column),
            ) else {
                continue;
            };
            if planned.name != landed.name
                || planned.parent_id != landed.parent_id
                || planned.data_type() != landed.data_type()
            {
                return Err(Error::commit_conflict_source(
                    commit_version,
                    format!(
                        "clustering definition {} is keyed on field id {column}, which named {:?} \
                         when the writer read the table and names {:?} now; reload the manifest \
                         and rebuild the list",
                        definition.id, planned.name, landed.name
                    )
                    .into(),
                ));
            }
        }
    }
    Ok(())
}

/// Refuse to remove a definition that a fragment or index segment in the
/// snapshot still references. Each collection is walked once, and only when
/// `removed_ids` is non-empty: a target that removes nothing costs no
/// per-object work.
fn refuse_referenced_removals(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    removed_ids: &HashSet<u32>,
) -> Result<()> {
    #[cfg(test)]
    LAST_REMOVAL_SCAN_VISITS.with(|cell| {
        cell.set(if removed_ids.is_empty() {
            0
        } else {
            manifest.fragments.len() + indices.len()
        })
    });
    if removed_ids.is_empty() {
        return Ok(());
    }
    if let Some(fragment) = manifest.fragments.iter().find(|fragment| {
        fragment
            .clustering_id
            .is_some_and(|id| removed_ids.contains(&id))
    }) {
        return Err(Error::invalid_input(format!(
            "cannot remove clustering definition {}: fragment {} still references it",
            fragment.clustering_id.unwrap_or_default(),
            fragment.id
        )));
    }
    if let Some(segment) = indices.iter().find(|segment| {
        segment
            .clustering_id
            .is_some_and(|id| removed_ids.contains(&id))
    }) {
        return Err(Error::invalid_input(format!(
            "cannot remove clustering definition {}: index segment {} ({}) still references it",
            segment.clustering_id.unwrap_or_default(),
            segment.uuid,
            segment.name
        )));
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Objects the last apply walked for the removal check: fragments plus
    /// index segments when the target removed ids, zero otherwise.
    static LAST_REMOVAL_SCAN_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Objects the last [`apply_complete_clustering_definitions`] on this thread
/// walked for its removal check.
#[cfg(test)]
pub fn last_removal_scan_visits() -> usize {
    LAST_REMOVAL_SCAN_VISITS.with(|cell| cell.get())
}

/// Forget the last removal-check count on this thread.
#[cfg(test)]
pub fn reset_removal_scan_visits() {
    LAST_REMOVAL_SCAN_VISITS.with(|cell| cell.set(0));
}

/// Add every definition of `other` that `target` lacks, keeping ids sorted.
///
/// A definition is immutable, so the same id must carry the same content on
/// both sides; anything else is an error rather than an overwrite.
pub fn merge_clustering_definitions(
    target: &mut Vec<ClusteringDefinition>,
    other: &[ClusteringDefinition],
) -> Result<()> {
    let mut positions: HashMap<u32, usize> = target
        .iter()
        .enumerate()
        .map(|(index, definition)| (definition.id, index))
        .collect();
    for definition in other {
        match positions.get(&definition.id) {
            Some(&index) if target[index] == *definition => {}
            Some(&index) => {
                return Err(Error::invalid_input(format!(
                    "clustering definition {} differs between manifests: {:?} versus {:?}",
                    definition.id, target[index], definition
                )));
            }
            None => {
                positions.insert(definition.id, target.len());
                target.push(definition.clone());
            }
        }
    }
    target.sort_by_key(|definition| definition.id);
    Ok(())
}

impl From<&ClusteringDefinition> for pb::Clustering {
    fn from(definition: &ClusteringDefinition) -> Self {
        Self {
            id: definition.id,
            columns: definition.columns.clone(),
            provider: definition.provider.clone(),
            version: definition.version,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn valid() -> pb::Clustering {
        pb::Clustering {
            id: 3,
            columns: vec![4, 1],
            provider: "org.example.cluster".into(),
            version: 2,
        }
    }

    #[test]
    fn round_trips_a_valid_definition() {
        let definition = ClusteringDefinition::try_from(valid()).unwrap();
        assert_eq!(definition.id, 3);
        assert_eq!(definition.columns, vec![4, 1]);
        assert_eq!(pb::Clustering::from(&definition), valid());
    }

    #[rstest]
    #[case::zero_id(pb::Clustering { id: 0, ..valid() }, "id 0")]
    #[case::no_columns(pb::Clustering { columns: vec![], ..valid() }, "no columns")]
    #[case::duplicate_column(pb::Clustering { columns: vec![1, 2, 1], ..valid() }, "repeats field id 1")]
    #[case::empty_provider(pb::Clustering { provider: String::new(), ..valid() }, "empty provider")]
    #[case::zero_version(pb::Clustering { version: 0, ..valid() }, "version 0")]
    fn rejects_an_invalid_definition(#[case] proto: pb::Clustering, #[case] message: &str) {
        let error = ClusteringDefinition::try_from(proto).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }

    fn manifest_with(definitions: Vec<ClusteringDefinition>, max_id: u32) -> Manifest {
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]);
        let mut manifest = Manifest::new(
            lance_core::datatypes::Schema::try_from(&schema).unwrap(),
            std::sync::Arc::new(vec![crate::format::Fragment::new(0)]),
            crate::format::DataStorageFormat::new(lance_file::version::ConcreteFileVersion::V2_0),
            std::collections::HashMap::new(),
        );
        manifest.clustering = definitions;
        manifest.max_clustering_id = max_id;
        manifest
    }

    /// The accumulator must witness every clustering-specific object visit,
    /// including its own collection: a counter placed after the inherited
    /// check could report one visit while having walked both snapshots.
    /// Holding the operation fixed, growing the untouched object set must not
    /// change the count at all.
    #[test]
    fn clustering_visits_do_not_grow_with_the_untouched_object_set() {
        let mut counts = Vec::new();
        for untouched in [1usize, 1_000] {
            let mut references = ChangedReferences::default();
            // Objects this commit did not touch are never handed to the
            // accumulator, which is the whole point: the builder only reports
            // what it introduced.
            let _ = untouched;
            let mut introduced = crate::format::Fragment::new(7);
            introduced.clustering_id = Some(1);
            references.fragment(&introduced);
            counts.push(references.visits());
        }
        assert_eq!(
            counts[0], counts[1],
            "clustering work must depend on what the commit changed, not on table size"
        );
        assert_eq!(counts[0], 1);
    }

    /// An introduced label that names no definition is refused.
    #[test]
    fn an_introduced_reference_to_an_undefined_id_is_refused() {
        let manifest = manifest_with(vec![definition(1)], 1);
        let mut dangling = crate::format::Fragment::new(7);
        dangling.clustering_id = Some(2);
        let mut references = ChangedReferences::default();
        references.fragment(&dangling);
        let error = validate_clustering_of(&manifest, &references).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("which the manifest does not define"),
            "{error}"
        );
    }

    /// An introduced index segment label is checked the same way.
    #[test]
    fn an_introduced_index_segment_reference_to_an_undefined_id_is_refused() {
        let manifest = manifest_with(vec![definition(1)], 1);
        let mut segment = crate::transaction::test_support::sample_index_metadata("idx");
        segment.clustering_id = Some(9);
        let mut references = ChangedReferences::default();
        references.index_segment(&segment);
        let error = validate_clustering_of(&manifest, &references).unwrap_err();
        assert!(error.to_string().contains("index segment"), "{error}");
    }

    /// A label that resolves is accepted, and the definition set is still
    /// checked in full even when nothing was introduced.
    #[test]
    fn an_empty_change_set_still_validates_the_definition_set() {
        let manifest = manifest_with(vec![definition(1)], 1);
        validate_clustering_of(&manifest, &ChangedReferences::default()).unwrap();

        let above_mark = manifest_with(vec![definition(5)], 1);
        let error = validate_clustering_of(&above_mark, &ChangedReferences::default()).unwrap_err();
        assert!(
            error.to_string().contains("above max_clustering_id"),
            "{error}"
        );
    }

    /// The desired list is the target, not an append list: an omitted
    /// definition the writer did read is removed, and the watermark stays put
    /// so the id is never handed out again.
    #[test]
    fn a_removal_only_target_deletes_and_keeps_the_watermark() {
        let read = vec![definition(1), definition(2), definition(3)];
        let mut manifest = manifest_with(read.clone(), 3);
        apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(2), definition(3)],
            &read,
            &[],
            4,
        )
        .unwrap();
        assert_eq!(ids(&manifest), vec![2, 3], "definition 1 is removed");
        assert_eq!(
            manifest.max_clustering_id, 3,
            "a removal never releases the id"
        );
    }

    /// Removal and allocation in one list, which must be one atomic result.
    #[test]
    fn a_target_can_remove_and_add_at_once() {
        let read = vec![definition(1), definition(2), definition(3)];
        let mut manifest = manifest_with(read.clone(), 3);
        apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(2), definition(3), definition(4)],
            &read,
            &[],
            4,
        )
        .unwrap();
        assert_eq!(ids(&manifest), vec![2, 3, 4]);
        assert_eq!(manifest.max_clustering_id, 4);
    }

    /// An empty target clears the set and still preserves the mark.
    #[test]
    fn an_empty_target_clears_the_set_and_preserves_the_mark() {
        let read = vec![definition(1), definition(2), definition(3)];
        let mut manifest = manifest_with(read.clone(), 3);
        apply_complete_clustering_definitions(&mut manifest, &[], &read, &[], 4).unwrap();
        assert!(manifest.clustering.is_empty());
        assert_eq!(manifest.max_clustering_id, 3);
    }

    /// A removed id is spent: it cannot come back as a fresh allocation, since
    /// new ids continue the mark rather than filling gaps.
    #[test]
    fn a_removed_id_cannot_be_reallocated() {
        let read = vec![definition(1), definition(2)];
        let mut manifest = manifest_with(read.clone(), 2);
        apply_complete_clustering_definitions(&mut manifest, &[definition(2)], &read, &[], 3)
            .unwrap();
        assert_eq!(manifest.max_clustering_id, 2);

        // Resubmitting id 1 as new is below the mark, so it is refused.
        let read_after = vec![definition(2)];
        let error = apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(1), definition(2)],
            &read_after,
            &[],
            4,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("does not continue"),
            "a spent id is not a fresh allocation: {error}"
        );
    }

    /// The distinction that makes removal safe: omitting a definition the
    /// writer never read is a lost concurrent allocation, not an intent to
    /// delete, and publishing it would silently drop that definition.
    #[test]
    fn omitting_a_definition_the_writer_never_read_is_a_conflict() {
        // The writer planned against [1] and wants to keep just [1]; by the
        // time it lands, someone else allocated 2.
        let read = vec![definition(1)];
        let mut manifest = manifest_with(vec![definition(1), definition(2)], 2);
        let error =
            apply_complete_clustering_definitions(&mut manifest, &[definition(1)], &read, &[], 3)
                .unwrap_err();
        assert!(
            error.to_string().contains("never read"),
            "a concurrent allocation must not be dropped: {error}"
        );
        assert_eq!(
            ids(&manifest),
            vec![1, 2],
            "the manifest is left untouched by a refused apply"
        );
    }

    /// Retained definitions stay immutable even while others are removed.
    #[test]
    fn a_removal_cannot_smuggle_a_change_to_a_retained_definition() {
        let read = vec![definition(1), definition(2)];
        let mut manifest = manifest_with(read.clone(), 2);
        let mut altered = definition(2);
        altered.columns = vec![0, 0];
        let error = apply_complete_clustering_definitions(&mut manifest, &[altered], &read, &[], 3)
            .unwrap_err();
        assert!(error.to_string().contains("immutable"), "{error}");
    }

    /// A fragment still labelled with a definition pins it: the target that
    /// omits it is refused and the manifest is left exactly as it was.
    #[test]
    fn a_fragment_reference_blocks_the_removal() {
        let read = vec![definition(1), definition(2), definition(3)];
        let mut manifest = manifest_with(read.clone(), 3);
        std::sync::Arc::make_mut(&mut manifest.fragments)[0].clustering_id = Some(1);
        let error = apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(2), definition(3)],
            &read,
            &[],
            4,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot remove clustering definition 1: fragment 0"),
            "{error}"
        );
        assert_eq!(ids(&manifest), vec![1, 2, 3]);
        assert_eq!(manifest.max_clustering_id, 3);
    }

    /// An index segment alone pins a definition just the same.
    #[test]
    fn an_index_segment_reference_blocks_the_removal() {
        let read = vec![definition(1), definition(2), definition(3)];
        let mut manifest = manifest_with(read.clone(), 3);
        let mut segment = crate::transaction::test_support::sample_index_metadata("idx");
        segment.clustering_id = Some(2);
        let error = apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(1), definition(3)],
            &read,
            std::slice::from_ref(&segment),
            4,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(&format!(
                "cannot remove clustering definition 2: index segment {} (idx)",
                segment.uuid
            )),
            "{error}"
        );
        assert_eq!(ids(&manifest), vec![1, 2, 3]);

        // The unreferenced one can still go.
        apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(1), definition(2)],
            &read,
            &[segment],
            4,
        )
        .unwrap();
        assert_eq!(ids(&manifest), vec![1, 2]);
        assert_eq!(manifest.max_clustering_id, 3);
    }

    /// One referenced id refuses the whole mixed update: no removal, no
    /// allocation and no watermark movement get through.
    #[test]
    fn one_referenced_id_refuses_the_whole_mixed_update() {
        let read = vec![definition(1), definition(2), definition(3)];
        let mut manifest = manifest_with(read.clone(), 3);
        std::sync::Arc::make_mut(&mut manifest.fragments)[0].clustering_id = Some(2);
        let error = apply_complete_clustering_definitions(
            &mut manifest,
            &[definition(3), definition(4)],
            &read,
            &[],
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("definition 2"), "{error}");
        assert_eq!(ids(&manifest), vec![1, 2, 3], "nothing removed");
        assert_eq!(manifest.max_clustering_id, 3, "nothing allocated");
    }

    /// The removal check walks each collection once per apply, however many
    /// ids the target removes, and not at all when it removes none.
    #[test]
    fn the_removal_check_walks_each_collection_once() {
        let read = vec![definition(1), definition(2), definition(3)];
        let fragments: Vec<Fragment> = (0..500).map(Fragment::new).collect();
        let segments: Vec<IndexMetadata> = (0..7)
            .map(|_| crate::transaction::test_support::sample_index_metadata("idx"))
            .collect();
        let fresh = || {
            let mut manifest = manifest_with(read.clone(), 3);
            manifest.fragments = std::sync::Arc::new(fragments.clone());
            manifest
        };

        apply_complete_clustering_definitions(
            &mut fresh(),
            &[definition(1), definition(2), definition(3), definition(4)],
            &read,
            &segments,
            4,
        )
        .unwrap();
        assert_eq!(last_removal_scan_visits(), 0, "an addition walks nothing");

        apply_complete_clustering_definitions(
            &mut fresh(),
            &[definition(2), definition(3)],
            &read,
            &segments,
            4,
        )
        .unwrap();
        assert_eq!(
            last_removal_scan_visits(),
            507,
            "one removed id: one pass each"
        );

        apply_complete_clustering_definitions(&mut fresh(), &[], &read, &segments, 4).unwrap();
        assert_eq!(
            last_removal_scan_visits(),
            507,
            "three removed ids: still one pass each"
        );
    }

    /// A rebased allocation is refused when its key field id names a
    /// different column than the writer read; a same-identity field, a field
    /// the landing schema lacks, and a retained definition are all left alone.
    #[test]
    fn a_key_field_that_changed_identity_refuses_the_allocation() {
        let schema_of = |name: &str, data_type: arrow_schema::DataType| {
            lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![
                arrow_schema::Field::new(name, data_type, false),
            ]))
            .unwrap()
        };
        let read = schema_of("region", arrow_schema::DataType::Int32);
        let mut landing = manifest_with(vec![], 0);
        landing.schema = schema_of("user_id", arrow_schema::DataType::Int32);
        let error = refuse_key_fields_that_changed_identity(&read, &landing, &[definition(1)], 3)
            .unwrap_err();
        assert!(matches!(error, Error::CommitConflict { .. }), "{error:?}");
        assert!(
            error.to_string().contains("named \"region\"")
                && error.to_string().contains("names \"user_id\""),
            "{error}"
        );

        landing.schema = schema_of("region", arrow_schema::DataType::Int64);
        let error = refuse_key_fields_that_changed_identity(&read, &landing, &[definition(1)], 3)
            .unwrap_err();
        assert!(error.to_string().contains("field id 0"), "{error}");

        landing.schema = schema_of("region", arrow_schema::DataType::Int32);
        refuse_key_fields_that_changed_identity(&read, &landing, &[definition(1)], 3).unwrap();

        // A retained definition is not re-checked, whatever the field became.
        landing.schema = schema_of("user_id", arrow_schema::DataType::Int32);
        landing.clustering = vec![definition(1)];
        landing.max_clustering_id = 1;
        refuse_key_fields_that_changed_identity(&read, &landing, &[definition(1)], 3).unwrap();

        // A key the landing schema lacks is the allocation check's business.
        landing.clustering = vec![];
        landing.max_clustering_id = 0;
        landing.schema = lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("x", arrow_schema::DataType::Int32, false),
        ]))
        .unwrap();
        for field in landing.schema.fields.iter_mut() {
            field.id = 9;
        }
        refuse_key_fields_that_changed_identity(&read, &landing, &[definition(1)], 3).unwrap();
    }

    fn ids(manifest: &Manifest) -> Vec<u32> {
        manifest.clustering.iter().map(|d| d.id).collect()
    }

    fn definition(id: u32) -> ClusteringDefinition {
        ClusteringDefinition {
            id,
            columns: vec![0],
            provider: "org.example.cluster".into(),
            version: 1,
        }
    }

    #[test]
    fn validates_definitions_and_references() {
        let manifest = manifest_with(vec![definition(1), definition(2)], 2);
        validate_clustering(&manifest, &[]).unwrap();

        let above_mark = manifest_with(vec![definition(3)], 2);
        let error = validate_clustering(&above_mark, &[]).unwrap_err();
        assert!(
            error.to_string().contains("above max_clustering_id"),
            "{error}"
        );

        let duplicate = manifest_with(vec![definition(1), definition(1)], 1);
        let error = validate_clustering(&duplicate, &[]).unwrap_err();
        assert!(error.to_string().contains("more than once"), "{error}");

        let mut dangling = manifest_with(vec![definition(1)], 1);
        std::sync::Arc::make_mut(&mut dangling.fragments)[0].clustering_id = Some(2);
        let error = validate_clustering(&dangling, &[]).unwrap_err();
        assert!(
            error.to_string().contains("fragment 0 references"),
            "{error}"
        );

        let mut zero = manifest_with(vec![definition(1)], 1);
        std::sync::Arc::make_mut(&mut zero.fragments)[0].clustering_id = Some(0);
        assert!(validate_clustering(&zero, &[]).is_err());

        // A key column dropped after allocation does not invalidate the definition.
        let dropped_column = manifest_with(
            vec![ClusteringDefinition {
                columns: vec![42],
                ..definition(1)
            }],
            1,
        );
        validate_clustering(&dropped_column, &[]).unwrap();
    }

    #[test]
    fn validates_index_segment_references() {
        let manifest = manifest_with(vec![definition(1)], 1);
        let mut segment = crate::transaction::test_support::sample_index_metadata("idx");
        segment.clustering_id = Some(1);
        validate_clustering(&manifest, std::slice::from_ref(&segment)).unwrap();
        segment.clustering_id = Some(7);
        let error = validate_clustering(&manifest, &[segment]).unwrap_err();
        assert!(error.to_string().contains("index segment"), "{error}");
    }

    #[test]
    fn merges_definitions_by_id_and_rejects_conflicts() {
        let mut target = vec![definition(2)];
        merge_clustering_definitions(&mut target, &[definition(1), definition(2)]).unwrap();
        assert_eq!(target, vec![definition(1), definition(2)]);

        let conflicting = ClusteringDefinition {
            provider: "org.example.other".into(),
            ..definition(2)
        };
        let error = merge_clustering_definitions(&mut target, &[conflicting]).unwrap_err();
        assert!(
            error.to_string().contains("differs between manifests"),
            "{error}"
        );
    }
}
