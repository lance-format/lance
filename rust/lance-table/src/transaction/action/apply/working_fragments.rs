// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Copy-on-write fragment working set and lazy dropped-field masking for
//! [`WorkingManifest`](super::WorkingManifest).

use super::{MAX_FRAGMENT_ID, TOMBSTONED_FIELD};
use crate::format::{Fragment, Manifest, RowIdMeta};
use crate::rowids::read_row_ids;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

/// Copy-on-write fragment set over the version applied onto, with lazy masking
/// of fields dropped by [`DropField`](super::super::DropField).
#[derive(Clone)]
pub(super) struct WorkingFragments {
    base: Arc<Vec<Fragment>>,
    patched: BTreeMap<u64, Option<Arc<Fragment>>>,
    dropped_fields: Arc<BTreeSet<i32>>,
}

impl WorkingFragments {
    pub(super) fn from_manifest(manifest: &Manifest) -> Result<Self> {
        Ok(Self {
            base: base_fragments_by_id(manifest)?,
            patched: BTreeMap::new(),
            dropped_fields: Default::default(),
        })
    }

    pub(super) fn max_base_id(&self) -> Option<u64> {
        self.base.last().map(|fragment| fragment.id)
    }

    pub(super) fn base_fragment(&self, fragment_id: u64) -> Option<&Fragment> {
        self.base
            .binary_search_by_key(&fragment_id, |fragment| fragment.id)
            .ok()
            .map(|position| &self.base[position])
    }

    /// The step's own copy of `fragment_id`, if any step touched it. `Some(None)`
    /// means a step removed it.
    pub(super) fn patch(&self, fragment_id: u64) -> Option<Option<&Arc<Fragment>>> {
        self.patched.get(&fragment_id).map(Option::as_ref)
    }

    #[cfg(test)]
    pub(super) fn patched_ids(&self) -> Vec<u64> {
        self.patched.keys().copied().collect()
    }

    pub(super) fn contains(&self, fragment_id: u64) -> bool {
        match self.patched.get(&fragment_id) {
            Some(patch) => patch.is_some(),
            None => self.base_fragment(fragment_id).is_some(),
        }
    }

    /// Skips lazy [`DropField`](super::super::DropField) masking on unpatched
    /// base fragments. Use this only where field lists do not matter, or for
    /// fragments known to live in `patched` that already carry earlier drop
    /// masks. Cloning and reinserting an unpatched base fragment would lose
    /// its lazy drop mask.
    pub(super) fn get_unmasked(&self, fragment_id: u64) -> Option<&Fragment> {
        match self.patched.get(&fragment_id) {
            Some(patch) => patch.as_deref(),
            None => self.base_fragment(fragment_id),
        }
    }

    pub(super) fn fragment(
        &self,
        fragment_id: u64,
        schema: &Schema,
    ) -> Result<Option<Cow<'_, Fragment>>> {
        match self.patched.get(&fragment_id) {
            Some(patch) => Ok(patch.as_deref().map(Cow::Borrowed)),
            None => {
                let Some(base) = self.base_fragment(fragment_id) else {
                    return Ok(None);
                };
                if self.dropped_fields.is_empty() || !self.fragment_carries_dropped_field(base) {
                    Ok(Some(Cow::Borrowed(base)))
                } else {
                    let live = schema_field_ids(schema);
                    Ok(Some(Cow::Owned(
                        self.with_dropped_fields_tombstoned(base, &live)?,
                    )))
                }
            }
        }
    }

    pub(super) fn get_mut(
        &mut self,
        fragment_id: u64,
        schema: &Schema,
        action: &str,
    ) -> Result<&mut Fragment> {
        let missing = || {
            Error::invalid_input(format!(
                "{action} targets fragment {fragment_id}, which does not exist"
            ))
        };
        if !self.patched.contains_key(&fragment_id) {
            let base = self.base_fragment(fragment_id).ok_or_else(missing)?;
            let masked = if self.dropped_fields.is_empty() {
                base.clone()
            } else {
                let live = schema_field_ids(schema);
                self.with_dropped_fields_tombstoned(base, &live)?
            };
            self.patched.insert(fragment_id, Some(Arc::new(masked)));
        }
        self.patched
            .get_mut(&fragment_id)
            .and_then(Option::as_mut)
            .map(Arc::make_mut)
            .ok_or_else(missing)
    }

    pub(super) fn insert(&mut self, fragment: Fragment) {
        self.patched.insert(fragment.id, Some(Arc::new(fragment)));
    }

    pub(super) fn remove(&mut self, fragment_id: u64) {
        self.patched.insert(fragment_id, None);
    }

    pub(super) fn live_ids(&self) -> impl Iterator<Item = u64> + '_ {
        let unpatched_base = self
            .base
            .iter()
            .map(|fragment| fragment.id)
            .filter(|id| !self.patched.contains_key(id));
        let patched = self
            .patched
            .iter()
            .filter_map(|(id, patch)| patch.as_ref().map(|_| *id));
        unpatched_base.chain(patched)
    }

    pub(super) fn tombstone_dropped_field(&mut self, field_id: i32, schema: &Schema) -> Result<()> {
        Arc::make_mut(&mut self.dropped_fields).insert(field_id);
        if self.patched.is_empty() {
            return Ok(());
        }
        let single = BTreeSet::from([field_id]);
        let live = schema_field_ids(schema);
        for patched in self.patched.values_mut().flatten() {
            tombstone_fields_in_fragment(Arc::make_mut(patched), &single, &live)?;
        }
        Ok(())
    }

    pub(super) fn clear(&mut self) {
        self.base = Arc::default();
        self.patched.clear();
        self.dropped_fields = Arc::default();
    }

    pub(super) fn build_fragments(&self, schema: &Schema) -> Result<Vec<Fragment>> {
        let live = if self.dropped_fields.is_empty() {
            HashSet::new()
        } else {
            schema_field_ids(schema)
        };
        let mut fragments = Vec::with_capacity(self.base.len() + self.patched.len());
        let mut patches = self.patched.iter().peekable();
        for base in self.base.iter() {
            while let Some((_, patch)) = patches.next_if(|(id, _)| **id < base.id) {
                fragments.extend(patch.as_deref().cloned());
            }
            match patches.next_if(|(id, _)| **id == base.id) {
                Some((_, patch)) => fragments.extend(patch.as_deref().cloned()),
                None => fragments.push(self.with_dropped_fields_tombstoned(base, &live)?),
            }
        }
        fragments.extend(patches.filter_map(|(_, patch)| patch.as_deref().cloned()));
        Ok(fragments)
    }

    fn fragment_carries_dropped_field(&self, fragment: &Fragment) -> bool {
        fragment.files.iter().any(|file| {
            file.fields
                .iter()
                .any(|field_id| self.dropped_fields.contains(field_id))
        }) || fragment.overlays.iter().any(|overlay| {
            overlay
                .data_file
                .fields
                .iter()
                .any(|field_id| self.dropped_fields.contains(field_id))
        })
    }

    fn with_dropped_fields_tombstoned(
        &self,
        base: &Fragment,
        live_schema_fields: &HashSet<i32>,
    ) -> Result<Fragment> {
        let mut fragment = base.clone();
        if !self.dropped_fields.is_empty() {
            tombstone_fields_in_fragment(&mut fragment, &self.dropped_fields, live_schema_fields)?;
        }
        Ok(fragment)
    }
}

/// The manifest's fragments in strictly increasing id order, which is what
/// lookups by id binary search.
///
/// Manifests written by Lance 0.10 and earlier are not in id order, so those
/// get a sorted copy. Manifests from 0.16 and earlier can repeat an id, and an
/// id above `u32::MAX` cannot be recorded in the manifest high-water mark.
fn base_fragments_by_id(manifest: &Manifest) -> Result<Arc<Vec<Fragment>>> {
    let base = if manifest
        .fragments
        .windows(2)
        .all(|pair| pair[0].id < pair[1].id)
    {
        Arc::clone(&manifest.fragments)
    } else {
        let mut sorted = manifest.fragments.as_ref().clone();
        sorted.sort_by_key(|fragment| fragment.id);
        if let Some(repeated) = sorted.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(Error::invalid_input(format!(
                "version {} lists fragment id {} more than once, so its fragments cannot be \
                 looked up by id to apply actions",
                manifest.version, repeated[0].id
            )));
        }
        Arc::new(sorted)
    };
    if let Some(last) = base.last()
        && last.id > MAX_FRAGMENT_ID
    {
        return Err(Error::invalid_input(format!(
            "version {} contains fragment id {}, above the maximum fragment id {MAX_FRAGMENT_ID}",
            manifest.version, last.id
        )));
    }
    Ok(base)
}

fn schema_field_ids(schema: &Schema) -> HashSet<i32> {
    schema.fields_pre_order().map(|field| field.id).collect()
}

fn tombstone_fields_in_fragment(
    fragment: &mut Fragment,
    dropped_fields: &BTreeSet<i32>,
    live_schema_fields: &HashSet<i32>,
) -> Result<()> {
    for file in fragment.files.iter_mut() {
        if !file
            .fields
            .iter()
            .any(|field_id| dropped_fields.contains(field_id))
        {
            continue;
        }
        // A V1 reader locates pages by position in the file's field list, so
        // the list stays whole until no field in it remains in the schema.
        if file.file_version()? == ConcreteFileVersion::V1 {
            if !file
                .fields
                .iter()
                .any(|field_id| live_schema_fields.contains(field_id))
            {
                file.fields = vec![TOMBSTONED_FIELD; file.fields.len()].into();
            }
            continue;
        }
        file.fields = file
            .fields
            .iter()
            .map(|id| {
                if dropped_fields.contains(id) {
                    TOMBSTONED_FIELD
                } else {
                    *id
                }
            })
            .collect::<Vec<_>>()
            .into();
    }
    let overlaid: Vec<u32> = dropped_fields
        .iter()
        .filter_map(|id| u32::try_from(*id).ok())
        .collect();
    if !fragment.overlays.is_empty() && !overlaid.is_empty() {
        crate::format::overlay::tombstone_overlay_fields(&mut fragment.overlays, &overlaid);
    }
    Ok(())
}

pub(super) fn fresh_row_id_count(fragment: &Fragment) -> Result<u64> {
    let physical_rows = fragment.physical_rows.ok_or_else(|| {
        Error::internal(format!(
            "fragment {} does not record its physical rows, so its row ids cannot be counted",
            fragment.id
        ))
    })? as u64;
    match &fragment.row_id_meta {
        None => Ok(physical_rows),
        Some(RowIdMeta::Column) => Ok(0),
        Some(RowIdMeta::Inline(row_ids)) => {
            let supplied = read_row_ids(row_ids)?.len() as u64;
            physical_rows.checked_sub(supplied).ok_or_else(|| {
                Error::internal(format!(
                    "fragment {} carries {supplied} row ids for {physical_rows} physical rows",
                    fragment.id
                ))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::WorkingManifest;
    use super::super::tests::{apply_steps, config_update, remove_fragment_zero, whole_fragment};
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::{
        added_field, any_deletion_file, backed_manifest, frag_reuse,
    };
    use crate::transaction::action::{
        Action, DropField, FieldRef, Ref, RemoveFragment, SetDeletionFile, UserStep,
    };
    use crate::transaction::test_support::{default_build_config, overlay_with_field};

    #[test]
    fn test_a_step_patches_only_the_fragments_it_touches_and_shares_the_rest() {
        let base = Arc::new(frag_reuse::table(4, false));
        let config = default_build_config();
        let working = WorkingManifest::at_read_version(base.clone(), Vec::new(), &config).unwrap();
        assert!(Arc::ptr_eq(&working.fragments.base, &base.fragments));

        let mut stepped = working.clone();
        let mut actions = whole_fragment(0);
        actions.push(remove_fragment_zero());
        stepped.apply_step(&UserStep::new("step", actions)).unwrap();

        assert!(working.fragments.patched_ids().is_empty());
        assert!(Arc::ptr_eq(&stepped.fragments.base, &base.fragments));
        assert_eq!(stepped.fragments.patched_ids(), vec![0, 4]);
        assert_eq!(stepped.fragments.patch(0), Some(None));

        let mut mutated_clone = stepped.clone();
        assert!(Arc::ptr_eq(
            stepped.fragments.patch(4).flatten().unwrap(),
            mutated_clone.fragments.patch(4).flatten().unwrap()
        ));
        assert!(Arc::ptr_eq(
            &stepped.fragment_tokens,
            &mutated_clone.fragment_tokens
        ));
        assert!(Arc::ptr_eq(
            &stepped.new_fragments,
            &mutated_clone.new_fragments
        ));
        assert!(Arc::ptr_eq(&stepped.indices, &mutated_clone.indices));
        assert!(Arc::ptr_eq(&stepped.config, &mutated_clone.config));
        assert!(Arc::ptr_eq(&working.config, &stepped.config));

        mutated_clone
            .apply_step(&UserStep::new(
                "mutate_clone",
                vec![
                    config_update(),
                    Action::SetDeletionFile(
                        SetDeletionFile::try_new_unchecked(1, any_deletion_file(), None, true)
                            .unwrap(),
                    ),
                ],
            ))
            .unwrap();
        assert!(stepped.fragments.patch(1).is_none());
        assert!(stepped.config.is_empty());

        let (manifest, _) = stepped.build_manifest().unwrap();
        let fragment_ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(fragment_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_lazy_drop_field_masks_v2_and_preserves_v1_files_without_corrupting_base() {
        let mut manifest = backed_manifest();
        let mut keep_field = added_field("keep");
        keep_field.id = 1;
        manifest.schema.fields.push(keep_field);

        let mut v2_frag = manifest.fragments[0].clone();
        v2_frag.id = 0;
        v2_frag.files = vec![DataFile::new(
            "data/0.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_0,
            None,
            None,
        )];
        v2_frag.overlays = vec![
            overlay_with_field(0, manifest.version),
            overlay_with_field(1, manifest.version),
        ];

        let mut v1_frag = manifest.fragments[0].clone();
        v1_frag.id = 1;
        v1_frag.files = vec![DataFile::new_legacy_from_fields(
            "data/1.lance",
            vec![0, 1],
            None,
        )];
        manifest.fragments = Arc::new(vec![v2_frag, v1_frag]);
        manifest.max_fragment_id = Some(1);

        let base = Arc::new(manifest);
        let config = default_build_config();
        let mut state =
            WorkingManifest::at_read_version(base.clone(), Vec::new(), &config).unwrap();

        state
            .apply_step(&UserStep::new(
                "drop_0",
                vec![Action::DropField(DropField::new(FieldRef::Committed(0)))],
            ))
            .unwrap();

        assert!(state.fragments.patched_ids().is_empty());
        assert_eq!(base.fragments[0].files[0].fields.as_ref(), &[0, 1]);
        assert_eq!(base.fragments[1].files[0].fields.as_ref(), &[0, 1]);

        assert_eq!(
            state.live_fragment(0).unwrap().unwrap().files[0]
                .fields
                .as_ref(),
            &[TOMBSTONED_FIELD, 1]
        );
        assert_eq!(
            state.live_fragment(1).unwrap().unwrap().files[0]
                .fields
                .as_ref(),
            &[0, 1]
        );

        let (after_one_drop, _) = state.build_manifest().unwrap();
        assert_eq!(
            after_one_drop.fragments[0].files[0].fields.as_ref(),
            &[TOMBSTONED_FIELD, 1]
        );
        assert_eq!(after_one_drop.fragments[0].overlays.len(), 1);
        assert_eq!(
            after_one_drop.fragments[0].overlays[0]
                .data_file
                .fields
                .as_ref(),
            &[1]
        );
        assert_eq!(base.fragments[0].overlays.len(), 2);
        assert_eq!(
            after_one_drop.fragments[1].files[0].fields.as_ref(),
            &[0, 1]
        );

        state
            .apply_step(&UserStep::new(
                "drop_1",
                vec![Action::DropField(DropField::new(FieldRef::Committed(1)))],
            ))
            .unwrap();
        let (after_both_drops, _) = state.build_manifest().unwrap();
        assert!(after_both_drops.fragments[0].files.is_empty());
        assert!(after_both_drops.fragments[1].files.is_empty());
        assert_eq!(base.fragments[0].files[0].fields.as_ref(), &[0, 1]);
        assert_eq!(base.fragments[1].files[0].fields.as_ref(), &[0, 1]);
    }

    #[test]
    fn test_a_base_out_of_id_order_is_looked_up_by_id_and_built_sorted() {
        let mut base = frag_reuse::table(2, false);
        let mut fragments = base.fragments.as_ref().clone();
        fragments.reverse();
        base.fragments = Arc::new(fragments);

        let next = apply_steps(
            &base,
            vec![
                vec![Action::RemoveFragment(RemoveFragment::new(Ref::Committed(
                    1,
                )))],
                whole_fragment(0),
            ],
        )
        .unwrap();
        let fragment_ids: Vec<u64> = next.fragments.iter().map(|f| f.id).collect();
        assert_eq!(fragment_ids, vec![0, 2]);
        assert_eq!(next.fragments[0], base.fragments[1]);

        WorkingManifest::at_read_version(Arc::new(base), Vec::new(), &default_build_config())
            .unwrap();
    }

    #[test]
    fn test_base_manifests_with_duplicate_or_out_of_range_fragment_ids_are_rejected() {
        let mut duplicate = frag_reuse::table(1, false);
        let repeated = duplicate.fragments[0].clone();
        duplicate.fragments = Arc::new(vec![repeated.clone(), repeated]);
        for err in [
            apply_steps(&duplicate, vec![whole_fragment(0)]).unwrap_err(),
            WorkingManifest::at_read_version(
                Arc::new(duplicate),
                Vec::new(),
                &default_build_config(),
            )
            .err()
            .unwrap(),
        ] {
            assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
            assert!(
                err.to_string()
                    .contains("lists fragment id 0 more than once"),
                "{err}"
            );
        }

        let mut out_of_range = frag_reuse::table(1, false);
        let mut bad_frag = out_of_range.fragments[0].clone();
        bad_frag.id = MAX_FRAGMENT_ID + 1;
        out_of_range.fragments = Arc::new(vec![bad_frag]);
        for err in [
            apply_steps(&out_of_range, vec![whole_fragment(0)]).unwrap_err(),
            WorkingManifest::at_read_version(
                Arc::new(out_of_range),
                Vec::new(),
                &default_build_config(),
            )
            .err()
            .unwrap(),
        ] {
            assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
            assert!(
                err.to_string().contains("above the maximum fragment id"),
                "{err}"
            );
        }
    }
}
