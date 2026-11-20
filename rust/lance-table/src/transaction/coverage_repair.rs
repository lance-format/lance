// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index coverage a pending transaction adds, repaired against what committed
//! after it read.
//!
//! A segment describes the values it was built from. When a concurrent commit
//! rewrites those values, the segment can still land: it gives up its coverage
//! of the affected fragments and a reader scans those rows instead. Coverage
//! loss is always legal, and an index build is expensive to redo, so this is a
//! repair after the conflict check rather than a conflict.
//!
//! The repair is read off the manifests, not off the transactions that
//! produced them: a fragment's data for a field changed when the data files
//! backing that field differ between the version the segment was built from
//! and the latest. Of the versions between, it keeps only a [`FragmentTrace`]
//! of the fragments they created and removed, never a manifest per version.

use std::collections::HashMap;

use lance_core::datatypes::Schema;

use crate::format::overlay::staleness::{field_affects_index, overlay_affects_index};
use crate::format::{Fragment, Manifest};
use uuid::Uuid;

/// What committed after a pending transaction read, as the repair sees it.
pub struct CommittedSince<'a> {
    /// The manifest the transaction read.
    pub read: &'a Manifest,
    /// The latest manifest.
    pub latest: &'a Manifest,
    /// The fragments created and removed between them.
    pub trace: &'a FragmentTrace,
    /// The fragments a fragment reuse history records `fragment`'s rows as
    /// moved into, following one compaction after another: empty when it
    /// records no move of `fragment`, `None` when the history cannot be walked.
    pub moved_to: &'a dyn Fn(u64) -> Option<Vec<u64>>,
}

/// The fragments created and removed across a run of manifests: each created
/// fragment as it first appeared, and each removed one as it last was.
///
/// That is all the repair reads of the versions between the read and the
/// latest: a covered fragment still live is compared from where it first
/// appears to the latest, and one removed is judged by what it carried when
/// it left. So it holds what the commits touched, never a manifest per
/// version, and a commit carries it from one attempt to the next.
#[derive(Debug, Default, Clone)]
pub struct FragmentTrace {
    created: HashMap<u64, Fragment>,
    removed: HashMap<u64, Fragment>,
}

impl FragmentTrace {
    /// Record what changed from `before` to `after`, the next manifest
    /// observed. Both list their fragments sorted by id, which
    /// [`Manifest`] maintains.
    pub fn observe(&mut self, before: &Manifest, after: &Manifest) {
        let mut before = before.fragments.iter().peekable();
        let mut after = after.fragments.iter().peekable();
        loop {
            match (before.peek(), after.peek()) {
                (None, None) => break,
                (Some(old), Some(new)) if old.id == new.id => {
                    before.next();
                    after.next();
                }
                (None, Some(new)) => {
                    self.created.entry(new.id).or_insert_with(|| (*new).clone());
                    after.next();
                }
                (Some(old), Some(new)) if old.id > new.id => {
                    self.created.entry(new.id).or_insert_with(|| (*new).clone());
                    after.next();
                }
                (Some(old), _) => {
                    self.removed.insert(old.id, (*old).clone());
                    before.next();
                }
            }
        }
    }

    /// How many fragments are held.
    pub fn len(&self) -> usize {
        self.created.len() + self.removed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The coverage a pending transaction gives one index segment.
pub struct NewCoverage {
    /// The segment.
    pub segment: Uuid,
    /// Keyed and carried fields alike, by committed id: a carried column
    /// rewritten while the keyed one is untouched leaves the segment answering
    /// from an obsolete value.
    pub fields: Vec<i32>,
    /// The committed fragments covered.
    pub fragments: Vec<u64>,
    /// The version the segment's values were read at. An overlay committed
    /// after it is masked out of the segment by the read path, until a
    /// compaction folds it into base data.
    pub built_at: u64,
    /// Whether the segment cannot reach rows through the fragment reuse
    /// history, as an NGram segment cannot.
    pub cannot_follow_moves: bool,
}

/// The fragments `coverage` must give up to land on the latest version, in
/// the order it lists them.
///
/// - A covered fragment still live loses coverage when the data backing one of
///   the segment's fields was rewritten, in it or in a fragment the history
///   moved its rows into. A field backs the segment when it is, contains, or
///   is nested in one of its fields, in the read or the latest schema.
/// - A covered fragment removed since keeps its coverage: a plain delete
///   leaves no rows to describe, and a recorded move is translated by the
///   reader. It loses it when its data was rewritten before it was removed,
///   when the rows moved somewhere rewritten since, when a compaction folded
///   in an overlay the segment did not see, or when the segment cannot follow
///   moves at all.
/// - Where the history cannot be walked, coverage goes.
pub fn stale_coverage(since: &CommittedSince<'_>, coverage: &NewCoverage) -> Vec<u64> {
    if since.latest.version == since.read.version {
        return Vec::new();
    }
    let history = History::new(since);
    coverage
        .fragments
        .iter()
        .copied()
        .filter(|fragment| history.withdraws(*fragment, coverage))
        .collect()
}

/// The read manifest, the latest, and the trace between them, as the repair
/// reads them.
struct History<'a> {
    read: &'a Manifest,
    latest: &'a Manifest,
    trace: &'a FragmentTrace,
    schemas: [&'a Schema; 2],
    moved_to: &'a dyn Fn(u64) -> Option<Vec<u64>>,
}

impl<'a> History<'a> {
    fn new(since: &CommittedSince<'a>) -> Self {
        Self {
            read: since.read,
            latest: since.latest,
            trace: since.trace,
            schemas: [&since.read.schema, &since.latest.schema],
            moved_to: since.moved_to,
        }
    }

    fn withdraws(&self, fragment: u64, coverage: &NewCoverage) -> bool {
        let Some(moved_to) = (self.moved_to)(fragment) else {
            return true;
        };
        let live = self.latest.fragment_by_id(fragment).is_some();
        if !live && !moved_to.is_empty() && coverage.cannot_follow_moves {
            return true;
        }
        std::iter::once(fragment)
            .chain(moved_to.iter().copied())
            .any(|reached| {
                self.rewritten(reached, &coverage.fields) || self.folded_overlay(reached, coverage)
            })
    }

    /// Whether the data backing `fields` in `fragment` differs between when it
    /// was first and last seen in the window. A fragment removed since counts
    /// too: a compaction output carries the rewritten values from the start,
    /// so the rewrite shows only on the fragment it moved rows out of.
    fn rewritten(&self, fragment: u64, fields: &[i32]) -> bool {
        let Some(first) = self
            .read
            .fragment_by_id(fragment)
            .or_else(|| self.trace.created.get(&fragment))
        else {
            return false;
        };
        let Some(last) = self
            .latest
            .fragment_by_id(fragment)
            .or_else(|| self.trace.removed.get(&fragment))
        else {
            return false;
        };
        self.backing(first, fields) != self.backing(last, fields)
    }

    /// Whether `fragment` left the dataset within the window carrying an
    /// overlay the segment never saw, whose values a compaction folding it
    /// into base data now hides from the read path's masking.
    fn folded_overlay(&self, fragment: u64, coverage: &NewCoverage) -> bool {
        if self.latest.fragment_by_id(fragment).is_some() {
            return false;
        }
        let Some(last) = self
            .trace
            .removed
            .get(&fragment)
            .or_else(|| self.read.fragment_by_id(fragment))
        else {
            return false;
        };
        last.overlays.iter().any(|overlay| {
            self.schemas.iter().any(|schema| {
                overlay_affects_index(overlay, &coverage.fields, coverage.built_at, schema)
            })
        })
    }

    /// Which data file entry backs each field in `fragment` that is, contains,
    /// or is nested in one of `fields`: its position in the fragment's file
    /// list, path, and base. Per field rather than per file, so a sibling
    /// tombstoned out of a shared file and rewritten elsewhere leaves the
    /// others' entries as they were. The position catches a field rebound to
    /// a new entry over the same path, which is a rewrite as far as the
    /// manifest can tell.
    fn backing(
        &self,
        fragment: &Fragment,
        fields: &[i32],
    ) -> Vec<(i32, usize, String, Option<u32>)> {
        let mut backing: Vec<(i32, usize, String, Option<u32>)> = fragment
            .files
            .iter()
            .enumerate()
            .flat_map(|(position, file)| {
                file.fields
                    .iter()
                    .filter(|field| {
                        **field >= 0
                            && self
                                .schemas
                                .iter()
                                .any(|schema| field_affects_index(**field, fields, schema))
                    })
                    .map(move |field| (*field, position, file.path.clone(), file.base_id))
            })
            .collect();
        backing.sort();
        backing
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

    use super::*;
    use crate::format::{DataFile, DataStorageFormat};
    use crate::transaction::test_support::overlay_with_field;

    /// `a` and `c` top level, then a struct `s` with children `x` and `y`.
    fn schema() -> Schema {
        let arrow = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, true),
            ArrowField::new(
                "s",
                DataType::Struct(Fields::from(vec![
                    ArrowField::new("x", DataType::Int32, true),
                    ArrowField::new("y", DataType::Int32, true),
                ])),
                true,
            ),
        ]);
        Schema::try_from(&arrow).unwrap()
    }

    fn id(name: &str) -> i32 {
        schema().field(name).unwrap().id
    }

    fn file(path: &str, fields: &[i32]) -> DataFile {
        DataFile::new(
            format!("data/{path}.lance"),
            fields.to_vec(),
            (0..fields.len() as i32).collect::<Vec<_>>(),
            ConcreteFileVersion::V2_0,
            None,
            None,
        )
    }

    /// A fragment with one file for `a`, one for `c` and one packing `s`.
    fn fragment(id_: u64) -> Fragment {
        let mut fragment = Fragment::new(id_);
        fragment.physical_rows = Some(3);
        fragment.files = vec![
            file(&format!("a-{id_}"), &[id("a")]),
            file(&format!("c-{id_}"), &[id("c")]),
            file(&format!("s-{id_}"), &[id("s"), id("s.x"), id("s.y")]),
        ];
        fragment
    }

    fn manifest(version: u64, fragments: Vec<Fragment>) -> Manifest {
        let mut manifest = Manifest::new(
            schema(),
            Arc::new(fragments),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.version = version;
        manifest
    }

    /// `fragments` with `field`, and every field nested in it, rewritten in
    /// fragment `target` the way a replacement does it: tombstoned out of the
    /// file that held it and written to a new one.
    fn rewrite(fragments: &[Fragment], target: u64, field: &str) -> Vec<Fragment> {
        let schema = schema();
        let mut rewritten = vec![id(field)];
        rewritten.extend(
            schema
                .field(field)
                .unwrap()
                .children
                .iter()
                .map(|child| child.id),
        );
        fragments
            .iter()
            .cloned()
            .map(|mut fragment| {
                if fragment.id == target {
                    for file in &mut fragment.files {
                        file.fields = file
                            .fields
                            .iter()
                            .map(|id| if rewritten.contains(id) { -2 } else { *id })
                            .collect();
                    }
                    fragment
                        .files
                        .push(file(&format!("{field}-{target}-rewritten"), &rewritten));
                }
                fragment
            })
            .collect()
    }

    fn without(fragments: &[Fragment], removed: u64) -> Vec<Fragment> {
        fragments
            .iter()
            .filter(|fragment| fragment.id != removed)
            .cloned()
            .collect()
    }

    fn with(fragments: &[Fragment], added: Fragment) -> Vec<Fragment> {
        let mut fragments = fragments.to_vec();
        fragments.push(added);
        fragments.sort_by_key(|fragment| fragment.id);
        fragments
    }

    /// The fragments read at version 1: 0, 1 and 2.
    fn read() -> Manifest {
        manifest(1, (0..3).map(fragment).collect())
    }

    struct Case {
        committed: Vec<Manifest>,
        moved: HashMap<u64, Option<Vec<u64>>>,
        fields: Vec<i32>,
        built_at: u64,
        cannot_follow_moves: bool,
    }

    impl Case {
        fn new(committed: Vec<Manifest>, fields: &[&str]) -> Self {
            Self {
                committed,
                moved: HashMap::new(),
                fields: fields.iter().map(|name| id(name)).collect(),
                built_at: 1,
                cannot_follow_moves: false,
            }
        }

        fn moved(mut self, from: u64, to: Option<Vec<u64>>) -> Self {
            self.moved.insert(from, to);
            self
        }

        fn stale(&self) -> Vec<u64> {
            let read = read();
            let moved_to = |fragment: u64| match self.moved.get(&fragment) {
                Some(moved) => moved.clone(),
                None => Some(Vec::new()),
            };
            let mut trace = FragmentTrace::default();
            let mut previous = &read;
            for manifest in &self.committed {
                trace.observe(previous, manifest);
                previous = manifest;
            }
            stale_coverage(
                &CommittedSince {
                    read: &read,
                    latest: previous,
                    trace: &trace,
                    moved_to: &moved_to,
                },
                &NewCoverage {
                    segment: Uuid::nil(),
                    fields: self.fields.clone(),
                    fragments: vec![0, 1, 2],
                    built_at: self.built_at,
                    cannot_follow_moves: self.cannot_follow_moves,
                },
            )
        }
    }

    #[test]
    fn test_nothing_committed_since_keeps_everything() {
        assert!(Case::new(vec![read()], &["a"]).stale().is_empty());
        assert!(Case::new(vec![], &["a"]).stale().is_empty());
    }

    /// Keyed and carried fields alike, and any field the indexed one is nested
    /// in or contains: a packed struct is rewritten under the parent's id.
    #[rstest]
    #[case::keyed_field_rewritten(&["a"], "a", vec![1])]
    #[case::carried_field_rewritten(&["a", "c"], "c", vec![1])]
    #[case::unrelated_field_rewritten(&["a"], "c", vec![])]
    #[case::child_under_a_rewritten_parent(&["s.x"], "s", vec![1])]
    #[case::parent_over_a_rewritten_child(&["s"], "s.y", vec![1])]
    #[case::carried_child_under_a_rewritten_parent(&["a", "s.x"], "s", vec![1])]
    #[case::child_beside_a_rewritten_sibling(&["s.x"], "s.y", vec![])]
    fn test_a_column_rewritten_in_place(
        #[case] indexed: &[&str],
        #[case] rewritten: &str,
        #[case] expected: Vec<u64>,
    ) {
        let read = read();
        let committed = manifest(2, rewrite(&read.fragments, 1, rewritten));
        assert_eq!(Case::new(vec![committed], indexed).stale(), expected);
    }

    /// The read path masks an overlay newer than the segment out of it, so an
    /// overlay that lands on a covered fragment costs no coverage.
    #[test]
    fn test_an_overlay_on_a_live_fragment_keeps_coverage() {
        let read = read();
        let mut fragments = read.fragments.as_ref().clone();
        fragments[1].overlays.push(overlay_with_field(id("a"), 2));
        let committed = manifest(2, fragments);
        assert!(Case::new(vec![committed], &["a"]).stale().is_empty());
    }

    /// A deleted fragment has no rows left to describe, and a recorded move is
    /// translated by the reader, so neither costs coverage on its own.
    #[rstest]
    #[case::deleted(None)]
    #[case::moved_and_recorded(Some(vec![7]))]
    fn test_a_removed_fragment_keeps_coverage(#[case] moved_to: Option<Vec<u64>>) {
        let read = read();
        let mut fragments = without(&read.fragments, 1);
        if let Some(destinations) = &moved_to {
            fragments = with(&fragments, fragment(destinations[0]));
        }
        let case = Case::new(vec![manifest(2, fragments)], &["a"])
            .moved(1, Some(moved_to.unwrap_or_default()));
        assert!(case.stale().is_empty());
    }

    #[test]
    fn test_rows_moved_somewhere_rewritten_since_lose_coverage() {
        let read = read();
        let compacted = with(&without(&read.fragments, 1), fragment(7));
        let rewritten = rewrite(&compacted, 7, "a");
        let case = Case::new(vec![manifest(2, compacted), manifest(3, rewritten)], &["a"])
            .moved(1, Some(vec![7]));
        assert_eq!(case.stale(), vec![1]);
    }

    /// The compaction output carries the rewritten values from the version it
    /// first appears in, so the rewrite shows only on the covered fragment.
    #[rstest]
    #[case::indexed_field(&["a"], vec![1])]
    #[case::unrelated_field(&["c"], vec![])]
    fn test_a_fragment_rewritten_then_compacted(
        #[case] indexed: &[&str],
        #[case] expected: Vec<u64>,
    ) {
        let read = read();
        let rewritten = rewrite(&read.fragments, 1, "a");
        let compacted = with(&without(&rewritten, 1), fragment(7));
        let case = Case::new(
            vec![manifest(2, rewritten), manifest(3, compacted)],
            indexed,
        )
        .moved(1, Some(vec![7]));
        assert_eq!(case.stale(), expected);
    }

    /// A compaction output rewritten in the same window as the compaction is
    /// compared from the version it first appears in.
    #[test]
    fn test_a_destination_untouched_since_the_move_keeps_coverage() {
        let read = read();
        let compacted = with(&without(&read.fragments, 1), fragment(7));
        let case = Case::new(
            vec![manifest(2, compacted.clone()), manifest(3, compacted)],
            &["a"],
        )
        .moved(1, Some(vec![7]));
        assert!(case.stale().is_empty());
    }

    #[test]
    fn test_a_segment_that_cannot_follow_moves_loses_moved_coverage() {
        let read = read();
        let compacted = with(&without(&read.fragments, 1), fragment(7));
        let mut case = Case::new(vec![manifest(2, compacted)], &["a"]).moved(1, Some(vec![7]));
        case.cannot_follow_moves = true;
        assert_eq!(case.stale(), vec![1]);
    }

    /// Nothing is left to mask the values of an overlay a compaction folded
    /// into base data, whether it sat on a covered fragment or on an earlier
    /// compaction's output.
    #[rstest]
    #[case::newer_than_the_build(2, vec![1])]
    #[case::older_than_the_build(1, vec![])]
    fn test_a_folded_overlay(#[case] overlay_version: u64, #[case] expected: Vec<u64>) {
        let read = read();
        let mut overlaid = read.fragments.as_ref().clone();
        overlaid[1]
            .overlays
            .push(overlay_with_field(id("a"), overlay_version));
        let compacted = with(&without(&overlaid, 1), fragment(7));
        let case = Case::new(vec![manifest(2, overlaid), manifest(3, compacted)], &["a"])
            .moved(1, Some(vec![7]));
        assert_eq!(case.stale(), expected);
    }

    #[test]
    fn test_an_overlay_folded_by_a_later_compaction_of_a_compaction_output() {
        let read = read();
        let first = with(&without(&read.fragments, 1), fragment(7));
        let mut overlaid = first.clone();
        let output = overlaid.iter_mut().find(|f| f.id == 7).unwrap();
        output.overlays.push(overlay_with_field(id("a"), 3));
        let second = with(&without(&overlaid, 7), fragment(8));
        let case = Case::new(
            vec![
                manifest(2, first),
                manifest(3, overlaid),
                manifest(4, second),
            ],
            &["a"],
        )
        .moved(1, Some(vec![7, 8]))
        .moved(7, Some(vec![8]));
        assert_eq!(case.stale(), vec![1]);
    }

    #[test]
    fn test_a_history_that_cannot_be_walked_loses_coverage() {
        let read = read();
        let committed = manifest(2, read.fragments.as_ref().clone());
        let case = Case::new(vec![committed], &["a"]).moved(2, None);
        assert_eq!(case.stale(), vec![2]);
    }

    /// A field rebound to a new file entry over the same path is rewritten as
    /// far as the manifest can tell.
    #[test]
    fn test_a_field_rebound_to_a_new_entry_over_the_same_path_loses_coverage() {
        let read = read();
        let mut fragments = read.fragments.as_ref().clone();
        let path = fragments[1].files[0].path.clone();
        fragments[1].files[0].fields = vec![-2].into();
        let mut rebound = file("unused", &[id("a")]);
        rebound.path = path;
        fragments[1].files.push(rebound);
        let committed = manifest(2, fragments);
        assert_eq!(Case::new(vec![committed], &["a"]).stale(), vec![1]);
    }

    /// A re-encode swaps the files without changing the values. Nothing in
    /// the manifest tells it apart from a rewrite, so it costs the coverage.
    #[test]
    fn test_a_reencode_loses_coverage() {
        let read = read();
        let committed = manifest(2, rewrite(&read.fragments, 0, "a"));
        assert_eq!(Case::new(vec![committed], &["a"]).stale(), vec![0]);
    }
}
