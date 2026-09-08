// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Which rows an overlay makes stale with respect to an index.
//!
//! An overlay supplies replacement values for some `(row, field)` cells without
//! rewriting the base data. An index built before that overlay was committed still
//! reflects the old values, so those rows must be excluded from the index's results
//! and re-evaluated against current values on the flat path.
//!
//! Deciding which rows are affected needs only fragment and index metadata — the
//! overlay coverage bitmaps, the overlay `committed_version`, and the indexed field
//! ids — so it lives here rather than in the read path that consumes it.

use std::collections::HashMap;

use lance_core::Result;
use lance_core::datatypes::Schema;
use roaring::RoaringBitmap;

use crate::format::overlay::DataOverlayFile;
use crate::format::{Fragment, IndexMetadata};

/// The physical offsets within a fragment whose value for an indexed field may be
/// stale relative to an index built at `index_version`, and so must be excluded
/// from that index's results and re-evaluated against current values on the flat
/// path.
///
/// The set is the union, over every overlay whose `committed_version` is newer
/// than `index_version`, of that overlay's coverage **restricted to the indexed
/// fields**. The restriction makes exclusion field-aware: an overlay that touches
/// only non-indexed fields contributes nothing. An overlay whose
/// `committed_version <= index_version` is already incorporated by the index and
/// is ignored.
pub fn overlay_exclusion_offsets(
    overlays: &[DataOverlayFile],
    indexed_field_ids: &[i32],
    index_version: u64,
    schema: &Schema,
) -> Result<RoaringBitmap> {
    let mut excluded = RoaringBitmap::new();
    for overlay in overlays {
        if overlay.committed_version <= index_version {
            continue;
        }
        for (field_pos, field_id) in overlay.data_file.fields.iter().enumerate() {
            let overlay_ancestry = schema.field_ancestry_by_id(*field_id);
            let affects_index = indexed_field_ids.iter().any(|indexed_field_id| {
                indexed_field_id == field_id
                    || overlay_ancestry.as_ref().is_some_and(|ancestry| {
                        ancestry
                            .iter()
                            .any(|ancestor| ancestor.id == *indexed_field_id)
                    })
                    || schema
                        .field_ancestry_by_id(*indexed_field_id)
                        .is_some_and(|ancestry| {
                            ancestry.iter().any(|ancestor| ancestor.id == *field_id)
                        })
            });
            if affects_index {
                excluded |= &*overlay.coverage_for_field(field_pos)?;
            }
        }
    }
    Ok(excluded)
}

// Stale row offsets contributed by one fragment's overlays for a given index version.
// Applies a cheap version gate first: if every overlay predates the segment it is already
// incorporated by the index, so there is nothing stale and the field/bitmap work is skipped.
fn stale_offsets_for_fragment(
    fragment: &Fragment,
    fields: &[i32],
    index_version: u64,
    schema: &Schema,
) -> Result<RoaringBitmap> {
    if fragment
        .overlays
        .iter()
        .all(|o| o.committed_version <= index_version)
    {
        return Ok(RoaringBitmap::new());
    }
    overlay_exclusion_offsets(&fragment.overlays, fields, index_version, schema)
}

// A missing `fragment_bitmap` means the index predates fragment-bitmap tracking; treat it as
// covering every fragment (matching `lance::index::prefilter::DatasetPreFilter::new`) so
// overlay-stale rows can't slip through unmasked. Only skip fragments explicitly absent from a
// present bitmap.
fn covers_fragment(coverage: Option<&RoaringBitmap>, frag_id: u32) -> bool {
    coverage.is_none_or(|c| c.contains(frag_id))
}

/// Index by fragment id the fragments that carry at least one overlay. Overlays are rare, so
/// this is empty on the common path, letting callers skip index loading entirely; when non-empty
/// it bounds the stale-collection loops to `O(overlaid fragments)`.
pub fn overlaid_fragments(fragments: &[Fragment]) -> HashMap<u32, &Fragment> {
    fragments
        .iter()
        .filter(|f| !f.overlays.is_empty())
        .map(|f| (f.id as u32, f))
        .collect()
}

/// Insert into `stale` the ids of fragments covered by `segment` whose index entries may be
/// stale because an overlay committed after the segment was built touches a field the segment
/// indexes. Field-aware and version-gated via [`overlay_exclusion_offsets`].
///
/// `overlaid_frags` holds only the fragments that actually carry overlays (rare), so the loop is
/// `O(overlaid_frags)` rather than `O(fragments the segment covers)`.
pub fn collect_overlay_stale_frags(
    segment: &IndexMetadata,
    overlaid_frags: &HashMap<u32, &Fragment>,
    stale: &mut RoaringBitmap,
    schema: &Schema,
) -> Result<()> {
    let coverage = segment.fragment_bitmap.as_ref();
    for (&frag_id, fragment) in overlaid_frags {
        if stale.contains(frag_id) || !covers_fragment(coverage, frag_id) {
            continue;
        }
        if !stale_offsets_for_fragment(fragment, &segment.fields, segment.dataset_version, schema)?
            .is_empty()
        {
            stale.insert(frag_id);
        }
    }
    Ok(())
}

/// Like [`collect_overlay_stale_frags`] but with row-level granularity: instead of marking the
/// whole fragment stale, it computes exactly which row offsets within each covered fragment are
/// stale and accumulates them into `stale` (fragment_id → stale row offsets).
///
/// Used by the scalar and vector paths to block only the affected rows from index results and
/// re-evaluate only those rows on the flat path, keeping overhead proportional to the number of
/// overlaid rows rather than the whole fragment size.
pub fn collect_overlay_stale_rows_for_segment(
    segment: &IndexMetadata,
    overlaid_frags: &HashMap<u32, &Fragment>,
    stale: &mut HashMap<u32, RoaringBitmap>,
    schema: &Schema,
) -> Result<()> {
    let coverage = segment.fragment_bitmap.as_ref();
    for (&frag_id, fragment) in overlaid_frags {
        if !covers_fragment(coverage, frag_id) {
            continue;
        }
        let excluded =
            stale_offsets_for_fragment(fragment, &segment.fields, segment.dataset_version, schema)?;
        if !excluded.is_empty() {
            *stale.entry(frag_id).or_default() |= &excluded;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::overlay::OverlayCoverage;

    fn bitmap(offsets: impl IntoIterator<Item = u32>) -> RoaringBitmap {
        RoaringBitmap::from_iter(offsets)
    }

    fn flat_test_schema() -> Schema {
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

        let mut schema = Schema::try_from(&ArrowSchema::new(
            (0..5)
                .map(|id| ArrowField::new(format!("field_{id}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ))
        .unwrap();
        schema.set_field_id(None);
        schema
    }

    /// `outer: struct<middle: struct<a: i32, b: i32>>`, for the ancestry checks.
    fn nested_struct_schema() -> Schema {
        use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};

        let mid = Fields::from(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]);
        let outer_fields =
            Fields::from(vec![ArrowField::new("middle", DataType::Struct(mid), true)]);
        let mut schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "outer",
            DataType::Struct(outer_fields),
            true,
        )]))
        .unwrap();
        schema.set_field_id(None);
        schema
    }

    /// A dense overlay covering `offsets` for `field_ids`, committed at `version`.
    fn dense_overlay(
        field_ids: Vec<i32>,
        offsets: impl IntoIterator<Item = u32>,
        version: u64,
    ) -> DataOverlayFile {
        DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", field_ids, None),
            coverage: OverlayCoverage::dense(bitmap(offsets)),
            committed_version: version,
        }
    }

    #[test]
    fn test_exclusion_offsets_version_gate() {
        let schema = flat_test_schema();
        // index built at version 5; only overlays committed > 5 are excluded.
        let overlays = vec![
            dense_overlay(vec![3], [0, 1], 4),
            dense_overlay(vec![3], [2, 7], 6),
        ];
        let excluded = overlay_exclusion_offsets(&overlays, &[3], 5, &schema).unwrap();
        assert_eq!(excluded, bitmap([2, 7]));
        // An overlay exactly at the index version is already incorporated.
        let overlays = vec![dense_overlay(vec![3], [9], 5)];
        assert!(
            overlay_exclusion_offsets(&overlays, &[3], 5, &schema)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_exclusion_offsets_is_field_aware() {
        let schema = flat_test_schema();
        // An overlay touching only an unrelated field excludes nothing.
        let overlays = vec![dense_overlay(vec![2], [0, 1, 2], 9)];
        assert!(
            overlay_exclusion_offsets(&overlays, &[3], 1, &schema)
                .unwrap()
                .is_empty()
        );
        // The union spans only the indexed fields the overlay actually carries.
        let overlays = vec![dense_overlay(vec![2, 3], [4], 9)];
        assert_eq!(
            overlay_exclusion_offsets(&overlays, &[3], 1, &schema).unwrap(),
            bitmap([4])
        );
    }

    #[test]
    fn test_exclusion_offsets_matches_nested_fields() {
        let schema = nested_struct_schema();
        let outer = &schema.fields[0];
        let middle = &outer.children[0];
        let a = &middle.children[0];
        let b = &middle.children[1];

        let overlays = vec![dense_overlay(vec![a.id], [1], 9)];
        assert_eq!(
            overlay_exclusion_offsets(&overlays, &[outer.id], 1, &schema).unwrap(),
            bitmap([1])
        );
        assert!(
            overlay_exclusion_offsets(&overlays, &[b.id], 1, &schema)
                .unwrap()
                .is_empty()
        );

        let overlays = vec![dense_overlay(vec![middle.id], [2], 9)];
        assert_eq!(
            overlay_exclusion_offsets(&overlays, &[a.id], 1, &schema).unwrap(),
            bitmap([2])
        );
    }

    #[test]
    fn test_exclusion_offsets_sparse_per_field() {
        let schema = flat_test_schema();
        // Sparse overlay: field 2 covers {2,3}, field 4 covers {1}.
        let overlay = DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![2, 4], None),
            coverage: OverlayCoverage::sparse(vec![bitmap([2, 3]), bitmap([1])]),
            committed_version: 9,
        };
        let overlays = vec![overlay];
        // Only the bitmap for the indexed field (4) contributes.
        assert_eq!(
            overlay_exclusion_offsets(&overlays, &[4], 1, &schema).unwrap(),
            bitmap([1])
        );
        assert_eq!(
            overlay_exclusion_offsets(&overlays, &[2], 1, &schema).unwrap(),
            bitmap([2, 3])
        );
    }

    #[test]
    fn test_exclusion_offsets_unions_multiple_overlays() {
        let schema = flat_test_schema();
        let overlays = vec![
            dense_overlay(vec![3], [1], 6),
            dense_overlay(vec![3], [4, 5], 7),
        ];
        assert_eq!(
            overlay_exclusion_offsets(&overlays, &[3], 1, &schema).unwrap(),
            bitmap([1, 4, 5])
        );
    }

    /// An index segment covering `fields`, built at `dataset_version`, with the given
    /// fragment coverage (`None` = legacy index predating fragment-bitmap tracking).
    fn segment(
        fields: Vec<i32>,
        dataset_version: u64,
        fragment_bitmap: Option<RoaringBitmap>,
    ) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: "idx".into(),
            fields,
            covering_fields: vec![],
            dataset_version,
            fragment_bitmap,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    fn fragment_with_overlay(id: u64, overlay: DataOverlayFile) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment.overlays.push(overlay);
        fragment
    }

    #[test]
    fn test_collect_frags_missing_bitmap_covers_all() {
        let schema = flat_test_schema();
        // A segment with no fragment_bitmap (legacy index predating bitmap tracking) must treat
        // every overlaid fragment as covered so stale rows can't leak past the index unmasked.
        let fragment = fragment_with_overlay(3, dense_overlay(vec![3], [1, 2], 9));
        let overlaid: HashMap<u32, &Fragment> = HashMap::from([(3u32, &fragment)]);

        let mut stale = RoaringBitmap::new();
        collect_overlay_stale_frags(&segment(vec![3], 1, None), &overlaid, &mut stale, &schema)
            .unwrap();
        assert_eq!(stale, bitmap([3]), "missing bitmap must cover fragment 3");

        // A present bitmap that excludes fragment 3 leaves it untouched.
        let mut stale = RoaringBitmap::new();
        collect_overlay_stale_frags(
            &segment(vec![3], 1, Some(bitmap([0]))),
            &overlaid,
            &mut stale,
            &schema,
        )
        .unwrap();
        assert!(
            stale.is_empty(),
            "fragment absent from bitmap is not covered"
        );

        // A present bitmap that includes fragment 3 marks it stale.
        let mut stale = RoaringBitmap::new();
        collect_overlay_stale_frags(
            &segment(vec![3], 1, Some(bitmap([3]))),
            &overlaid,
            &mut stale,
            &schema,
        )
        .unwrap();
        assert_eq!(stale, bitmap([3]));
    }

    #[test]
    fn test_collect_rows_missing_bitmap_covers_all() {
        let schema = flat_test_schema();
        // Same covers-all guarantee at row-level granularity.
        let fragment = fragment_with_overlay(3, dense_overlay(vec![3], [1, 2], 9));
        let overlaid: HashMap<u32, &Fragment> = HashMap::from([(3u32, &fragment)]);

        let mut stale = HashMap::new();
        collect_overlay_stale_rows_for_segment(
            &segment(vec![3], 1, None),
            &overlaid,
            &mut stale,
            &schema,
        )
        .unwrap();
        assert_eq!(
            stale.get(&3),
            Some(&bitmap([1, 2])),
            "missing bitmap must cover fragment 3"
        );

        // A present bitmap that excludes fragment 3 yields no stale rows.
        let mut stale = HashMap::new();
        collect_overlay_stale_rows_for_segment(
            &segment(vec![3], 1, Some(bitmap([0]))),
            &overlaid,
            &mut stale,
            &schema,
        )
        .unwrap();
        assert!(
            stale.is_empty(),
            "fragment absent from bitmap contributes no rows"
        );
    }
}
