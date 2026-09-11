// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Feature flags

use crate::format::Manifest;
use lance_core::{Error, Result};

/// Fragments may contain deletion files, which record the tombstones of
/// soft-deleted rows.
pub const FLAG_DELETION_FILES: u64 = 1 << 0;
/// Row ids are stable for both moves and updates. Fragments contain an index
/// mapping row ids to row addresses.
pub const FLAG_STABLE_ROW_IDS: u64 = 1 << 1;
/// Files are written with the new v2 format (this flag is no longer used)
pub const FLAG_USE_V2_FORMAT_DEPRECATED: u64 = 1 << 2;
/// Table config is present
pub const FLAG_TABLE_CONFIG: u64 = 1 << 3;
/// Dataset uses multiple base paths (for shallow clones or multi-base datasets)
pub const FLAG_BASE_PATHS: u64 = 1 << 4;
/// Disable writing transaction file under _transaction/, this flag is set when we only want to write inline transaction in manifest
pub const FLAG_DISABLE_TRANSACTION_FILE: u64 = 1 << 5;
/// Fragments contain data overlay files, which supply new values for a subset of
/// cells without rewriting base data files. A reader that does not understand
/// overlays must refuse the dataset, since ignoring an overlay would silently
/// return stale base values.
///
/// Data overlay files are not yet a released feature: in release builds this flag
/// is treated as unknown (so a release reader/writer refuses an overlay dataset)
/// unless [`ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV`] is set, which lets benchmarks opt in.
/// Debug builds always understand it so tests exercise the path.
pub const FLAG_UNSTABLE_DATA_OVERLAY_FILES: u64 = 1 << 6;
/// Some index declares covering columns: `IndexMetadata.covering_fields` names
/// columns the index carries values for but is not keyed on.
///
/// Covering makes `fields` mean "keyed columns followed by carried columns"
/// rather than "the columns this index is searched on". A reader without this
/// bit still selects a vector index by testing membership of `fields`, so it
/// would answer a query on a merely-carried column with an index keyed on a
/// different column and return wrong neighbours with no error. A writer without
/// it would maintain the index as though every entry of `fields` were keyed.
/// Both must refuse the table.
///
/// This takes the bit reclaimed from the retired MemWAL index-catchup flag
/// (<https://github.com/lance-format/lance/pull/8680>), which is the boundary the
/// current released build treats as unknown -- so that build refuses a covering
/// dataset without needing a change of its own. Builds from the window where the
/// bit was allocated to index catch-up (v11.0.0-beta.4 through beta.17) still
/// count it as supported and will open a covering dataset rather than refuse it;
/// that exposure comes with the reclamation and is inherited by whichever flag
/// takes the bit.
pub const FLAG_COVERED_INDEX_METADATA: u64 = 1 << 7;
/// Reserved for datasets that reference recognized V2 data files with
/// different exact versions.
pub const FLAG_MIXED_DATA_FILE_VERSIONS: u64 = 1 << 8;
/// The first bit that is unknown as a feature flag
pub const FLAG_UNKNOWN: u64 = 1 << 8;
/// Some fragment stores its row id sequence as a hidden column of a data file
/// rather than inline in the manifest (`RowIdMeta::Column`).
///
/// A reader without this bit sees an unset `row_id_sequence` oneof and would
/// take the fragment to have no row ids at all, on a table whose manifest says
/// every fragment has them. Both readers and writers must refuse the table.
///
/// Deliberately sits above [`FLAG_UNKNOWN`]: spilled row ids are not a released
/// feature, so every released build already refuses the dataset without needing
/// a change of its own. This build understands the bit only in debug builds or
/// when [`ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV`] is set, mirroring
/// [`FLAG_UNSTABLE_DATA_OVERLAY_FILES`].
pub const FLAG_UNSTABLE_SPILLED_ROW_IDS: u64 = 1 << 9;

// Supported flags stay below the unknown boundary; the mixed-version bit is
// reserved at the boundary until its storage contract lands.
const _: () = assert!(FLAG_COVERED_INDEX_METADATA < FLAG_UNKNOWN);
// The fence needs a bit the current released build already refuses, which means
// at or above the boundary that build shipped with (bit 7).
const _: () = assert!(FLAG_COVERED_INDEX_METADATA >= 1 << 7);
const _: () = assert!(FLAG_MIXED_DATA_FILE_VERSIONS == FLAG_UNKNOWN);

pub(crate) const STICKY_PAIRED_FLAGS: u64 = FLAG_MIXED_DATA_FILE_VERSIONS;

/// Environment variable that opts a release build into reading and writing data
/// overlay files before the feature is generally released.
pub const ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV: &str = "LANCE_ENABLE_UNSTABLE_DATA_OVERLAY_FILES";

/// Environment variable that opts a release build into reading and writing row
/// id sequences spilled to a hidden data file column before the feature is
/// generally released.
pub const ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV: &str = "LANCE_ENABLE_UNSTABLE_SPILLED_ROW_IDS";

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(
    manifest: &mut Manifest,
    enable_stable_row_id: bool,
    disable_transaction_file: bool,
) -> Result<()> {
    // Carried across the reset: a `Manifest` only points at its index section,
    // so whether any index declares covering columns is not visible here. `build_manifest` decides it from the index list it is
    // committing and sets the bit after calling this; without the carry the
    // second call, from `write_manifest_file`, would clear that decision
    // immediately before the write.
    let covered_index_metadata = (manifest.reader_feature_flags | manifest.writer_feature_flags)
        & FLAG_COVERED_INDEX_METADATA;
    let sticky_paired_flags = validated_sticky_paired_flags(manifest)?;

    // Reset flags
    manifest.reader_feature_flags = 0;
    manifest.writer_feature_flags = 0;

    let has_deletion_files = manifest
        .fragments
        .iter()
        .any(|frag| frag.deletion_file.is_some());
    if has_deletion_files {
        // Both readers and writers need to be able to read deletion files
        manifest.reader_feature_flags |= FLAG_DELETION_FILES;
        manifest.writer_feature_flags |= FLAG_DELETION_FILES;
    }

    // If any fragment has row ids, they must all have row ids.
    let has_row_ids = manifest
        .fragments
        .iter()
        .any(|frag| frag.row_id_meta.is_some());
    if has_row_ids || enable_stable_row_id {
        if !manifest
            .fragments
            .iter()
            .all(|frag| frag.row_id_meta.is_some())
        {
            return Err(Error::invalid_input("All fragments must have row ids"));
        }
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_STABLE_ROW_IDS;
    }

    // Test whether any table metadata has been set
    if !manifest.config.is_empty() {
        manifest.writer_feature_flags |= FLAG_TABLE_CONFIG;
    }

    // Check if this dataset uses multiple base paths (for shallow clones or multi-base datasets)
    if !manifest.base_paths.is_empty() {
        manifest.reader_feature_flags |= FLAG_BASE_PATHS;
        manifest.writer_feature_flags |= FLAG_BASE_PATHS;
    }

    // Overlay files change cell values on read, so a reader that ignores them
    // would return stale base values. Both readers and writers must understand
    // them.
    let has_overlays = manifest
        .fragments
        .iter()
        .any(|frag| !frag.overlays.is_empty());
    if has_overlays {
        manifest.reader_feature_flags |= FLAG_UNSTABLE_DATA_OVERLAY_FILES;
        manifest.writer_feature_flags |= FLAG_UNSTABLE_DATA_OVERLAY_FILES;
    }

    let has_spilled_row_ids = manifest.fragments.iter().any(|frag| {
        frag.row_id_meta
            .as_ref()
            .is_some_and(|meta| meta.column_file().is_some())
    });
    if has_spilled_row_ids {
        manifest.reader_feature_flags |= FLAG_UNSTABLE_SPILLED_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_UNSTABLE_SPILLED_ROW_IDS;
    }

    if disable_transaction_file {
        manifest.writer_feature_flags |= FLAG_DISABLE_TRANSACTION_FILE;
    }

    manifest.reader_feature_flags |= covered_index_metadata;
    manifest.writer_feature_flags |= covered_index_metadata;
    manifest.reader_feature_flags |= sticky_paired_flags;
    manifest.writer_feature_flags |= sticky_paired_flags;

    Ok(())
}

/// Carry sticky paired capabilities from the manifest a new one is derived
/// from.
///
/// [`apply_feature_flags`] carries these bits across its own reset, but it only
/// ever sees one manifest. Constructors preserve these flags, and this helper
/// also validates that the source is not half-set before a derived manifest is
/// committed.
///
/// A half-set state is refused rather than normalized: one bit set means a
/// legacy reader or a legacy writer is still permitted, which is neither mode.
pub fn inherit_sticky_feature_flags(destination: &mut Manifest, source: &Manifest) -> Result<()> {
    let sticky_flags = validated_sticky_paired_flags(source)?;
    destination.reader_feature_flags |= sticky_flags;
    destination.writer_feature_flags |= sticky_flags;
    Ok(())
}

/// Whether this build understands data overlay files: always in debug builds,
/// and in release builds only when [`ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV`] is set.
fn data_overlay_files_enabled() -> bool {
    cfg!(debug_assertions) || std::env::var_os(ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV).is_some()
}

/// Whether this build understands row id sequences spilled to a data file
/// column. Gated the same way as [`data_overlay_files_enabled`].
pub fn spilled_row_ids_enabled() -> bool {
    cfg!(debug_assertions) || std::env::var_os(ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV).is_some()
}

/// Clear `flag` from `flags` when its gating feature is not enabled in this
/// build; leave it set otherwise. One call per unstable flag, so support for
/// several unstable features chains cleanly.
fn mark_supported(flags: &mut u64, flag: u64, feature_enabled: bool) {
    if !feature_enabled {
        *flags &= !flag;
    }
}

/// The feature-flag bits this build understands, given whether overlay support
/// is enabled. Split out from [`supported_flags`] so the policy is testable
/// without toggling the build profile or environment.
fn supported_flags_when(overlay_enabled: bool, spilled_row_ids: bool) -> u64 {
    let mut supported = FLAG_UNKNOWN - 1;
    mark_supported(
        &mut supported,
        FLAG_UNSTABLE_DATA_OVERLAY_FILES,
        overlay_enabled,
    );
    // Added rather than cleared: the bit is above `FLAG_UNKNOWN`, so it is not
    // in the base set to begin with.
    if spilled_row_ids {
        supported |= FLAG_UNSTABLE_SPILLED_ROW_IDS;
    }
    supported
}

fn supported_flags() -> u64 {
    supported_flags_when(data_overlay_files_enabled(), spilled_row_ids_enabled())
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags & !supported_flags() == 0
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags & !supported_flags() == 0
}

/// Refuse reads from manifests whose required reader features this build does
/// not support or whose paired capabilities are inconsistent.
pub fn ensure_can_read_manifest(manifest: &Manifest) -> Result<()> {
    validate_paired_feature_flags(manifest)?;
    if !can_read_dataset(manifest.reader_feature_flags) {
        return Err(Error::not_supported_source(
            format!(
                "This dataset cannot be read by this version of Lance. Please upgrade \
                 Lance to read this dataset. Flags: {}",
                manifest.reader_feature_flags
            )
            .into(),
        ));
    }
    Ok(())
}

/// Refuse writes to manifests whose required writer features this build does
/// not support or whose paired capabilities are inconsistent.
pub fn ensure_can_write_manifest(manifest: &Manifest) -> Result<()> {
    validate_paired_feature_flags(manifest)?;
    if !can_write_dataset(manifest.writer_feature_flags) {
        return Err(Error::not_supported_source(
            format!(
                "This dataset cannot be written by this version of Lance. Please upgrade \
                 Lance to write this dataset. Flags: {}",
                manifest.writer_feature_flags
            )
            .into(),
        ));
    }
    Ok(())
}

pub fn has_deprecated_v2_feature_flag(writer_flags: u64) -> bool {
    writer_flags & FLAG_USE_V2_FORMAT_DEPRECATED != 0
}

/// Refuse a manifest whose paired reader and writer capability bits disagree.
///
/// One word set and the other not is neither mode: it would let a legacy reader
/// or a legacy writer through on a table where the other half is enforcing. The
/// commit path refuses to *produce* this, so seeing it on read means the
/// manifest was written by something that did not.
pub fn validate_paired_feature_flags(manifest: &Manifest) -> Result<()> {
    let reader = manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS != 0;
    let writer = manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS != 0;
    if reader != writer {
        return Err(Error::corrupt_file_named(
            "manifest",
            "Manifest has only one of the mixed data-file-version reader and writer feature bits set, \
             so its semantics are undefined",
        ));
    }
    Ok(())
}

fn validated_sticky_paired_flags(manifest: &Manifest) -> Result<u64> {
    validate_paired_feature_flags(manifest)?;
    Ok(manifest.reader_feature_flags & STICKY_PAIRED_FLAGS)
}

#[cfg(test)]
mod tests {
    /// The covering fence only works if the bit is one the current released
    /// build already rejects. That build's unknown boundary is 128, so the bit
    /// has to be 128 and this build has to have moved its own boundary past it
    /// -- otherwise either that build accepts a covering dataset, or we refuse
    /// our own.
    #[test]
    fn test_covered_index_metadata_fences_older_builds_only() {
        assert_eq!(
            FLAG_COVERED_INDEX_METADATA, 128,
            "the fence must sit on the boundary the released build shipped with"
        );
        assert!(
            can_read_dataset(FLAG_COVERED_INDEX_METADATA),
            "this build implements covering, so it must accept its own datasets"
        );
        assert!(can_write_dataset(FLAG_COVERED_INDEX_METADATA));
        // A build whose boundary is still 128 refuses the bit, which is the fence;
        // the module-level `const _` assertion keeps it at or above that boundary.
    }

    use super::*;
    use crate::format::BasePath;

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES));
        assert!(can_read_dataset(super::FLAG_STABLE_ROW_IDS));
        assert!(can_read_dataset(super::FLAG_USE_V2_FORMAT_DEPRECATED));
        assert!(can_read_dataset(super::FLAG_TABLE_CONFIG));
        assert!(can_read_dataset(super::FLAG_BASE_PATHS));
        assert!(can_read_dataset(super::FLAG_DISABLE_TRANSACTION_FILE));
        // Overlay support is gated on the build profile / env opt-in, so the
        // flag is readable exactly when overlays are enabled (see
        // test_data_overlay_flag_release_gating for the full policy).
        assert_eq!(
            can_read_dataset(super::FLAG_UNSTABLE_DATA_OVERLAY_FILES),
            data_overlay_files_enabled()
        );
        assert!(can_read_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
        ));
        assert!(!can_read_dataset(super::FLAG_UNKNOWN));
    }

    #[test]
    fn test_data_overlay_flag_release_gating() {
        // Release default (overlays disabled): the overlay flag is treated as
        // unknown so the dataset is refused, while other known flags still pass.
        let supported = supported_flags_when(false, false);
        assert_eq!(supported & FLAG_UNSTABLE_DATA_OVERLAY_FILES, 0);
        assert_eq!(FLAG_DELETION_FILES & !supported, 0);
        assert_ne!(FLAG_UNSTABLE_DATA_OVERLAY_FILES & !supported, 0);
        // Enabled (debug or env opt-in): the overlay flag is understood.
        let supported = supported_flags_when(true, false);
        assert_eq!(FLAG_UNSTABLE_DATA_OVERLAY_FILES & !supported, 0);
    }

    #[test]
    fn test_spilled_row_ids_flag_release_gating() {
        // The bit sits above `FLAG_UNKNOWN`, so a build that has not opted in
        // refuses the dataset without any change of its own.
        let supported = supported_flags_when(true, false);
        assert_ne!(FLAG_UNSTABLE_SPILLED_ROW_IDS & !supported, 0);
        let supported = supported_flags_when(true, true);
        assert_eq!(FLAG_UNSTABLE_SPILLED_ROW_IDS & !supported, 0);
    }

    #[test]
    fn test_apply_feature_flags_sets_overlay_flag() {
        use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
        use crate::format::{DataFile, DataStorageFormat, Fragment};
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use roaring::RoaringBitmap;
        use std::collections::HashMap;
        use std::sync::Arc;

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "id",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let mut fragment = Fragment::new(0);
        fragment.overlays = vec![DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![0], None),
            coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
            committed_version: 1,
        }];
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        apply_feature_flags(&mut manifest, false, false).unwrap();
        assert_ne!(
            manifest.reader_feature_flags & FLAG_UNSTABLE_DATA_OVERLAY_FILES,
            0
        );
        assert_ne!(
            manifest.writer_feature_flags & FLAG_UNSTABLE_DATA_OVERLAY_FILES,
            0
        );
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(can_write_dataset(super::FLAG_STABLE_ROW_IDS));
        assert!(can_write_dataset(super::FLAG_USE_V2_FORMAT_DEPRECATED));
        assert!(can_write_dataset(super::FLAG_TABLE_CONFIG));
        assert!(can_write_dataset(super::FLAG_BASE_PATHS));
        assert!(can_write_dataset(super::FLAG_DISABLE_TRANSACTION_FILE));
        // Overlay support is gated on the build profile / env opt-in, so the
        // flag is writable exactly when overlays are enabled (see
        // test_data_overlay_flag_release_gating for the full policy).
        assert_eq!(
            can_write_dataset(super::FLAG_UNSTABLE_DATA_OVERLAY_FILES),
            data_overlay_files_enabled()
        );
        assert!(can_write_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
                | super::FLAG_TABLE_CONFIG
                | super::FLAG_BASE_PATHS
        ));
        assert!(!can_write_dataset(super::FLAG_UNKNOWN));
    }

    #[test]
    fn test_base_paths_feature_flags() {
        use crate::format::{DataStorageFormat, Manifest};
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use std::collections::HashMap;
        use std::sync::Arc;
        // Create a basic schema for testing
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "test_field",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        // Test 1: Normal dataset (no base_paths) should not have FLAG_BASE_PATHS
        let mut normal_manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(), // Empty base_paths
        );
        apply_feature_flags(&mut normal_manifest, false, false).unwrap();
        assert_eq!(normal_manifest.reader_feature_flags & FLAG_BASE_PATHS, 0);
        assert_eq!(normal_manifest.writer_feature_flags & FLAG_BASE_PATHS, 0);
        // Test 2: Dataset with base_paths (shallow clone or multi-base) should have FLAG_BASE_PATHS
        let mut base_paths: HashMap<u32, BasePath> = HashMap::new();
        base_paths.insert(
            1,
            BasePath::new(
                1,
                "file:///path/to/original".to_string(),
                Some("test_ref".to_string()),
                true,
            ),
        );
        let mut multi_base_manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            base_paths,
        );
        apply_feature_flags(&mut multi_base_manifest, false, false).unwrap();
        assert_ne!(
            multi_base_manifest.reader_feature_flags & FLAG_BASE_PATHS,
            0
        );
        assert_ne!(
            multi_base_manifest.writer_feature_flags & FLAG_BASE_PATHS,
            0
        );
    }
    #[test]
    fn inheriting_carries_sticky_paired_bits_from_the_source() {
        let mut source = empty_manifest();
        source.reader_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;
        source.writer_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;
        // A fresh destination models any derived manifest before inheritance.
        let mut destination = empty_manifest();

        inherit_sticky_feature_flags(&mut destination, &source).unwrap();

        assert_ne!(
            destination.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_ne!(
            destination.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
    }

    #[test]
    fn inheriting_refuses_a_half_set_source() {
        for (reader, writer) in [
            (FLAG_MIXED_DATA_FILE_VERSIONS, 0),
            (0, FLAG_MIXED_DATA_FILE_VERSIONS),
        ] {
            let mut source = empty_manifest();
            source.reader_feature_flags = reader;
            source.writer_feature_flags = writer;
            let mut destination = empty_manifest();

            let err = inherit_sticky_feature_flags(&mut destination, &source).unwrap_err();

            assert!(err.to_string().contains("only one of"), "{err}");
        }
    }

    #[test]
    fn apply_feature_flags_carries_sticky_paired_bits_across_its_reset() {
        let mut manifest = empty_manifest();
        manifest.reader_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;
        manifest.writer_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;

        apply_feature_flags(&mut manifest, false, false).unwrap();

        assert_ne!(
            manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_ne!(
            manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
    }

    #[test]
    fn apply_feature_flags_rejects_half_set_sticky_bits() {
        let mut manifest = empty_manifest();
        manifest.reader_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;

        let err = apply_feature_flags(&mut manifest, false, false).unwrap_err();

        assert!(matches!(err, Error::CorruptFile { .. }));
        assert!(err.to_string().contains("only one of"), "{err}");
    }

    #[test]
    fn writer_gate_rejects_reserved_mixed_capability() {
        let mut manifest = empty_manifest();
        manifest.reader_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;
        manifest.writer_feature_flags = FLAG_MIXED_DATA_FILE_VERSIONS;

        let err = ensure_can_write_manifest(&manifest).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }));
        assert!(err.to_string().contains("cannot be written"), "{err}");
    }

    fn empty_manifest() -> Manifest {
        use crate::format::DataStorageFormat;
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use std::collections::HashMap;
        use std::sync::Arc;

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]);
        Manifest::new(
            Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        )
    }

    /// A build that does not know the bit must refuse the table rather than
    /// continue with legacy semantics.
    #[test]
    fn mixed_capability_remains_at_the_unknown_boundary() {
        assert!(can_read_dataset(FLAG_COVERED_INDEX_METADATA));
        assert!(can_write_dataset(FLAG_COVERED_INDEX_METADATA));
        assert!(!can_read_dataset(FLAG_MIXED_DATA_FILE_VERSIONS));
        assert!(!can_write_dataset(FLAG_MIXED_DATA_FILE_VERSIONS));
        assert_eq!(FLAG_MIXED_DATA_FILE_VERSIONS, FLAG_UNKNOWN);
    }
}
