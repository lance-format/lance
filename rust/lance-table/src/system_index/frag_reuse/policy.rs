// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a table's state allows a fragment reuse writer to do.
//!
//! Every writer that records, carries or refuses fragment reuse history asks
//! [`FriPolicy`] instead of inspecting flags and entries itself, so a rule
//! about the table (which record format, whether a rewrite may remap eagerly,
//! whether tagged records are allowed) is decided in one place.
//!
//! ```
//! use lance_table::format::Manifest;
//! use lance_table::system_index::frag_reuse::policy::{FriPolicy, RecordForm};
//!
//! # fn example(manifest: &Manifest) {
//! let policy = FriPolicy::new(manifest, &[]);
//! assert_eq!(policy.record_form(), RecordForm::Legacy);
//! # }
//! ```

use lance_core::{Error, Result};

use super::FRAG_REUSE_INDEX_NAME;
use super::metadata::{is_tagged, uses_tagged_fri};
use crate::format::{Fragment, IndexMetadata, Manifest};

/// The format a new fragment reuse record takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordForm {
    /// `index_version` 0 compaction versions, which every released reader
    /// and writer understands.
    Legacy,
    /// `index_version` 1 transitions. Once a table carries them it keeps the
    /// format (the feature flag is sticky).
    Tagged,
}

impl RecordForm {
    /// The `IndexMetadata::index_version` of an entry in this format.
    pub fn index_version(self) -> i32 {
        match self {
            Self::Legacy => 0,
            Self::Tagged => 1,
        }
    }
}

/// The fragment reuse rules of one table snapshot.
#[derive(Debug, Clone, Copy)]
pub struct FriPolicy<'a> {
    manifest: &'a Manifest,
    entry: Option<&'a IndexMetadata>,
}

impl<'a> FriPolicy<'a> {
    /// The policy of the snapshot `manifest`, whose stored index list is
    /// `indices`.
    pub fn new(manifest: &'a Manifest, indices: &'a [IndexMetadata]) -> Self {
        Self {
            manifest,
            entry: indices
                .iter()
                .find(|index| index.name == FRAG_REUSE_INDEX_NAME),
        }
    }

    /// The table's fragment reuse entry, if it has one.
    pub fn entry(&self) -> Option<&'a IndexMetadata> {
        self.entry
    }

    /// Whether the table carries a tagged history.
    pub fn has_tagged_history(&self) -> bool {
        self.entry.is_some_and(is_tagged)
    }

    /// The format a rewrite records in: tagged once the table is tagged (by
    /// its sticky flag or its entry), the legacy format otherwise.
    pub fn record_form(&self) -> RecordForm {
        if uses_tagged_fri(self.manifest, self.entry) {
            RecordForm::Tagged
        } else {
            RecordForm::Legacy
        }
    }

    /// Refuse a rewrite that would remap indices eagerly over a tagged
    /// history: the history depends on index bitmaps keeping their retired
    /// provenance, which an eager remap rewrites.
    pub fn require_rewrite_allowed(&self, defers_index_remap: bool) -> Result<()> {
        if self.has_tagged_history() && !defers_index_remap {
            return Err(Error::not_supported(
                "Compacting a table with a tagged fragment reuse history requires \
                 defer_index_remap; set defer_index_remap, or upgrade to a writer that \
                 remaps tagged histories eagerly",
            ));
        }
        Ok(())
    }

    /// Refuse recording tagged records on a table that cannot carry them:
    /// tagged histories translate row addresses, which stable row ids
    /// replace.
    pub fn require_tagged_records_allowed(&self) -> Result<()> {
        if self.manifest.uses_stable_row_ids() {
            return Err(Error::not_supported(
                "Tagged fragment reuse histories are address-based and excluded on \
                 stable-row-id datasets; this rewrite cannot carry transition intent here",
            ));
        }
        Ok(())
    }

    /// Refuse a tagged transition over sources that carry data overlays.
    ///
    /// Rewriting such a source materializes the overlaid values, while an
    /// index built before the overlay keeps the source as provenance and
    /// would serve the destination through translation with the old values.
    /// The legacy format handles this by dropping destination ids from stale
    /// bitmaps, which has no effect on tagged provenance.
    pub fn require_tagged_sources_supported<'f>(
        &self,
        sources: impl IntoIterator<Item = &'f Fragment>,
    ) -> Result<()> {
        if let Some(overlaid) = sources
            .into_iter()
            .find(|fragment| !fragment.overlays.is_empty())
        {
            return Err(Error::not_supported(format!(
                "source fragment {} carries data overlay files; a rewrite recording a tagged \
                 fragment reuse transition would materialize the overlaid values while indices \
                 keep translated coverage over the old addresses. Compact the overlays away or \
                 rebuild the covering indices eagerly first",
                overlaid.id
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;

    fn entry(index_version: i32) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: FRAG_REUSE_INDEX_NAME.to_string(),
            fields: vec![],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap: Some(Default::default()),
            index_details: None,
            index_version,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    fn manifest(is_flagged: bool) -> Manifest {
        let mut manifest = crate::format::Manifest::new(
            lance_core::datatypes::Schema::default(),
            std::sync::Arc::new(vec![]),
            crate::format::DataStorageFormat::default(),
            Default::default(),
        );
        if is_flagged {
            manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
            manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        }
        manifest
    }

    #[test]
    fn the_sticky_flag_decides_the_record_form_without_an_entry() {
        let untagged = manifest(false);
        assert_eq!(
            FriPolicy::new(&untagged, &[]).record_form(),
            RecordForm::Legacy
        );
        let legacy = [entry(0)];
        assert_eq!(
            FriPolicy::new(&untagged, &legacy).record_form(),
            RecordForm::Legacy
        );

        let flagged = manifest(true);
        assert_eq!(
            FriPolicy::new(&flagged, &[]).record_form(),
            RecordForm::Tagged
        );
        assert!(!FriPolicy::new(&flagged, &[]).has_tagged_history());
        let tagged = [entry(1)];
        assert!(FriPolicy::new(&flagged, &tagged).has_tagged_history());
    }

    #[test]
    fn only_a_tagged_history_requires_deferred_remap() {
        let flagged = manifest(true);
        let tagged = [entry(1)];
        let error = FriPolicy::new(&flagged, &tagged)
            .require_rewrite_allowed(false)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        FriPolicy::new(&flagged, &tagged)
            .require_rewrite_allowed(true)
            .unwrap();
        // A drained history leaves the flag but nothing to translate.
        FriPolicy::new(&flagged, &[])
            .require_rewrite_allowed(false)
            .unwrap();
    }
}
