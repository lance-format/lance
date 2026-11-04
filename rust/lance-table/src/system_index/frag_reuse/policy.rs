// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a table's state allows a fragment reuse writer to do.
//!
//! Every writer that records, carries or refuses fragment reuse history asks
//! [`FriPolicy`] instead of inspecting flags and entries itself, so a rule
//! about the table (which record format, whether a rewrite may remap
//! eagerly) is decided in one place. The policy picks the history a rewrite
//! is recorded onto ([`FriPolicy::recording_history`]); what that history
//! may record is then refused by
//! [`FragReuseHistory::append`](super::history::FragReuseHistory::append).
//!
//! ```
//! use lance_table::format::Manifest;
//! use lance_table::system_index::frag_reuse::history::HistoryFormat;
//! use lance_table::system_index::frag_reuse::policy::FriPolicy;
//!
//! # fn example(manifest: &Manifest) {
//! let policy = FriPolicy::new(manifest, &[]);
//! assert_eq!(policy.history_format(), HistoryFormat::Legacy);
//! # }
//! ```

use lance_core::{Error, Result};

use super::FRAG_REUSE_INDEX_NAME;
use super::history::{FragReuseHistory, HistoryFormat, RewriteRecord};
use super::metadata::{is_tagged, uses_tagged_fri};
use crate::format::{IndexMetadata, Manifest};

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
    pub fn history_format(&self) -> HistoryFormat {
        if uses_tagged_fri(self.manifest, self.entry) {
            HistoryFormat::Tagged
        } else {
            HistoryFormat::Legacy
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

    /// The history a rewrite on this table appends to: `current`, the
    /// table's history, or an empty one in [`Self::history_format`] when it
    /// has none. It is read as the table reads it and lifted when the table
    /// records tagged history or only the tagged format can record `rewrite`,
    /// so the format of what [`FragReuseHistory::append`] records is decided
    /// here.
    ///
    /// Refused, with `NotSupported`, on a table with stable row IDs whose
    /// rewrite would need the tagged format: tagged histories are
    /// address-based.
    pub fn recording_history(
        &self,
        current: Option<FragReuseHistory>,
        rewrite: &RewriteRecord,
    ) -> Result<FragReuseHistory> {
        let format = self.history_format();
        let mut history = current.unwrap_or_else(|| FragReuseHistory::empty(format));
        if self.manifest.uses_stable_row_ids() {
            history = history.on_stable_row_ids()?;
        }
        if format == HistoryFormat::Tagged || rewrite.needs_tagged_format() {
            history = history.lift()?;
        }
        Ok(history)
    }

    /// Refuse, before any rewrite runs, a table whose history no rewrite
    /// could be recorded in (see [`Self::recording_history`]).
    pub fn require_recording_allowed(&self) -> Result<()> {
        self.recording_history(None, &RewriteRecord::new(self.manifest.version, Vec::new()))
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::{FLAG_FRAGMENT_REUSE_INDEX, FLAG_STABLE_ROW_IDS};
    use crate::system_index::frag_reuse::history::tests::{
        MAP_A, ordered as ordered_transition, partition, whole,
    };

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
    fn the_sticky_flag_decides_the_history_format_without_an_entry() {
        let untagged = manifest(false);
        assert_eq!(
            FriPolicy::new(&untagged, &[]).history_format(),
            HistoryFormat::Legacy
        );
        let legacy = [entry(0)];
        assert_eq!(
            FriPolicy::new(&untagged, &legacy).history_format(),
            HistoryFormat::Legacy
        );

        let flagged = manifest(true);
        assert_eq!(
            FriPolicy::new(&flagged, &[]).history_format(),
            HistoryFormat::Tagged
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

    /// The table records tagged history once it is tagged, or for a rewrite
    /// only the tagged format can hold; a table with stable row ids never
    /// does.
    #[test]
    fn recording_history_picks_the_format_a_rewrite_is_recorded_in() {
        let ordered = RewriteRecord::new(1, vec![]);
        let stable_partition = RewriteRecord::new(1, vec![partition(0, 1, MAP_A)]);
        let format = |manifest: &Manifest, rewrite: &RewriteRecord| {
            FriPolicy::new(manifest, &[])
                .recording_history(None, rewrite)
                .map(|history| history.format())
        };

        let untagged = manifest(false);
        assert_eq!(format(&untagged, &ordered).unwrap(), HistoryFormat::Legacy);
        // Carrying a version 0 entry forward on an untagged table writes
        // version 0 again.
        let legacy = [entry(0)];
        let rewrite = RewriteRecord::new(2, vec![ordered_transition(&[whole(0, 1)], &[(1, 1)])]);
        let carried = FriPolicy::new(&untagged, &legacy)
            .recording_history(
                Some(FragReuseHistory::empty(HistoryFormat::Legacy)),
                &rewrite,
            )
            .unwrap()
            .append(rewrite)
            .unwrap()
            .encode(2)
            .entry;
        assert_eq!(carried.index_version, 0);
        assert!(!is_tagged(&carried));
        assert_eq!(
            format(&untagged, &stable_partition).unwrap(),
            HistoryFormat::Tagged
        );
        let flagged = manifest(true);
        assert_eq!(format(&flagged, &ordered).unwrap(), HistoryFormat::Tagged);
        // A legacy entry under the sticky flag is lifted.
        let lifted = FriPolicy::new(&flagged, &[])
            .recording_history(
                Some(FragReuseHistory::empty(HistoryFormat::Legacy)),
                &ordered,
            )
            .unwrap();
        assert_eq!(lifted.format(), HistoryFormat::Tagged);

        let mut stable_row_ids = manifest(false);
        stable_row_ids.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        let history = FriPolicy::new(&stable_row_ids, &[])
            .recording_history(None, &ordered)
            .unwrap();
        assert_eq!(history.format(), HistoryFormat::Legacy);
        FriPolicy::new(&stable_row_ids, &[])
            .require_recording_allowed()
            .unwrap();
        for result in [
            history.lift().map(|_| ()),
            format(&stable_row_ids, &stable_partition).map(|_| ()),
        ] {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            assert!(error.to_string().contains("stable-row-id"), "{error}");
        }
        stable_row_ids.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        let error = FriPolicy::new(&stable_row_ids, &[])
            .require_recording_allowed()
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("tagged"), "{error}");
    }
}
