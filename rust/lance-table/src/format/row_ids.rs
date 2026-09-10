// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::pb;

/// A reference to a part of a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct ExternalFile {
    pub path: String,
    pub offset: u64,
    pub size: u64,
}

/// A fragment's row id sequence, encoded inline in the manifest.
///
/// Carries a memoized [`digest`](Self::digest) of the encoded bytes. The digest
/// identifies *which* sequence these bytes are, which is what the row id
/// sequence cache keys on: a fragment id alone does not identify a sequence,
/// because fragment ids are reused across dataset generations.
///
/// The digest is memoized because computing it is proportional to the encoded
/// size. A run-encoded sequence is a handful of bytes per run, but a heavily
/// fragmented one is array-encoded at 8 bytes per row, and the cache is
/// consulted on every scan, count, prefilter and index load.
///
/// Bytes and memo share one immutable allocation, so cloning shares both. That
/// is what makes the memo worth having: callers clone `Fragment` before loading
/// its sequence (see `count_from_mask`), and a memo held per clone would be
/// filled and dropped by each scan, rehashing the whole sequence every time.
/// Sharing also keeps a cloned fragment from duplicating the encoded bytes.
///
/// The digest lives *with* the bytes rather than beside them so the two cannot
/// drift: several write paths replace a fragment's `row_id_meta` after the
/// fragment is built, and a digest that outlived its bytes would silently
/// resolve to another generation's sequence.
#[derive(Clone)]
pub struct InlineRowIds {
    inner: Arc<InlineRowIdsInner>,
}

struct InlineRowIdsInner {
    data: Vec<u8>,
    digest: OnceLock<[u8; 32]>,
}

impl InlineRowIds {
    /// Digest of the encoded bytes, computed on first use and shared by clones.
    pub fn digest(&self) -> &[u8; 32] {
        self.inner
            .digest
            .get_or_init(|| blake3::hash(&self.inner.data).into())
    }
}

impl From<Vec<u8>> for InlineRowIds {
    fn from(data: Vec<u8>) -> Self {
        Self {
            inner: Arc::new(InlineRowIdsInner {
                data,
                digest: OnceLock::new(),
            }),
        }
    }
}

impl Deref for InlineRowIds {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

// Debug, equality and serialization all present the bytes alone: the memo is a
// derived value and must not show up in output, comparisons or the manifest.
impl std::fmt::Debug for InlineRowIds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.data.fmt(f)
    }
}

impl PartialEq for InlineRowIds {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || self.inner.data == other.inner.data
    }
}

impl Eq for InlineRowIds {}

impl Serialize for InlineRowIds {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.inner.data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InlineRowIds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Vec::<u8>::deserialize(deserializer).map(Self::from)
    }
}

impl DeepSizeOf for InlineRowIds {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        // Delegate to the `Arc` so clones sharing one allocation are counted once.
        self.inner.deep_size_of_children(context)
    }
}

impl DeepSizeOf for InlineRowIdsInner {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.data.deep_size_of_children(context)
    }
}

/// Metadata about location of the row id sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum RowIdMeta {
    Inline(InlineRowIds),
    External(ExternalFile),
}

impl TryFrom<pb::data_fragment::RowIdSequence> for RowIdMeta {
    type Error = Error;

    fn try_from(value: pb::data_fragment::RowIdSequence) -> Result<Self> {
        match value {
            pb::data_fragment::RowIdSequence::InlineRowIds(data) => Ok(Self::Inline(data.into())),
            pb::data_fragment::RowIdSequence::ExternalRowIds(file) => {
                Ok(Self::External(ExternalFile {
                    path: file.path.clone(),
                    offset: file.offset,
                    size: file.size,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Fragment;

    #[test]
    fn inline_row_ids_digest_identifies_contents() {
        let first = InlineRowIds::from(vec![1, 2, 3]);
        let same = InlineRowIds::from(vec![1, 2, 3]);
        let other = InlineRowIds::from(vec![1, 2, 4]);

        // The digest is what the row id sequence cache keys on, so it must
        // follow the bytes exactly: same bytes, same key; any change, new key.
        assert_eq!(first.digest(), same.digest());
        assert_ne!(first.digest(), other.digest());
        assert_eq!(first, same);
        assert_ne!(first, other);

        // Memoized on first use, so repeat use must return the same digest.
        let memoized = *first.digest();
        assert_eq!(first.digest(), &memoized);
    }

    #[test]
    fn inline_row_ids_clones_share_bytes_and_memo() {
        let first = InlineRowIds::from(vec![1, 2, 3]);
        let cloned = first.clone();

        // Callers clone `Fragment` before loading its row id sequence, so a memo
        // held per clone would be filled and dropped by each scan and rehash the
        // whole sequence every time. Sharing one allocation is what makes the
        // memo pay off, and it keeps clones from duplicating the bytes.
        assert!(std::ptr::eq(first.as_ptr(), cloned.as_ptr()));
        assert!(std::ptr::eq(first.digest(), cloned.digest()));

        // Guard against the assertions above passing vacuously: only clones
        // share storage, equal bytes built separately do not.
        let separate = InlineRowIds::from(vec![1, 2, 3]);
        assert_eq!(first, separate);
        assert!(!std::ptr::eq(first.as_ptr(), separate.as_ptr()));

        // Computing through one clone must be visible from the other.
        let fresh = InlineRowIds::from(vec![4, 5, 6]);
        let fresh_clone = fresh.clone();
        let via_clone = *fresh_clone.digest();
        assert_eq!(fresh.digest(), &via_clone);
    }

    #[test]
    fn inline_row_ids_serializes_as_bare_bytes() {
        // The manifest is a stable format: the memo must not reach the wire.
        let meta = RowIdMeta::Inline(InlineRowIds::from(vec![7, 8, 9]));
        let json = serde_json::to_string(&meta).unwrap();
        assert_eq!(json, r#"{"Inline":[7,8,9]}"#);
        assert_eq!(serde_json::from_str::<RowIdMeta>(&json).unwrap(), meta);

        // ...and round-trips through protobuf unchanged.
        let fragment = Fragment {
            row_id_meta: Some(meta.clone()),
            ..Fragment::new(0)
        };
        let restored = Fragment::try_from(pb::DataFragment::from(&fragment)).unwrap();
        assert_eq!(restored.row_id_meta, Some(meta));
    }
}
