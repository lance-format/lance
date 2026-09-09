// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata for index

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::StreamExt;
use lance_core::deepsize::DeepSizeOf;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::pb;
use lance_core::cache::{CacheEntryReader, CacheEntryWriter};
use lance_core::{Error, Result};

/// Metadata about a single file within an index segment.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct IndexFile {
    /// Path relative to the index directory (e.g., "index.idx", "auxiliary.idx")
    pub path: String,
    /// Size of the file in bytes
    pub size_bytes: u64,
}

/// Index metadata
#[derive(Debug, Clone, PartialEq)]
pub struct IndexMetadata {
    /// Unique ID across all dataset versions.
    pub uuid: Uuid,

    /// Fields to build the index.
    ///
    /// `fields[0]` is always a column the index is keyed on. Trailing entries
    /// may instead be merely carried, not keyed on -- see [`Self::covering_fields`].
    pub fields: Vec<i32>,

    /// Fields whose values this index carries but is not keyed on.
    ///
    /// Always a suffix of [`Self::fields`], and never all of it, so
    /// `fields[0]` is always a column the index is keyed on. Empty for an
    /// index that carries no extra columns.
    ///
    /// These ids also appear in [`Self::fields`]. That is deliberate: every
    /// consumer that reads `fields` as the index's dependency set then covers
    /// them with no change.
    pub covering_fields: Vec<i32>,

    /// Human readable index name
    pub name: String,

    /// The version of the dataset this index was last updated on
    ///
    /// This is set when the index is created (based on the version used to train the index)
    /// This is updated when the index is updated or remapped
    pub dataset_version: u64,

    /// The fragment ids this index covers.
    ///
    /// This may contain fragment ids that no longer exist in the dataset.
    ///
    /// If this is None, then this is unknown.
    pub fragment_bitmap: Option<RoaringBitmap>,

    /// Metadata specific to the index type
    ///
    /// This is an Option because older versions of Lance may not have this defined.  However, it should always
    /// be present in newer versions.
    pub index_details: Option<Arc<prost_types::Any>>,

    /// The index version.
    pub index_version: i32,

    /// Timestamp when the index was created
    ///
    /// This field is optional for backward compatibility. For existing indices created before
    /// this field was added, this will be None.
    pub created_at: Option<DateTime<Utc>>,

    /// The base path index of the index files. Used when the index is imported or referred from another dataset.
    /// Lance uses it as key of the base_paths field in Manifest to determine the actual base path of the index files.
    pub base_id: Option<u32>,

    /// List of files and their sizes for this index segment.
    /// This enables skipping HEAD calls when opening indices and provides
    /// visibility into index storage size via describe_indices().
    /// This is None if the file sizes are unknown. This happens for indices created
    /// before this field was added.
    pub files: Option<Vec<IndexFile>>,
}

impl IndexMetadata {
    pub fn effective_fragment_bitmap(
        &self,
        existing_fragments: &RoaringBitmap,
    ) -> Option<RoaringBitmap> {
        let fragment_bitmap = self.fragment_bitmap.as_ref()?;
        Some(fragment_bitmap & existing_fragments)
    }

    /// Returns a map of relative file paths to their sizes.
    /// Returns an empty map if file information is not available.
    pub fn file_size_map(&self) -> HashMap<String, u64> {
        self.files
            .as_ref()
            .map(|files| {
                files
                    .iter()
                    .map(|f| (f.path.clone(), f.size_bytes))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the total size of all files in this index segment in bytes.
    /// Returns None if file information is not available.
    pub fn total_size_bytes(&self) -> Option<u64> {
        self.files
            .as_ref()
            .map(|files| files.iter().map(|f| f.size_bytes).sum())
    }

    /// Returns the set of fragments which are part of the fragment bitmap
    /// but no longer in the dataset.
    pub fn deleted_fragment_bitmap(
        &self,
        existing_fragments: &RoaringBitmap,
    ) -> Option<RoaringBitmap> {
        let fragment_bitmap = self.fragment_bitmap.as_ref()?;
        Some(fragment_bitmap - existing_fragments)
    }

    /// True when the index reports matches as physical row addresses rather than row ids
    /// (`ScalarIndex::results_are_row_addresses`).
    ///
    /// Such an index cannot follow its data through a rewrite: the addresses it stores
    /// name fragments and offsets, and neither kind supports remap.
    pub fn results_are_row_addrs(&self) -> bool {
        self.index_details.as_ref().is_some_and(|details| {
            let is_fm = details
                .type_url
                .rsplit_once('/')
                .is_some_and(|(_, details_type_name)| {
                    details_type_name.eq_ignore_ascii_case("lance.index.pb.FMIndexDetails")
                });
            details.type_url.ends_with("ZoneMapIndexDetails")
                || details.type_url.ends_with("BloomFilterIndexDetails")
                || is_fm
        })
    }

    /// The prefix of [`Self::fields`] this index is keyed on, with the carried
    /// columns of [`Self::covering_fields`] removed.
    ///
    /// Only this prefix decides which column an index answers for; the full
    /// `fields` vector answers what invalidates it. Empty for a system index
    /// that declares no fields, and empty for a declaration longer than
    /// `fields`: decoding validates, but metadata built by a caller this build
    /// never validated does not, and failing closed beats an underflow.
    pub fn keyed_fields(&self) -> &[i32] {
        let keyed = self.fields.len().saturating_sub(self.covering_fields.len());
        &self.fields[..keyed]
    }

    /// The single column this index is keyed on, or `None` when it is keyed on
    /// several -- a genuinely composite index -- or on none at all.
    ///
    /// Most selection paths are only defined for one keyed column, so they can
    /// compare this against the column they are resolving.
    pub fn keyed_field(&self) -> Option<i32> {
        match self.keyed_fields() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// Check the covering declaration against [`Self::fields`].
    ///
    /// Carried columns must be a suffix of `fields` and must not consume all of
    /// it, so `fields[0]` is always a column the index is keyed on. An empty
    /// declaration is always valid, which is what keeps the system indices --
    /// `mem_wal` and `frag_reuse`, both of which commit no fields at all --
    /// passing this check.
    ///
    /// The rules are checked from most to least specific, because one bad
    /// declaration usually trips several: an id that is not a field at all is
    /// reported ahead of the length and ordering rules, which would otherwise
    /// name a consequence instead of the cause.
    pub fn validate_covering_fields(&self) -> Result<()> {
        if self.covering_fields.is_empty() {
            return Ok(());
        }

        let missing: Vec<i32> = self
            .covering_fields
            .iter()
            .copied()
            .filter(|f| !self.fields.contains(f))
            .collect();
        if !missing.is_empty() {
            return Err(Error::invalid_input(format!(
                "index '{}' declares covering fields {:?} but {:?} are not \
                 among its fields {:?}",
                self.name, self.covering_fields, missing, self.fields,
            )));
        }

        // A column carried twice would be projected twice. The suffix check
        // below cannot stand in for this, because `fields` may repeat the id in
        // the same positions -- `fields = [7, 11, 11]` has `[11, 11]` as a
        // genuine tail.
        let mut seen = HashSet::with_capacity(self.covering_fields.len());
        if let Some(duplicate) = self.covering_fields.iter().find(|f| !seen.insert(**f)) {
            return Err(Error::invalid_input(format!(
                "index '{}' declares covering field {} more than once in {:?}",
                self.name, duplicate, self.covering_fields,
            )));
        }

        if self.covering_fields.len() >= self.fields.len() {
            return Err(Error::invalid_input(format!(
                "index '{}' declares covering fields {:?} but its fields are {:?}; \
                 at least one field must remain indexed",
                self.name, self.covering_fields, self.fields,
            )));
        }

        let suffix_start = self.fields.len() - self.covering_fields.len();
        if self.fields[suffix_start..] != self.covering_fields[..] {
            return Err(Error::invalid_input(format!(
                "index '{}' declares covering fields {:?} which are not the trailing \
                 entries of {:?}; covering fields must come last",
                self.name, self.covering_fields, self.fields,
            )));
        }

        Ok(())
    }
}

impl DeepSizeOf for IndexMetadata {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.fields.deep_size_of_children(context)
            + self.covering_fields.deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.dataset_version.deep_size_of_children(context)
            + self
                .fragment_bitmap
                .as_ref()
                .map(|fragment_bitmap| fragment_bitmap.serialized_size())
                .unwrap_or(0)
            + self.files.deep_size_of_children(context)
    }
}

impl TryFrom<pb::IndexMetadata> for IndexMetadata {
    type Error = Error;

    fn try_from(proto: pb::IndexMetadata) -> Result<Self> {
        let fragment_bitmap = if proto.fragment_bitmap.is_empty() {
            None
        } else {
            Some(RoaringBitmap::deserialize_from(
                &mut proto.fragment_bitmap.as_slice(),
            )?)
        };

        let files = if proto.files.is_empty() {
            None
        } else {
            Some(
                proto
                    .files
                    .into_iter()
                    .map(|f| IndexFile {
                        path: f.path,
                        size_bytes: f.size_bytes,
                    })
                    .collect(),
            )
        };

        let metadata = Self {
            uuid: proto.uuid.as_ref().map(Uuid::try_from).ok_or_else(|| {
                Error::invalid_input("uuid field does not exist in Index metadata".to_string())
            })??,
            name: proto.name,
            fields: proto.fields,
            covering_fields: proto.covering_fields,
            dataset_version: proto.dataset_version,
            fragment_bitmap,
            index_details: proto.index_details.map(Arc::new),
            index_version: proto.index_version.unwrap_or_default(),
            created_at: proto.created_at.map(|ts| {
                DateTime::from_timestamp_millis(ts as i64)
                    .expect("Invalid timestamp in index metadata")
            }),
            base_id: proto.base_id,
            files,
        };

        // This is the single boundary between manifest bytes and
        // `IndexMetadata`, so validating once here is what lets every reader
        // treat the declaration as a trailing slice of `fields`. A manifest
        // that fails this was written by something that did not follow the
        // format contract; refuse it rather than let each use site quietly
        // ignore the index.
        metadata.validate_covering_fields()?;

        Ok(metadata)
    }
}

impl From<&IndexMetadata> for pb::IndexMetadata {
    fn from(idx: &IndexMetadata) -> Self {
        let mut fragment_bitmap = Vec::new();
        if let Some(bitmap) = &idx.fragment_bitmap {
            // Fragment ids are allocated monotonically, so index coverage is
            // highly contiguous. Run containers are part of the standard
            // roaring serialization format, so converting eligible containers
            // to runs before writing shrinks the bitmap from O(fragments) to
            // O(runs) bytes.
            let mut bitmap = bitmap.clone();
            bitmap.optimize();
            if let Err(e) = bitmap.serialize_into(&mut fragment_bitmap) {
                // In theory, this should never error. But if we do, just
                // recover gracefully.
                log::error!("Failed to serialize fragment bitmap: {}", e);
                fragment_bitmap.clear();
            }
        }

        let files = idx
            .files
            .as_ref()
            .map(|files| {
                files
                    .iter()
                    .map(|f| pb::IndexFile {
                        path: f.path.clone(),
                        size_bytes: f.size_bytes,
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            uuid: Some((&idx.uuid).into()),
            name: idx.name.clone(),
            fields: idx.fields.clone(),
            covering_fields: idx.covering_fields.clone(),
            dataset_version: idx.dataset_version,
            fragment_bitmap,
            index_details: idx
                .index_details
                .as_ref()
                .map(|details| details.as_ref().clone()),
            index_version: Some(idx.index_version),
            created_at: idx.created_at.map(|dt| dt.timestamp_millis() as u64),
            base_id: idx.base_id,
            files,
        }
    }
}

/// Returns a [`CacheCodec`](lance_core::cache::CacheCodec) for `Vec<IndexMetadata>`.
///
/// Uses `pb::IndexSection` (which wraps `repeated IndexMetadata`) as the wire
/// format, reusing the existing `TryFrom`/`From` conversions.
///
/// Uses [`CacheCodec::new`](lance_core::cache::CacheCodec::new) because the
/// orphan rule prevents `impl CacheCodecImpl for Vec<IndexMetadata>`.
type ArcAny = Arc<dyn std::any::Any + Send + Sync>;

/// Stable type identifier for the `Vec<IndexMetadata>` cache entry.
const INDEX_METADATA_TYPE_ID: &str = "lance.table.IndexMetadataList";
/// Body schema version written by this build.
const INDEX_METADATA_VERSION: u32 = 1;

fn serialize_index_metadata(
    any: &ArcAny,
    writer: &mut CacheEntryWriter<'_>,
) -> lance_core::Result<()> {
    let vec = any
        .downcast_ref::<Vec<IndexMetadata>>()
        .expect("index_metadata_codec: wrong type (this is a bug in the cache layer)");
    let section = pb::IndexSection {
        indices: vec.iter().map(pb::IndexMetadata::from).collect(),
    };
    writer.write_header(&section)
}

fn deserialize_index_metadata(reader: &mut CacheEntryReader<'_>) -> lance_core::Result<ArcAny> {
    let section: pb::IndexSection = reader.read_header()?;
    let indices: Vec<IndexMetadata> = section
        .indices
        .into_iter()
        .map(IndexMetadata::try_from)
        .collect::<lance_core::Result<_>>()?;
    Ok(Arc::new(indices))
}

pub fn index_metadata_codec() -> lance_core::cache::CacheCodec {
    lance_core::cache::CacheCodec::new(
        INDEX_METADATA_TYPE_ID,
        INDEX_METADATA_VERSION,
        serialize_index_metadata,
        deserialize_index_metadata,
    )
}

/// List all files in an index directory with their sizes.
///
/// Returns a list of `IndexFile` structs containing relative paths and sizes.
/// This is used to capture file metadata after index creation/modification.
pub async fn list_index_files_with_sizes(
    object_store: &ObjectStore,
    index_dir: &Path,
) -> Result<Vec<IndexFile>> {
    let mut files = Vec::new();
    let mut stream = object_store.read_dir_all(index_dir, None);
    while let Some(meta) = stream.next().await {
        let meta = meta?;
        // Get relative path by stripping the index_dir prefix
        let relative_path = meta
            .location
            .as_ref()
            .strip_prefix(index_dir.as_ref())
            .map(|s| s.trim_start_matches('/').to_string())
            .unwrap_or_else(|| meta.location.filename().unwrap_or("").to_string());
        files.push(IndexFile {
            path: relative_path,
            size_bytes: meta.size,
        });
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::collections::HashMap;

    #[test]
    fn test_fragment_bitmap_serialized_run_optimized() {
        let bitmap = RoaringBitmap::from_sorted_iter(0..1_000_000).unwrap();
        let unoptimized_size = bitmap.serialized_size();

        let metadata = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "my_index".to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap: Some(bitmap.clone()),
            index_details: None,
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        };

        let proto = pb::IndexMetadata::from(&metadata);
        assert!(
            proto.fragment_bitmap.len() < unoptimized_size / 100,
            "expected run-optimized bitmap ({} bytes) to be <1% of the \
             unoptimized serialization ({} bytes)",
            proto.fragment_bitmap.len(),
            unoptimized_size
        );

        let recovered = IndexMetadata::try_from(proto).unwrap();
        assert_eq!(recovered.fragment_bitmap, Some(bitmap));
    }

    /// Demonstrates the pattern a disk-backed cache backend would use:
    /// serialize entries to bytes, store in a key-value map, then
    /// deserialize on retrieval.
    #[test]
    fn test_index_metadata_codec_roundtrip() {
        let codec = index_metadata_codec();

        let original = vec![
            IndexMetadata {
                uuid: Uuid::new_v4(),
                name: "my_index".to_string(),
                fields: vec![0, 1],
                covering_fields: vec![],
                dataset_version: 42,
                fragment_bitmap: Some(RoaringBitmap::from_iter([1, 2, 3])),
                index_details: None,
                index_version: 1,
                created_at: None,
                base_id: None,
                files: Some(vec![IndexFile {
                    path: "index.idx".to_string(),
                    size_bytes: 1024,
                }]),
            },
            IndexMetadata {
                uuid: Uuid::new_v4(),
                name: "second_index".to_string(),
                fields: vec![2],
                covering_fields: vec![],
                dataset_version: 43,
                fragment_bitmap: None,
                index_details: None,
                index_version: 2,
                created_at: None,
                base_id: Some(7),
                files: None,
            },
        ];

        // Simulate a disk-backed store: HashMap<String, Vec<u8>>
        let mut store: HashMap<String, Vec<u8>> = HashMap::new();

        // Serialize into the store
        let key = "dataset/v42/Vec<IndexMetadata>".to_string();
        let mut buf = Vec::new();
        let entry: Arc<dyn std::any::Any + Send + Sync> = Arc::new(original.clone());
        codec.serialize(&entry, &mut buf).unwrap();
        store.insert(key.clone(), buf);

        // Deserialize from the store
        let bytes = store.get(&key).unwrap();
        let recovered = codec
            .deserialize(&bytes::Bytes::copy_from_slice(bytes))
            .hit()
            .expect("entry should decode as a hit");
        let recovered = recovered
            .downcast::<Vec<IndexMetadata>>()
            .expect("downcast should succeed");

        assert_eq!(original.len(), recovered.len());
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert_eq!(orig.uuid, rec.uuid);
            assert_eq!(orig.name, rec.name);
            assert_eq!(orig.fields, rec.fields);
            assert_eq!(orig.dataset_version, rec.dataset_version);
            assert_eq!(orig.fragment_bitmap, rec.fragment_bitmap);
            assert_eq!(orig.index_version, rec.index_version);
            assert_eq!(orig.base_id, rec.base_id);
            assert_eq!(orig.files, rec.files);
        }
    }

    /// The covering declaration must survive both conversion directions.
    /// A dropped `covering_fields` would leave index files holding carried
    /// columns the manifest no longer names.
    #[test]
    fn test_covering_fields_survives_proto_roundtrip() {
        let original = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "covered".to_string(),
            fields: vec![7, 11, 13],
            covering_fields: vec![11, 13],
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };

        let proto = pb::IndexMetadata::from(&original);
        assert_eq!(proto.covering_fields, vec![11, 13]);

        let recovered = IndexMetadata::try_from(proto).unwrap();
        assert_eq!(recovered, original);
    }

    /// `TryFrom<pb::IndexMetadata>` is the only path from manifest bytes to
    /// `IndexMetadata`, so validating here is what lets every reader downstream
    /// assume the declaration really is a trailing slice of `fields`. Without
    /// it, a malformed declaration reaches each use site instead, where the
    /// keyed count saturates to zero and the index is silently ignored.
    #[test]
    fn test_try_from_proto_rejects_a_malformed_covering_declaration() {
        let mut proto = pb::IndexMetadata::from(&index_metadata_with(vec![7, 11], vec![11]));
        // The leading entry, not the trailing one: claims the keyed column is
        // carried.
        proto.covering_fields = vec![7];

        let err = IndexMetadata::try_from(proto)
            .expect_err("a malformed covering declaration must not decode");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {:?}",
            err
        );
        assert!(
            err.to_string().contains("must come last"),
            "unexpected message: {err}"
        );
    }

    /// Only the keyed prefix decides which column an index answers for, and
    /// nearly every selection path needs exactly one such column. Metadata read
    /// from a manifest this build never wrote can still be malformed, so both
    /// accessors must fail closed -- no keyed field -- rather than underflow.
    #[rstest]
    #[case::not_covered(vec![7], vec![], vec![7], Some(7))]
    #[case::covered(vec![7, 11], vec![11], vec![7], Some(7))]
    #[case::covered_multi(vec![7, 11, 13], vec![11, 13], vec![7], Some(7))]
    #[case::covered_composite(vec![7, 11, 13], vec![13], vec![7, 11], None)]
    #[case::composite(vec![7, 11], vec![], vec![7, 11], None)]
    #[case::system_index_no_fields(vec![], vec![], vec![], None)]
    #[case::malformed_longer_than_fields(vec![7], vec![11, 13], vec![], None)]
    fn test_keyed_fields(
        #[case] fields: Vec<i32>,
        #[case] covering_fields: Vec<i32>,
        #[case] expected_keyed: Vec<i32>,
        #[case] expected_single: Option<i32>,
    ) {
        let metadata = index_metadata_with(fields, covering_fields);

        assert_eq!(metadata.keyed_fields(), expected_keyed.as_slice());
        assert_eq!(metadata.keyed_field(), expected_single);
    }

    fn index_metadata_with(fields: Vec<i32>, covering_fields: Vec<i32>) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "idx".to_string(),
            fields,
            covering_fields,
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    #[rstest]
    #[case::zone_map("type.googleapis.com/lance.table.ZoneMapIndexDetails", true)]
    #[case::bloom_filter("type.googleapis.com/lance.index.pb.BloomFilterIndexDetails", true)]
    #[case::fm("type.googleapis.com/lance.index.pb.FMIndexDetails", true)]
    #[case::fm_case_insensitive("type.googleapis.com/LANCE.INDEX.PB.FMINDEXDETAILS", true)]
    #[case::foreign_fm_terminal_name("type.googleapis.com/example.FMIndexDetails", false)]
    #[case::btree("type.googleapis.com/lance.table.BTreeIndexDetails", false)]
    fn test_results_are_row_addrs(#[case] type_url: &str, #[case] expected: bool) {
        let mut metadata = index_metadata_with(vec![0], vec![]);
        metadata.index_details = Some(Arc::new(prost_types::Any {
            type_url: type_url.to_string(),
            value: Vec::new(),
        }));

        assert_eq!(metadata.results_are_row_addrs(), expected);
    }

    #[rstest]
    #[case::empty_is_valid(vec![7], vec![], None)]
    // mem_wal and frag_reuse both commit no fields at all; a bare
    // `covering_fields.len() < fields.len()` check would reject them.
    #[case::system_index_no_fields(vec![], vec![], None)]
    #[case::valid_single_covered(vec![7, 11], vec![11], None)]
    #[case::valid_suffix(vec![7, 11, 13], vec![11, 13], None)]
    #[case::not_a_suffix(vec![7, 11, 13], vec![11], Some("must come last"))]
    #[case::not_a_subset(vec![7, 11], vec![99], Some("are not among its fields"))]
    #[case::wrong_order(vec![7, 11, 13], vec![13, 11], Some("must come last"))]
    #[case::all_fields_covered(vec![7], vec![7], Some("at least one field must remain indexed"))]
    #[case::covers_the_search_key(vec![7, 11], vec![7, 11], Some("at least one field must remain indexed"))]
    // `fields` repeats the id in the same positions, so `[11, 11]` is a genuine
    // tail of it -- only the duplicate check rejects this.
    #[case::duplicate_covered(vec![7, 11, 11], vec![11, 11], Some("more than once"))]
    // Both over-long and naming an unknown id: the unknown id is the cause, so
    // it must be reported ahead of "at least one field must remain indexed".
    #[case::unknown_id_reported_before_length(vec![7, 11], vec![99, 11], Some("are not among its fields"))]
    fn test_validate_covering_fields(
        #[case] fields: Vec<i32>,
        #[case] covering_fields: Vec<i32>,
        #[case] expected_error: Option<&str>,
    ) {
        let metadata = index_metadata_with(fields, covering_fields);
        let result = metadata.validate_covering_fields();

        match expected_error {
            None => assert!(result.is_ok(), "expected valid, got {:?}", result),
            Some(fragment) => {
                let err = result.expect_err("expected a validation error");
                assert!(
                    matches!(err, Error::InvalidInput { .. }),
                    "expected InvalidInput, got {:?}",
                    err
                );
                let message = err.to_string();
                assert!(
                    message.contains(fragment),
                    "expected message to contain {:?}, got {:?}",
                    fragment,
                    message
                );
            }
        }
    }
}
