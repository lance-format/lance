// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading a segment built in the same commit as the fragments it covers.
//!
//! Such a segment stores row addresses under provisional fragment ids, and
//! [`IndexMetadata::fragment_id_map`] says which committed fragment each one
//! became. The translation runs ahead of any fragment reuse translation,
//! which is written in committed ids.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::scalar::RowIdRemapper;
use lance_select::RowAddrTreeMap;
use lance_table::format::IndexMetadata;
use roaring::RoaringTreemap;

/// Translates a segment's provisional fragment ids, then hands the address to
/// the fragment reuse remapper, if any.
#[derive(Debug)]
pub struct LocalFragmentRemapper {
    fragment_id_map: BTreeMap<u32, u32>,
    then: Option<Arc<dyn RowIdRemapper>>,
}

impl LocalFragmentRemapper {
    /// The remapper for `index`, composed in front of `then`: `then` itself
    /// when the segment stores committed ids.
    pub fn compose(
        index: &IndexMetadata,
        then: Option<Arc<dyn RowIdRemapper>>,
    ) -> Option<Arc<dyn RowIdRemapper>> {
        if index.fragment_id_map.is_empty() {
            return then;
        }
        Some(Arc::new(Self {
            fragment_id_map: index.fragment_id_map.clone(),
            then,
        }))
    }

    fn translate(&self, address: u64) -> u64 {
        let address = RowAddress::from(address);
        match self.fragment_id_map.get(&address.fragment_id()) {
            Some(committed) => {
                u64::from(RowAddress::new_from_parts(*committed, address.row_offset()))
            }
            None => u64::from(address),
        }
    }
}

impl RowIdRemapper for LocalFragmentRemapper {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let translated = self.translate(row_id);
        match &self.then {
            Some(then) => then.remap_row_id(translated),
            None => Some(translated),
        }
    }

    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        RowAddrTreeMap::from_iter(
            row_addrs
                .row_addrs()
                .expect("an index lookup yields an exact set of row addresses")
                .filter_map(|address| self.remap_row_id(u64::from(address))),
        )
    }

    fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        RoaringTreemap::from_iter(row_ids.iter().filter_map(|id| self.remap_row_id(id)))
    }

    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        let addresses = batch
            .column(row_id_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "column {row_id_idx} of an index batch holds row addresses and must be \
                     UInt64, not {}",
                    batch.column(row_id_idx).data_type()
                ))
            })?;
        let translated = UInt64Array::from_unary(addresses, |address| self.translate(address));
        let mut columns = batch.columns().to_vec();
        columns[row_id_idx] = Arc::new(translated);
        let batch = RecordBatch::try_new(batch.schema(), columns)?;
        match &self.then {
            Some(then) => then.remap_row_ids_record_batch(batch, row_id_idx),
            None => Ok(batch),
        }
    }
}

/// The cache namespace of content decoded through a segment's translation, or
/// `None` for a segment that stores committed ids.
///
/// Cached content holds translated addresses, so it depends on the map and
/// not only on the segment uuid: the same segment is read untranslated on the
/// speculative manifest it was built against, and translated once committed,
/// to ids that differ again on each rebase.
pub fn translated_namespace(index: &IndexMetadata) -> Option<String> {
    if index.fragment_id_map.is_empty() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"local-fragment-addresses/1");
    for (provisional, committed) in &index.fragment_id_map {
        hasher.update(&provisional.to_le_bytes());
        hasher.update(&committed.to_le_bytes());
    }
    Some(format!("local-frag/{}", hasher.finalize().to_hex()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use uuid::Uuid;

    fn segment(fragment_id_map: BTreeMap<u32, u32>) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![0],
            covering_fields: vec![],
            name: "idx".into(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
            fragment_id_map,
        }
    }

    fn address(fragment: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(fragment, offset))
    }

    #[test]
    fn test_only_mapped_fragments_are_translated() {
        let provisional = u32::MAX - 1;
        let remapper =
            LocalFragmentRemapper::compose(&segment([(provisional, 7)].into()), None).unwrap();

        assert_eq!(
            remapper.remap_row_id(address(provisional, 3)),
            Some(address(7, 3))
        );
        assert_eq!(remapper.remap_row_id(address(2, 3)), Some(address(2, 3)));

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("ids", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(UInt64Array::from(vec![
                    address(provisional, 0),
                    address(2, 1),
                ])),
            ],
        )
        .unwrap();
        let remapped = remapper.remap_row_ids_record_batch(batch, 1).unwrap();
        assert_eq!(
            remapped
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec(),
            vec![address(7, 0), address(2, 1)]
        );
    }

    #[test]
    fn test_a_segment_with_committed_ids_needs_no_remapper_or_namespace() {
        assert!(LocalFragmentRemapper::compose(&segment(BTreeMap::new()), None).is_none());
        assert!(translated_namespace(&segment(BTreeMap::new())).is_none());
    }

    #[test]
    fn test_the_namespace_follows_the_map() {
        let provisional = u32::MAX - 1;
        assert_ne!(
            translated_namespace(&segment([(provisional, 7)].into())),
            translated_namespace(&segment([(provisional, 8)].into())),
            "a rebase that moves the fragment must not reuse translated content"
        );
    }
}
