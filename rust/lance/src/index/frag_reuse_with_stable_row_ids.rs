// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Readers apply the fragment reuse index to every index they load, which would
//! rewrite stable row ids as if they were addresses. So on a stable-row-id table
//! it may only coexist with indices proven to store row addresses.

use lance_core::{Error, Result};
use lance_index::mem_wal::MEM_WAL_INDEX_NAME;
use lance_index::registry::display_type_from_url;
use lance_table::feature_flags::{
    FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS, frag_reuse_with_stable_row_ids_enabled,
};
use lance_table::format::{IndexMetadata, Manifest, pb as table_pb};
use lance_table::system_index::frag_reuse::is_frag_reuse_index_entry;
use prost::Name;
use prost_types::Any;

use super::scalar::IndexDetails;
use super::{index_is_usable, unsupported_index_version};

fn details_type_name(details: &Any) -> &str {
    details
        .type_url
        .rsplit_once('/')
        .map_or(details.type_url.as_str(), |(_, name)| name)
}

fn is_mem_wal_index_entry(index: &IndexMetadata) -> bool {
    index.name == MEM_WAL_INDEX_NAME
        && index.index_details.as_ref().is_some_and(|details| {
            details_type_name(details)
                .eq_ignore_ascii_case(&table_pb::MemWalIndexDetails::full_name())
        })
}

/// An empty fragment reuse index counts.
pub fn has_frag_reuse_with_stable_row_ids(manifest: &Manifest, indices: &[IndexMetadata]) -> bool {
    manifest.uses_stable_row_ids() && indices.iter().any(is_frag_reuse_index_entry)
}

/// `supports_batch_row_id_remapping` is only a proxy for applying the legacy
/// fragment reuse index on load, so every admitted type needs a query test on
/// that path.
fn address_index_incompatibility(index: &IndexMetadata) -> Option<String> {
    let Some(details) = index.index_details.as_ref() else {
        return Some("a legacy index without type details, which stores stable row IDs".into());
    };
    let type_name = details_type_name(details);
    let display_type = display_type_from_url(&details.type_url);
    let details = IndexDetails(details.clone());
    if !details.has_reader() {
        return Some(format!(
            "an index of type `{type_name}`, which is not known to this build of Lance"
        ));
    }
    if type_name.eq_ignore_ascii_case(&lance_index::pb::JsonIndexDetails::full_name()) {
        return Some(
            "a JSON index, which does not record the identifiers its target index stores".into(),
        );
    }
    if let Some(max_supported_version) = unsupported_index_version(index) {
        return Some(format!(
            "a {display_type} index at version {}, which is not known to this build of Lance \
             (it supports up to version {max_supported_version})",
            index.index_version
        ));
    }
    if !index.results_are_row_addrs() {
        return Some(format!(
            "a {display_type} index, which stores stable row IDs"
        ));
    }
    let applies_frag_reuse = details
        .get_plugin()
        .is_ok_and(|plugin| plugin.supports_batch_row_id_remapping());
    if !applies_frag_reuse {
        return Some(format!(
            "a {display_type} index, which does not apply the fragment reuse index"
        ));
    }
    None
}

fn ordinary_index_incompatibility(index: &IndexMetadata) -> Option<String> {
    if is_frag_reuse_index_entry(index) || is_mem_wal_index_entry(index) {
        return None;
    }
    address_index_incompatibility(index)
}

fn incompatible_indices(indices: &[IndexMetadata]) -> Vec<(&IndexMetadata, String)> {
    indices
        .iter()
        .filter_map(|index| ordinary_index_incompatibility(index).map(|reason| (index, reason)))
        .collect()
}

pub fn apply_frag_reuse_with_stable_row_ids_flag(
    manifest: &mut Manifest,
    indices: &[IndexMetadata],
) {
    if has_frag_reuse_with_stable_row_ids(manifest, indices) {
        manifest.reader_feature_flags |= FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
    } else {
        manifest.reader_feature_flags &= !FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
        manifest.writer_feature_flags &= !FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
    }
}

pub fn validate_frag_reuse_with_stable_row_ids(
    manifest: &Manifest,
    indices: &[IndexMetadata],
) -> Result<()> {
    validate_frag_reuse_with_stable_row_ids_when(
        manifest,
        indices,
        frag_reuse_with_stable_row_ids_enabled(),
    )
}

/// Split out so the release-build refusal is testable from a debug build.
fn validate_frag_reuse_with_stable_row_ids_when(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    is_supported: bool,
) -> Result<()> {
    if !has_frag_reuse_with_stable_row_ids(manifest, indices) {
        return Ok(());
    }
    let version = manifest.version;
    if !is_supported {
        return Err(Error::not_supported(format!(
            "Cannot commit version {version}: the dataset uses stable row IDs and carries a \
             fragment reuse index, which this build of Lance does not support"
        )));
    }
    let frag_reuse_entries: Vec<_> = indices
        .iter()
        .filter(|index| is_frag_reuse_index_entry(index))
        .collect();
    if frag_reuse_entries.len() != 1 {
        return Err(Error::invalid_input(format!(
            "Cannot commit version {version}: the dataset carries {} fragment reuse index \
             entries, but exactly one is allowed",
            frag_reuse_entries.len()
        )));
    }
    let frag_reuse_version = frag_reuse_entries[0].index_version;
    if frag_reuse_version != 0 {
        return Err(Error::invalid_input(format!(
            "Cannot commit version {version}: the dataset uses stable row IDs and carries a \
             fragment reuse index with index_version {frag_reuse_version}, but datasets with \
             stable row IDs support only index_version 0"
        )));
    }
    let incompatible = incompatible_indices(indices);
    if incompatible.is_empty() {
        return Ok(());
    }
    let listed = incompatible
        .iter()
        .map(|(index, reason)| format!("`{}` ({}): {reason}", index.name, index.uuid))
        .collect::<Vec<_>>()
        .join("; ");
    Err(Error::invalid_input(format!(
        "Cannot commit version {version}: the dataset uses stable row IDs and carries a \
         fragment reuse index, which is applied to every index as row-address moves, so each \
         index must be one this build of Lance can confirm stores row addresses and applies \
         the fragment reuse index. These indices cannot be confirmed: {listed}"
    )))
}

pub fn ensure_index_allowed_with_frag_reuse(
    manifest: &Manifest,
    existing_indices: &[IndexMetadata],
    index: &IndexMetadata,
) -> Result<()> {
    if !has_frag_reuse_with_stable_row_ids(manifest, existing_indices) {
        return Ok(());
    }
    match ordinary_index_incompatibility(index) {
        None => Ok(()),
        Some(reason) => Err(Error::invalid_input(format!(
            "Cannot create index `{}`: the dataset uses stable row IDs and carries a fragment \
             reuse index, which would be applied to this index as row-address moves, and it \
             is {reason}. On such a dataset only indices that store row addresses and apply \
             the fragment reuse index can be created",
            index.name
        ))),
    }
}

pub fn ensure_index_kind_allowed_with_frag_reuse(
    manifest: &Manifest,
    existing_indices: &[IndexMetadata],
    index_kind: &str,
    column: &str,
) -> Result<()> {
    if !has_frag_reuse_with_stable_row_ids(manifest, existing_indices) {
        return Ok(());
    }
    Err(Error::invalid_input(format!(
        "Cannot create a {index_kind} index on column `{column}`: the dataset uses stable row \
         IDs and carries a fragment reuse index, which would be applied to the index as \
         row-address moves, and {index_kind} indices store stable row IDs. On such a dataset \
         only indices that store row addresses and apply the fragment reuse index can be \
         created"
    )))
}

/// Only meaningful where [`has_frag_reuse_with_stable_row_ids`] holds.
pub fn is_hidden_by_frag_reuse(index: &IndexMetadata) -> bool {
    ordinary_index_incompatibility(index).is_some()
}

pub fn warn_about_indices_hidden_by_frag_reuse(manifest: &Manifest, indices: &[IndexMetadata]) {
    if !has_frag_reuse_with_stable_row_ids(manifest, indices) {
        return;
    }
    for (index, reason) in incompatible_indices(indices) {
        // Unreadable indices are already reported by `warn_about_unsupported_indices`.
        if index_is_usable(index) {
            log::warn!(
                "Index {} is not used: the dataset uses stable row IDs and carries a fragment \
                 reuse index, and the index is {reason}",
                index.name,
            );
        }
    }
}

/// Backstop for callers that load an index from the full list, not the usable
/// view that already hides incompatible indices.
pub fn ensure_frag_reuse_applies(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    index: &IndexMetadata,
) -> Result<()> {
    if !has_frag_reuse_with_stable_row_ids(manifest, indices) {
        return Ok(());
    }
    match ordinary_index_incompatibility(index) {
        None => Ok(()),
        Some(reason) => Err(Error::invalid_input(format!(
            "Refusing to apply the fragment reuse index to index `{}` ({}): the dataset uses \
             stable row IDs and the index is {reason}",
            index.name, index.uuid
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::types::{Float32Type, Int32Type};
    use arrow_array::{Array, Float32Array, Int32Array, RecordBatchIterator};
    use lance_core::ROW_ADDR;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{Dimension, array, gen_batch};
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::{BuiltinIndexType, InvertedIndexParams, ScalarIndexParams};
    use lance_index::{IndexType, pb, pbold};
    use lance_linalg::distance::MetricType;
    use lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
    use lance_table::format::DataStorageFormat;
    use lance_table::io::commit::write_manifest_file_to_path;
    use lance_table::system_index::frag_reuse::{
        FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
    };
    use prost_types::Any;
    use roaring::{RoaringBitmap, RoaringTreemap};
    use rstest::rstest;

    use super::*;
    use crate::Dataset;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::transaction::{Operation, TransactionBuilder};
    use crate::dataset::{CommitBuilder, WriteParams};
    use crate::index::frag_reuse::build_frag_reuse_index_metadata;
    use crate::index::vector::VectorIndexParams;
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt, load_all_indices};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    fn index_with(name: &str, details: Option<Any>, index_version: i32) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::from_iter([0])),
            index_details: details.map(Arc::new),
            index_version,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    fn details<M: prost::Name>(message: &M) -> Option<Any> {
        Some(Any::from_msg(message).unwrap())
    }

    fn foreign(type_name: &str) -> Option<Any> {
        Some(Any {
            type_url: format!("type.googleapis.com/{type_name}"),
            value: vec![],
        })
    }

    fn json_over(target: Option<Any>) -> Option<Any> {
        details(&pb::JsonIndexDetails {
            path: "x".to_string(),
            target_details: target,
        })
    }

    fn undecodable_json() -> Option<Any> {
        details(&pb::JsonIndexDetails::default()).map(|details| Any {
            type_url: details.type_url,
            value: vec![0xff; 4],
        })
    }

    fn newer_than_supported<M: prost::Name>(message: &M) -> i32 {
        let details = IndexDetails(Arc::new(Any::from_msg(message).unwrap()));
        details.index_version().unwrap() as i32 + 1
    }

    #[rstest]
    #[case::zone_map(details(&pbold::ZoneMapIndexDetails::default()), 0, true)]
    #[case::bloom_filter(details(&pb::BloomFilterIndexDetails::default()), 0, true)]
    #[case::btree(details(&pbold::BTreeIndexDetails::default()), 0, false)]
    #[case::bitmap(details(&pbold::BitmapIndexDetails::default()), 0, false)]
    #[case::label_list(details(&pbold::LabelListIndexDetails::default()), 0, false)]
    #[case::ngram(details(&pbold::NGramIndexDetails::default()), 0, false)]
    #[case::inverted(details(&pbold::InvertedIndexDetails::default()), 0, false)]
    #[case::vector(details(&pb::VectorIndexDetails::default()), 0, false)]
    #[case::fm(details(&pb::FmIndexDetails::default()), 0, false)]
    #[case::min_hash(details(&pb::MinHashLshIndexDetails::default()), 0, false)]
    #[case::json(details(&pb::JsonIndexDetails::default()), 0, false)]
    #[case::json_over_zone_map(
        json_over(details(&pbold::ZoneMapIndexDetails::default())),
        0,
        false
    )]
    #[case::json_over_btree(json_over(details(&pbold::BTreeIndexDetails::default())), 0, false)]
    #[case::undecodable_json(undecodable_json(), 0, false)]
    #[case::legacy_without_details(None, 0, false)]
    #[case::unknown_type(foreign("example.ForeignIndexDetails"), 0, false)]
    #[case::zone_map_look_alike(foreign("example.ZoneMapIndexDetails"), 0, false)]
    #[case::zone_map_from_a_newer_build(
        details(&pbold::ZoneMapIndexDetails::default()),
        newer_than_supported(&pbold::ZoneMapIndexDetails::default()),
        false
    )]
    fn test_only_proven_address_indices_are_compatible(
        #[case] index_details: Option<Any>,
        #[case] index_version: i32,
        #[case] is_compatible: bool,
    ) {
        let index = index_with("idx", index_details, index_version);
        assert_eq!(
            address_index_incompatibility(&index).is_none(),
            is_compatible
        );
        assert_eq!(is_hidden_by_frag_reuse(&index), !is_compatible);
    }

    #[test]
    fn test_system_indices_are_recognized_by_name_and_type() {
        let frag_reuse = index_with(
            FRAG_REUSE_INDEX_NAME,
            details(&table_pb::FragmentReuseIndexDetails::default()),
            0,
        );
        let mem_wal = index_with(
            MEM_WAL_INDEX_NAME,
            details(&table_pb::MemWalIndexDetails::default()),
            0,
        );
        assert!(is_frag_reuse_index_entry(&frag_reuse));
        assert!(!is_hidden_by_frag_reuse(&frag_reuse));
        assert!(!is_hidden_by_frag_reuse(&mem_wal));

        let btree = details(&pbold::BTreeIndexDetails::default());
        for borrowed in [
            index_with(FRAG_REUSE_INDEX_NAME, btree.clone(), 0),
            index_with(MEM_WAL_INDEX_NAME, btree, 0),
            index_with(FRAG_REUSE_INDEX_NAME, None, 0),
            index_with(MEM_WAL_INDEX_NAME, None, 0),
        ] {
            assert!(!is_frag_reuse_index_entry(&borrowed), "{borrowed:?}");
            assert!(is_hidden_by_frag_reuse(&borrowed), "{borrowed:?}");
        }
    }

    fn stable_row_id_manifest() -> Manifest {
        let arrow_schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "i",
            arrow_schema::DataType::Int32,
            false,
        )]);
        let mut manifest = Manifest::new(
            lance_core::datatypes::Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest
    }

    fn frag_reuse_entry(index_version: i32) -> IndexMetadata {
        index_with(
            FRAG_REUSE_INDEX_NAME,
            details(&table_pb::FragmentReuseIndexDetails::default()),
            index_version,
        )
    }

    #[test]
    fn test_publication_policy_and_invariant() {
        let manifest = stable_row_id_manifest();
        let zone_map = index_with(
            "zone_map",
            details(&pbold::ZoneMapIndexDetails::default()),
            0,
        );
        let valid = vec![frag_reuse_entry(0), zone_map.clone()];
        validate_frag_reuse_with_stable_row_ids_when(&manifest, &valid, true).unwrap();

        // As a release build.
        let error =
            validate_frag_reuse_with_stable_row_ids_when(&manifest, &valid, false).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("does not support"), "{error}");

        let mut without_stable_row_ids = manifest.clone();
        without_stable_row_ids.reader_feature_flags = 0;
        without_stable_row_ids.writer_feature_flags = 0;
        let bitmap = index_with("bitmap", details(&pbold::BitmapIndexDetails::default()), 0);
        validate_frag_reuse_with_stable_row_ids_when(
            &without_stable_row_ids,
            &[frag_reuse_entry(0), bitmap.clone()],
            false,
        )
        .unwrap();
        validate_frag_reuse_with_stable_row_ids_when(
            &manifest,
            std::slice::from_ref(&bitmap),
            false,
        )
        .unwrap();

        for (indices, expected) in [
            (
                vec![frag_reuse_entry(0), frag_reuse_entry(0)],
                "exactly one is allowed",
            ),
            (vec![frag_reuse_entry(1)], "support only index_version 0"),
            (vec![frag_reuse_entry(0), zone_map, bitmap], "`bitmap`"),
        ] {
            let error = validate_frag_reuse_with_stable_row_ids_when(&manifest, &indices, true)
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(expected), "{message}");
            for advice in ["retry", "drain", "delete"] {
                assert!(!message.to_lowercase().contains(advice), "{message}");
            }
        }
    }

    async fn stable_row_id_dataset(uri: &str) -> Dataset {
        gen_batch()
            .col("i", array::step::<Int32Type>())
            .col(
                "category",
                array::cycle_utf8_literals(&["a", "b", "c", "d"]),
            )
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
            .into_dataset_with_params(
                uri,
                FragmentCount::from(3),
                FragmentRowCount::from(20),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 20,
                    ..Default::default()
                }),
            )
            .await
            .unwrap()
    }

    async fn empty_frag_reuse_index(dataset: &Dataset, bitmap: RoaringBitmap) -> IndexMetadata {
        build_frag_reuse_index_metadata(
            dataset,
            None,
            FragReuseIndexDetails { versions: vec![] },
            bitmap,
        )
        .await
        .unwrap()
    }

    async fn commit_new_indices(
        dataset: &mut Dataset,
        new_indices: Vec<IndexMetadata>,
    ) -> Result<()> {
        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices,
                removed_indices: vec![],
            },
        )
        .build();
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
    }

    /// Publishes below `write_manifest_file`'s checks, as another writer could.
    async fn plant(dataset: &mut Dataset, indices: Vec<IndexMetadata>) {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.version += 1;
        manifest.transaction_file = None;
        manifest.transaction_section = None;
        apply_frag_reuse_with_stable_row_ids_flag(&mut manifest, &indices);
        let location = dataset
            .commit_handler
            .commit(
                &mut manifest,
                Some(indices),
                &dataset.base,
                &dataset.object_store,
                write_manifest_file_to_path,
                dataset.manifest_location.naming_scheme,
                None,
            )
            .await
            .unwrap();
        *dataset = dataset.checkout_version(location.version).await.unwrap();
    }

    fn has_flag(dataset: &Dataset) -> bool {
        let flag = FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
        let in_reader = dataset.manifest.reader_feature_flags & flag != 0;
        let in_writer = dataset.manifest.writer_feature_flags & flag != 0;
        assert_eq!(in_reader, in_writer, "the flag must be set in both words");
        in_reader
    }

    async fn create_index(
        dataset: &mut Dataset,
        column: &str,
        index_type: IndexType,
        params: &dyn lance_index::IndexParams,
    ) -> Result<()> {
        dataset
            .create_index_builder(&[column], index_type, params)
            .name(format!("{column}_idx"))
            .await
            .map(|_| ())
    }

    fn assert_rejected(result: Result<()>, expected: &str) {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains(expected), "{message}");
        for advice in ["retry", "drain", "delete"] {
            assert!(!message.to_lowercase().contains(advice), "{message}");
        }
    }

    #[tokio::test]
    async fn test_flag_is_derived_for_every_version() {
        let dir = TempStrDir::default();
        let uri = format!("{}/source", dir.as_str());
        let mut dataset = stable_row_id_dataset(&uri).await;
        create_index(
            &mut dataset,
            "i",
            IndexType::ZoneMap,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        )
        .await
        .unwrap();
        assert!(!has_flag(&dataset));

        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        commit_new_indices(&mut dataset, vec![frag_reuse])
            .await
            .unwrap();
        assert!(has_flag(&dataset));

        dataset.append(five_rows(), None).await.unwrap();
        assert!(has_flag(&dataset));
        dataset.delete("i < 3").await.unwrap();
        assert!(has_flag(&dataset));

        let version = dataset.version().version;
        let shallow = dataset
            .shallow_clone(&format!("{}/shallow", dir.as_str()), version, None)
            .await
            .unwrap();
        assert!(shallow.manifest.uses_stable_row_ids());
        assert!(has_flag(&shallow));
        let deep = dataset
            .deep_clone(&format!("{}/deep", dir.as_str()), version, None)
            .await
            .unwrap();
        assert!(deep.manifest.uses_stable_row_ids());
        assert!(has_flag(&deep));

        dataset.drop_index(FRAG_REUSE_INDEX_NAME).await.unwrap();
        assert!(!has_flag(&dataset));
    }

    #[tokio::test]
    async fn test_flag_is_not_set_without_stable_row_ids() {
        let mut dataset = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("category", array::cycle_utf8_literals(&["a", "b"]))
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(10))
            .await
            .unwrap();
        create_index(
            &mut dataset,
            "category",
            IndexType::Bitmap,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
        )
        .await
        .unwrap();
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        commit_new_indices(&mut dataset, vec![frag_reuse])
            .await
            .unwrap();
        assert!(dataset.frag_reuse_index().await.unwrap().is_some());
        assert!(!has_flag(&dataset));
    }

    #[derive(Clone, Copy, Debug)]
    enum Publication {
        Create,
        DetachedCreate,
        Rewrite,
    }

    #[rstest]
    #[case::create_index(Publication::Create)]
    #[case::detached_create_index(Publication::DetachedCreate)]
    #[case::rewrite_carrying_frag_reuse_index(Publication::Rewrite)]
    #[tokio::test]
    async fn test_publication_rejects_an_incompatible_index(#[case] publication: Publication) {
        let mut dataset = stable_row_id_dataset("memory://").await;
        dataset
            .create_index_builder(
                &["category"],
                IndexType::Bitmap,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            )
            .name("category_idx".to_string())
            .train(false)
            .await
            .unwrap();
        let version = dataset.version().version;
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        let result = match publication {
            Publication::Create => commit_new_indices(&mut dataset, vec![frag_reuse]).await,
            Publication::DetachedCreate => {
                let transaction = TransactionBuilder::new(
                    version,
                    Operation::CreateIndex {
                        new_indices: vec![frag_reuse],
                        removed_indices: vec![],
                    },
                )
                .build();
                CommitBuilder::new(Arc::new(dataset.clone()))
                    .with_detached(true)
                    .execute(transaction)
                    .await
                    .map(|_| ())
            }
            Publication::Rewrite => {
                let transaction = TransactionBuilder::new(
                    version,
                    Operation::Rewrite {
                        groups: vec![],
                        rewritten_indices: vec![],
                        frag_reuse_index: Some(frag_reuse),
                    },
                )
                .build();
                dataset
                    .apply_commit(transaction, &Default::default(), &Default::default())
                    .await
            }
        };
        assert_rejected(result, "`category_idx`");
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.version().version, version);
    }

    /// Typed requests ignore the params' index type; generic ones resolve it.
    #[rstest]
    #[case::typed_zone_map(IndexType::ZoneMap, ScalarIndexParams::default(), None)]
    #[case::generic_zone_map(
        IndexType::Scalar,
        ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        None
    )]
    #[case::generic_default_btree(IndexType::Scalar, ScalarIndexParams::default(), Some("BTree"))]
    #[case::typed_btree_with_zone_map_params(
        IndexType::BTree,
        ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        Some("BTree")
    )]
    #[case::typed_bitmap(IndexType::Bitmap, ScalarIndexParams::default(), Some("Bitmap"))]
    #[tokio::test]
    async fn test_create_scalar_index_under_frag_reuse(
        #[case] index_type: IndexType,
        #[case] params: ScalarIndexParams,
        #[case] rejected_type: Option<&str>,
    ) {
        let mut dataset = stable_row_id_dataset("memory://").await;
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        commit_new_indices(&mut dataset, vec![frag_reuse])
            .await
            .unwrap();
        let version = dataset.version().version;

        let result = create_index(&mut dataset, "i", index_type, &params).await;
        match rejected_type {
            None => result.unwrap(),
            Some(type_name) => {
                let message = result.as_ref().unwrap_err().to_string();
                assert!(message.contains("Cannot create index `i_idx`"), "{message}");
                assert_rejected(result, type_name);
                dataset.checkout_latest().await.unwrap();
                assert_eq!(dataset.version().version, version);
            }
        }
    }

    fn generic_full_text_params() -> ScalarIndexParams {
        ScalarIndexParams {
            index_type: "inverted".to_string(),
            ..Default::default()
        }
    }

    /// Only the refusal before training names the index kind.
    #[rstest]
    #[case::vector(
        "vec",
        IndexType::IvfFlat,
        Box::new(VectorIndexParams::ivf_flat(1, MetricType::L2)),
        "Cannot create a vector index on column `vec`"
    )]
    #[case::full_text(
        "category",
        IndexType::Inverted,
        Box::new(InvertedIndexParams::default()),
        "Cannot create a full-text index on column `category`"
    )]
    #[case::generic_full_text(
        "category",
        IndexType::Scalar,
        Box::new(generic_full_text_params()),
        "Cannot create a full-text index on column `category`"
    )]
    #[tokio::test]
    async fn test_row_id_index_kinds_are_refused_before_training(
        #[case] column: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn lance_index::IndexParams>,
        #[case] expected: &str,
    ) {
        let mut dataset = stable_row_id_dataset("memory://").await;
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        commit_new_indices(&mut dataset, vec![frag_reuse])
            .await
            .unwrap();

        let result = create_index(&mut dataset, column, index_type, params.as_ref()).await;
        assert_rejected(result, expected);
    }

    #[derive(Clone, Copy, Debug)]
    enum Republish {
        Append,
        Restore,
        ShallowClone,
        DeepClone,
    }

    fn five_rows() -> impl arrow_array::RecordBatchReader + Send + 'static {
        let batch = gen_batch()
            .col("i", array::step_custom::<Int32Type>(1000, 1))
            .col("category", array::cycle_utf8_literals(&["a"]))
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
            .into_batch_rows(lance_datagen::RowCount::from(5))
            .unwrap();
        let schema = batch.schema();
        RecordBatchIterator::new(vec![Ok(batch)], schema)
    }

    #[rstest]
    #[case::append(Republish::Append)]
    #[case::restore(Republish::Restore)]
    #[case::shallow_clone(Republish::ShallowClone)]
    #[case::deep_clone(Republish::DeepClone)]
    #[tokio::test]
    async fn test_planted_incompatible_state_cannot_be_republished(#[case] operation: Republish) {
        let dir = TempStrDir::default();
        let uri = format!("{}/source", dir.as_str());
        let mut dataset = stable_row_id_dataset(&uri).await;
        dataset
            .create_index_builder(
                &["category"],
                IndexType::Bitmap,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            )
            .name("category_idx".to_string())
            .train(false)
            .await
            .unwrap();
        let mut indices = load_all_indices(&dataset).await.unwrap().as_ref().clone();
        indices.push(empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await);
        plant(&mut dataset, indices).await;
        let planted_version = dataset.version().version;
        let clone_uri = format!("{}/clone", dir.as_str());

        let (result, latest_version) = match operation {
            Republish::Append => (dataset.append(five_rows(), None).await, planted_version),
            Republish::Restore => {
                dataset.drop_index("category_idx").await.unwrap();
                let mut planted = dataset.checkout_version(planted_version).await.unwrap();
                (planted.restore().await, planted_version + 1)
            }
            Republish::ShallowClone => (
                dataset
                    .shallow_clone(&clone_uri, planted_version, None)
                    .await
                    .map(|_| ()),
                planted_version,
            ),
            Republish::DeepClone => (
                dataset
                    .deep_clone(&clone_uri, planted_version, None)
                    .await
                    .map(|_| ()),
                planted_version,
            ),
        };
        assert_rejected(result, "`category_idx`");
        let latest = Dataset::open(&uri).await.unwrap();
        assert_eq!(latest.version().version, latest_version);
        assert!(Dataset::open(&clone_uri).await.is_err());
    }

    #[rstest]
    #[case::append(false)]
    #[case::delete(true)]
    #[tokio::test]
    async fn test_index_from_a_newer_build_blocks_writes(#[case] is_delete: bool) {
        let mut dataset = stable_row_id_dataset("memory://").await;
        create_index(
            &mut dataset,
            "i",
            IndexType::ZoneMap,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        )
        .await
        .unwrap();
        let mut indices = load_all_indices(&dataset).await.unwrap().as_ref().clone();
        for index in indices.iter_mut() {
            index.index_version += 1;
        }
        indices.push(empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await);
        plant(&mut dataset, indices).await;
        let version = dataset.version().version;

        let result = if is_delete {
            dataset.delete("i < 3").await.map(|_| ())
        } else {
            dataset.append(five_rows(), None).await
        };
        let message = result.as_ref().unwrap_err().to_string();
        assert!(
            message.contains("not known to this build of Lance"),
            "{message}"
        );
        assert_rejected(result, "`i_idx`");
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.version().version, version);
    }

    async fn query_i(dataset: &Dataset, filter: &str, use_scalar_index: bool) -> Vec<i32> {
        let batch = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .use_scalar_index(use_scalar_index)
            .try_into_batch()
            .await
            .unwrap();
        let mut values = batch["i"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        values.sort_unstable();
        values
    }

    /// The bitmap's stable row ids collide with compacted fragment 0's old
    /// addresses, so applying the fragment reuse index to it would lose rows.
    #[tokio::test]
    async fn test_incompatible_indices_are_hidden_from_readers() {
        let mut dataset = stable_row_id_dataset("memory://").await;
        create_index(
            &mut dataset,
            "category",
            IndexType::Bitmap,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
        )
        .await
        .unwrap();
        create_index(
            &mut dataset,
            "vec",
            IndexType::IvfFlat,
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
        )
        .await
        .unwrap();
        dataset.delete("i < 5").await.unwrap();
        let expected = query_i(&dataset, "category = 'a'", false).await;

        // The fragment reuse index this compaction would have recorded.
        let old_fragments = dataset.manifest.fragments.as_ref().clone();
        let batch = dataset
            .scan()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let old_addresses: RoaringTreemap = batch[ROW_ADDR]
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap()
            .values()
            .iter()
            .copied()
            .collect();
        let read_version = dataset.version().version;
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1_000,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let new_fragments = dataset.manifest.fragments.as_ref().clone();
        let mut changed_row_addrs = Vec::new();
        old_addresses
            .serialize_into(&mut changed_row_addrs)
            .unwrap();
        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: read_version,
                groups: vec![FragReuseGroup {
                    changed_row_addrs,
                    old_frags: old_fragments.iter().map(FragDigest::from).collect(),
                    new_frags: new_fragments.iter().map(FragDigest::from).collect(),
                }],
            }],
        };
        let new_bitmap = new_fragments.iter().map(|f| f.id as u32).collect();
        let frag_reuse = build_frag_reuse_index_metadata(&dataset, None, details, new_bitmap)
            .await
            .unwrap();
        let mut indices = load_all_indices(&dataset).await.unwrap().as_ref().clone();
        indices.push(frag_reuse);
        plant(&mut dataset, indices).await;

        assert_eq!(query_i(&dataset, "category = 'a'", true).await, expected);
        let query = Float32Array::from(vec![0.5; 4]);
        let nearest = |use_index: bool| {
            let dataset = dataset.clone();
            let query = query.clone();
            async move {
                let batch = dataset
                    .scan()
                    .nearest("vec", &query, 5)
                    .unwrap()
                    .use_index(use_index)
                    .try_into_batch()
                    .await
                    .unwrap();
                batch["i"]
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            }
        };
        assert_eq!(nearest(true).await, nearest(false).await);

        let visible = dataset.load_indices().await.unwrap();
        assert!(
            visible
                .iter()
                .all(|index| index.name != "category_idx" && index.name != "vec_idx"),
            "{visible:?}"
        );
        assert!(
            dataset
                .load_index_by_name("category_idx")
                .await
                .unwrap()
                .is_none()
        );
        let all_indices = load_all_indices(&dataset).await.unwrap();
        let hidden = |name: &str| {
            all_indices
                .iter()
                .find(|index| index.name == name)
                .unwrap()
                .clone()
        };
        let bitmap = hidden("category_idx");
        let error = dataset
            .open_scalar_index("category", &bitmap.uuid, &NoOpMetricsCollector)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("does not exist"), "{error}");

        // Metadata from the full list reaches the remapper's backstop.
        let error = crate::index::scalar::open_scalar_index(
            &dataset,
            "category",
            &bitmap,
            &NoOpMetricsCollector,
        )
        .await
        .err()
        .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("Refusing to apply the fragment reuse index to index `category_idx`"),
            "{error}"
        );
        let error = crate::index::frag_reuse::open_row_id_remapping(
            &dataset,
            &hidden("vec_idx"),
            &NoOpMetricsCollector,
        )
        .await
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains("Refusing to apply the fragment reuse index to index `vec_idx`"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_rewrite_skips_the_frag_reuse_index_bitmap() {
        let mut dataset = stable_row_id_dataset("memory://").await;
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::from_iter([1])).await;
        commit_new_indices(&mut dataset, vec![frag_reuse])
            .await
            .unwrap();

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1_000,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert!(dataset.frag_reuse_index().await.unwrap().is_some());
        assert!(has_flag(&dataset));
    }

    #[tokio::test]
    async fn test_empty_frag_reuse_index_is_still_present() {
        let mut dataset = stable_row_id_dataset("memory://").await;
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        commit_new_indices(&mut dataset, vec![frag_reuse])
            .await
            .unwrap();
        assert!(has_flag(&dataset));
        let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
        assert!(frag_reuse.is_empty());
        let addresses = arrow_array::UInt64Array::from(vec![0_u64, 7, 1 << 32]);
        assert_eq!(
            frag_reuse.remap_row_ids_array(Arc::new(addresses.clone())),
            addresses
        );

        assert_rejected(
            create_index(
                &mut dataset,
                "category",
                IndexType::Bitmap,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            )
            .await,
            "Cannot create index `category_idx`",
        );
        let bitmap = index_with(
            "category_idx",
            details(&pbold::BitmapIndexDetails::default()),
            0,
        );
        assert_rejected(
            commit_new_indices(&mut dataset, vec![bitmap]).await,
            "`category_idx`",
        );
        create_index(
            &mut dataset,
            "i",
            IndexType::ZoneMap,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        )
        .await
        .unwrap();
    }

    #[rstest]
    #[case::frag_reuse_index_first(true)]
    #[case::bitmap_first(false)]
    #[tokio::test]
    async fn test_racing_index_creation_is_rejected_after_rebase(#[case] frag_reuse_first: bool) {
        let dir = TempStrDir::default();
        let mut dataset = stable_row_id_dataset(dir.as_str()).await;
        let mut other = Dataset::open(dir.as_str()).await.unwrap();
        let frag_reuse = empty_frag_reuse_index(&dataset, RoaringBitmap::new()).await;
        let bitmap = other
            .create_index_builder(
                &["category"],
                IndexType::Bitmap,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            )
            .name("category_idx".to_string())
            .train(false)
            .execute_uncommitted()
            .await
            .unwrap();

        let result = if frag_reuse_first {
            commit_new_indices(&mut dataset, vec![frag_reuse])
                .await
                .unwrap();
            commit_new_indices(&mut other, vec![bitmap]).await
        } else {
            commit_new_indices(&mut other, vec![bitmap]).await.unwrap();
            commit_new_indices(&mut dataset, vec![frag_reuse]).await
        };
        assert_rejected(result, "`category_idx`");
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        let winner_index = if frag_reuse_first {
            FRAG_REUSE_INDEX_NAME
        } else {
            "category_idx"
        };
        let latest_names: Vec<_> = load_all_indices(&latest)
            .await
            .unwrap()
            .iter()
            .map(|index| index.name.clone())
            .collect();
        assert_eq!(latest_names, vec![winner_index.to_string()]);
    }
}
