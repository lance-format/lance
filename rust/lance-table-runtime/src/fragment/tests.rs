// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{RecordBatch, cast::AsArray, record_batch};
use arrow_select::concat::concat_batches;
use futures::{StreamExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::utils::{address::RowAddress, deletion::DeletionVector};
use lance_core::{ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, Result};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_table::format::{
    DataStorageFormat, Fragment, Manifest, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowIdMeta,
};
use lance_table::io::deletion::write_deletion_file;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use lance_table::utils::stream::ReadBatchFutStream;
use object_store::path::Path;

use super::{FragReadConfig, resolve_actual_row_ids};
use crate::Table;
use crate::rowids::{get_row_id_index, validate_stable_row_ids};
use crate::test_util::{publish_manifest, write_file};

#[test]
fn test_resolve_actual_row_ids() {
    // Doc example on resolve_actual_row_ids: with deletions [2, 3, 5],
    // logical row 4 resolves to physical row 7.
    assert_eq!(resolve_actual_row_ids(&[4], &[2, 3, 5]).unwrap(), vec![7]);

    // Rows below every deletion map to themselves.
    assert_eq!(
        resolve_actual_row_ids(&[0, 1], &[2, 3, 5]).unwrap(),
        vec![0, 1]
    );
    // Indices that land on deleted positions skip past them.
    assert_eq!(resolve_actual_row_ids(&[2], &[2, 3, 5]).unwrap(), vec![4]);
    assert_eq!(resolve_actual_row_ids(&[5], &[2, 3, 5]).unwrap(), vec![8]);
    // Repeated indices are resolved independently.
    assert_eq!(
        resolve_actual_row_ids(&[0, 0, 1, 1], &[1]).unwrap(),
        vec![0, 0, 2, 2]
    );
    // No deletions is the identity mapping.
    assert_eq!(resolve_actual_row_ids(&[3, 7], &[]).unwrap(), vec![3, 7]);
}

#[test]
fn test_resolve_actual_row_ids_matches_a_walk_over_live_rows() {
    // Deletion sets that are empty, dense, sparse, a long prefix and a
    // tail, checked against walking the live rows one by one.
    let num_rows = 2_000u32;
    let deletion_sets: Vec<Vec<u32>> = vec![
        vec![],
        (0..num_rows).filter(|row| row % 3 != 0).collect(),
        (0..num_rows).filter(|row| row % 97 == 5).collect(),
        (0..1_500).collect(),
        (1_000..num_rows - 1).collect(),
    ];
    for deleted in deletion_sets {
        let live: Vec<u32> = (0..num_rows)
            .filter(|row| deleted.binary_search(row).is_err())
            .collect();
        let indices: Vec<u32> = (0..live.len() as u32).rev().collect();
        let expected: Vec<u32> = indices.iter().map(|&i| live[i as usize]).collect();
        assert_eq!(
            resolve_actual_row_ids(&indices, &deleted).unwrap(),
            expected
        );
    }
}

const FILE_VERSION: ConcreteFileVersion = ConcreteFileVersion::V2_1;

async fn combine(stream: ReadBatchFutStream) -> Result<RecordBatch> {
    let batches = stream.buffered(2).try_collect::<Vec<_>>().await?;
    Ok(concat_batches(batches[0].schema_ref(), &batches)?)
}

fn ids_in(batch: &RecordBatch) -> Vec<Option<i32>> {
    batch["id"].as_primitive::<Int32Type>().iter().collect()
}

#[tokio::test]
async fn reads_fragments_with_deletions_overlays_and_spilled_lineage() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let store = ObjectStore::local();
    let base = Path::from_filesystem_path(dir.path()).unwrap();

    let first = record_batch!(("id", Int32, [10, 20, 30, 40, 50])).unwrap();
    let second = record_batch!(("id", Int32, [60, 70])).unwrap();
    let first_file = write_file(&store, &base, "first.lance", &first, FILE_VERSION).await;
    let second_file = write_file(&store, &base, "second.lance", &second, FILE_VERSION).await;

    let lineage = record_batch!(
        (ROW_ID, UInt64, [100, 101, 102, 103, 104]),
        (ROW_CREATED_AT_VERSION, UInt64, [1, 1, 1, 1, 1]),
        (ROW_LAST_UPDATED_AT_VERSION, UInt64, [1, 2, 3, 1, 3])
    )
    .unwrap();
    let mut lineage_file = write_file(&store, &base, "lineage.lance", &lineage, FILE_VERSION).await;
    lineage_file.fields = Arc::from([
        ROW_ID_FIELD_ID,
        ROW_CREATED_AT_VERSION_FIELD_ID,
        ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
    ]);

    let older = record_batch!(("id", Int32, [200, 300, 500])).unwrap();
    let newer = record_batch!(("id", Int32, [Some(333), None])).unwrap();
    let overlays = vec![
        DataOverlayFile {
            data_file: write_file(&store, &base, "older.lance", &older, FILE_VERSION).await,
            coverage: OverlayCoverage::dense([1, 2, 4].into_iter().collect()),
            committed_version: 2,
        },
        DataOverlayFile {
            data_file: write_file(&store, &base, "newer.lance", &newer, FILE_VERSION).await,
            coverage: OverlayCoverage::dense([2, 4].into_iter().collect()),
            committed_version: 3,
        },
    ];

    let deleted = [1, 3].into_iter().collect::<DeletionVector>();
    let mut deletion_file = write_deletion_file(&base, 0, 2, &deleted, &store, 0)
        .await
        .unwrap()
        .unwrap();
    // Read the deletion file when counting live rows, as with older metadata.
    deletion_file.num_deleted_rows = None;
    let first_fragment = Fragment {
        files: vec![first_file, lineage_file],
        physical_rows: Some(first.num_rows()),
        deletion_file: Some(deletion_file),
        row_id_meta: Some(RowIdMeta::Column),
        created_at_version_meta: Some(RowDatasetVersionMeta::Column),
        last_updated_at_version_meta: Some(RowDatasetVersionMeta::Column),
        overlays,
        ..Fragment::new(0)
    };
    let second_fragment = Fragment {
        files: vec![second_file],
        physical_rows: Some(second.num_rows()),
        row_id_meta: Some(RowIdMeta::Inline(
            write_row_ids(&RowIdSequence::from(&[105_u64, 106][..])).into(),
        )),
        ..Fragment::new(1)
    };
    let mut manifest = Manifest::new(
        Schema::try_from(first.schema_ref().as_ref()).unwrap(),
        Arc::new(vec![first_fragment, second_fragment]),
        DataStorageFormat::new(FILE_VERSION),
        Default::default(),
    );
    manifest.version = 3;
    manifest.next_row_id = 107;
    publish_manifest(&store, &base, &mut manifest).await;

    let table = Table::open(uri).await.unwrap();
    let fragments = table.get_fragments();
    assert_eq!(table.manifest().version, 3);
    assert_eq!(
        fragments
            .iter()
            .map(|fragment| fragment.id())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(std::ptr::eq(fragments[0].table(), fragments[1].table()));
    assert!(table.get_fragment(2).is_none());
    assert_eq!(table.count_rows().await.unwrap(), 5);

    let fragment = &fragments[0];
    assert_eq!(fragment.live_rows().await.unwrap(), 3);
    assert_eq!(fragment.physical_rows().await.unwrap(), 5);
    let actual_deletions = fragment.get_deletion_vector().await.unwrap().unwrap();
    assert_eq!(
        actual_deletions.to_sorted_iter().collect::<Vec<_>>(),
        [1, 3]
    );
    fragment.validate().await.unwrap();
    validate_stable_row_ids(&table).await.unwrap();

    let schema = table.schema();
    let reader = fragment
        .open(
            schema,
            FragReadConfig::default()
                .with_row_id(true)
                .with_row_created_at_version(true)
                .with_row_last_updated_at_version(true),
        )
        .await
        .unwrap();
    let all = combine(reader.read_all(2).await.unwrap()).await.unwrap();
    assert_eq!(ids_in(&all), [Some(10), Some(333), None]);
    assert_eq!(
        all[ROW_ID].as_primitive::<UInt64Type>().values().as_ref(),
        &[100, 102, 104]
    );
    assert_eq!(
        all[ROW_CREATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values()
            .as_ref(),
        &[1, 1, 1]
    );
    assert_eq!(
        all[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values()
            .as_ref(),
        &[1, 3, 3]
    );

    let logical = combine(reader.read_range(1..3, 2).await.unwrap())
        .await
        .unwrap();
    assert_eq!(ids_in(&logical), [Some(333), None]);
    let ranges = Arc::from([0..2, 4..5]);
    let physical = combine(reader.read_ranges(ranges, 2).await.unwrap())
        .await
        .unwrap();
    assert_eq!(ids_in(&physical), [Some(10), None]);
    let taken = combine(reader.take(&[1, 2, 4], 2, None).await.unwrap())
        .await
        .unwrap();
    assert_eq!(ids_in(&taken), [Some(333), None]);

    let session = fragment.open_session(schema, true).await.unwrap();
    let selected = session.take(&[2, 0, 2, 1]).await.unwrap();
    assert_eq!(ids_in(&selected), [None, Some(10), None, Some(333)]);
    assert_eq!(
        selected[ROW_ADDR]
            .as_primitive::<UInt64Type>()
            .values()
            .as_ref(),
        &[4, 0, 4, 2]
    );

    let fragment = table.get_fragment(1).unwrap();
    assert_eq!(fragment.count_rows().await.unwrap(), 2);
    assert_eq!(
        ids_in(&fragment.take(&[1, 0], schema).await.unwrap()),
        [Some(70), Some(60)]
    );

    let index = get_row_id_index(&table).await.unwrap().unwrap();
    assert_eq!(index.get(101).unwrap(), None);
    assert_eq!(
        index.get(104).unwrap(),
        Some(RowAddress::new_from_parts(0, 4))
    );
    assert_eq!(
        index.get(105).unwrap(),
        Some(RowAddress::new_from_parts(1, 0))
    );
}

#[tokio::test]
async fn storage_statistics_require_full_file_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let store = ObjectStore::local();
    let base = Path::from_filesystem_path(dir.path()).unwrap();
    let batch = record_batch!(
        ("a", Int32, [1, 2]),
        ("b", Int32, [1, 2]),
        ("c", Int32, [1, 2]),
        ("d", Int32, [1, 2]),
        ("e", Int32, [1, 2])
    )
    .unwrap();
    let file = write_file(&store, &base, "wide.lance", &batch, FILE_VERSION).await;
    let fragment = Fragment {
        files: vec![file],
        physical_rows: Some(2),
        ..Fragment::new(0)
    };
    let mut manifest = Manifest::new(
        Schema::try_from(batch.schema_ref().as_ref()).unwrap(),
        Arc::new(vec![fragment]),
        DataStorageFormat::new(FILE_VERSION),
        Default::default(),
    );
    publish_manifest(&store, &base, &mut manifest).await;

    let table = Table::open(uri).await.unwrap();
    let fragment = table.get_fragment(0).unwrap();
    let file = &fragment.metadata().files[0];
    let projection = table.schema().project(&["a"]).unwrap();
    let config = FragReadConfig::default();
    let reader = fragment
        .open_reader(file, Some(&projection), &config, None)
        .await
        .unwrap()
        .unwrap();
    let err = reader.storage_stats().unwrap_err();
    assert!(
        err.to_string()
            .contains("storage_stats requires full file metadata")
    );
    let reader = fragment
        .open_reader_with_full_metadata(file, Some(&projection), &config)
        .await
        .unwrap()
        .unwrap();
    let stats = reader.storage_stats().unwrap();
    assert_eq!(stats.len(), 5);
}

mod open_cache {
    use std::sync::Arc;

    use lance_core::datatypes::Schema;
    use lance_file::version::ConcreteFileVersion;
    use lance_table::format::DataFile;
    use rstest::rstest;

    use super::super::FileFragment;
    use super::super::open_cache::FileOpenState;
    use super::FragReadConfig;
    use crate::fragment::FragmentOpenCache;
    use crate::test_util::test_commit_table;

    fn uncached_open_state(
        fragment: &FileFragment,
        data_file: &DataFile,
        projection: &Schema,
    ) -> Arc<FileOpenState> {
        fragment
            .file_open_state(data_file, Some(projection), &FragReadConfig::default())
            .unwrap()
    }

    fn same_open_state(a: &FileOpenState, b: &FileOpenState) -> bool {
        a.data_file_schema == b.data_file_schema
            && a.schema_per_file == b.schema_per_file
            && a.field_id_to_column_idx == b.field_id_to_column_idx
            && a.physical == b.physical
    }

    /// `data_file` with its column indices shifted, which makes another layout.
    fn layout_variant(data_file: &DataFile, shift: i32) -> DataFile {
        let mut variant = data_file.clone();
        variant.column_indices = data_file
            .column_indices
            .iter()
            .map(|column_index| column_index + shift)
            .collect();
        variant
    }

    fn with_version(data_file: &DataFile, version: ConcreteFileVersion) -> DataFile {
        let mut data_file = data_file.clone();
        (data_file.file_major_version, data_file.file_minor_version) =
            version.to_data_file_numbers();
        data_file
    }

    /// A table, its first fragment and that fragment's first data file.
    async fn first_file(num_fragments: usize) -> (Arc<crate::Table>, FileFragment, DataFile) {
        let table = Arc::new(test_commit_table(num_fragments * 10, num_fragments).await);
        let fragment = FileFragment::new(table.clone(), table.manifest.fragments[0].clone());
        let data_file = fragment.metadata().files[0].clone();
        (table, fragment, data_file)
    }

    #[rstest]
    #[case::v2_0(ConcreteFileVersion::V2_0)]
    #[case::v2_1(ConcreteFileVersion::V2_1)]
    #[case::v2_2(ConcreteFileVersion::V2_2)]
    #[tokio::test]
    async fn test_open_cache_keeps_one_entry_per_layout(#[case] version: ConcreteFileVersion) {
        let (table, fragment, data_file) = first_file(3).await;
        let data_file = with_version(&data_file, version);
        let projection = Arc::new(table.schema().clone());
        let cache = Arc::new(FragmentOpenCache::new(
            table.manifest.clone(),
            projection.clone(),
        ));
        let config = FragReadConfig::default().with_open_cache(cache.clone());

        let first = fragment
            .file_open_state(&data_file, Some(projection.as_ref()), &config)
            .unwrap();
        let second = fragment
            .file_open_state(&data_file, Some(projection.as_ref()), &config)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(first.physical.is_some());
        assert!(same_open_state(
            &first,
            &uncached_open_state(&fragment, &data_file, &projection)
        ));
        assert_eq!(cache.num_entries(), 1);

        // Every fragment's file shares the entry, by pointer or by value.
        let by_value = layout_variant(&data_file, 0);
        assert!(!Arc::ptr_eq(
            &by_value.column_indices,
            &data_file.column_indices
        ));
        for state in table
            .manifest
            .fragments
            .iter()
            .map(|f| FileFragment::new(table.clone(), f.clone()))
            .map(|f| {
                let file = with_version(&f.metadata().files[0], version);
                f.file_open_state(&file, Some(projection.as_ref()), &config)
                    .unwrap()
            })
            .chain([fragment
                .file_open_state(&by_value, Some(projection.as_ref()), &config)
                .unwrap()])
        {
            assert!(Arc::ptr_eq(&first, &state));
        }
        assert_eq!(cache.num_entries(), 1);

        // Parallel misses on one layout keep the first insert.
        let a = uncached_open_state(&fragment, &layout_variant(&data_file, 1), &projection);
        let b = uncached_open_state(&fragment, &layout_variant(&data_file, 1), &projection);
        assert!(Arc::ptr_eq(&cache.insert(a.clone()), &a));
        assert!(Arc::ptr_eq(&cache.insert(b), &a));
        assert_eq!(cache.num_entries(), 2);
    }

    /// `(0, 3)` and `(2, 0)` both decode as 2.0 but are distinct layouts.
    #[tokio::test]
    async fn test_open_cache_keys_on_file_version_numbers() {
        let (table, fragment, data_file) = first_file(1).await;
        let data_file = with_version(&data_file, ConcreteFileVersion::V2_0);
        let projection = Arc::new(table.schema().clone());
        let cache = Arc::new(FragmentOpenCache::new(
            table.manifest.clone(),
            projection.clone(),
        ));
        let config = FragReadConfig::default().with_open_cache(cache.clone());

        let mut other_version = data_file.clone();
        (
            other_version.file_major_version,
            other_version.file_minor_version,
        ) = match (data_file.file_major_version, data_file.file_minor_version) {
            (0, 3) => (2, 0),
            _ => (0, 3),
        };
        let first = fragment
            .file_open_state(&data_file, Some(projection.as_ref()), &config)
            .unwrap();
        let second = fragment
            .file_open_state(&other_version, Some(projection.as_ref()), &config)
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(cache.num_entries(), 2);
    }

    #[tokio::test]
    async fn test_open_cache_beyond_capacity_computes_uncached() {
        let (table, fragment, data_file) = first_file(1).await;
        let projection = Arc::new(table.schema().clone());
        let cache = Arc::new(FragmentOpenCache::new(
            table.manifest.clone(),
            projection.clone(),
        ));
        let config = FragReadConfig::default().with_open_cache(cache.clone());

        let num_layouts = FragmentOpenCache::MAX_ENTRIES as i32 + 4;
        for shift in 0..num_layouts {
            let variant = layout_variant(&data_file, shift);
            let state = fragment
                .file_open_state(&variant, Some(projection.as_ref()), &config)
                .unwrap();
            assert!(same_open_state(
                &state,
                &uncached_open_state(&fragment, &variant, &projection)
            ));
        }
        assert_eq!(cache.num_entries(), FragmentOpenCache::MAX_ENTRIES);

        let overflow = layout_variant(&data_file, num_layouts - 1);
        let first = fragment
            .file_open_state(&overflow, Some(projection.as_ref()), &config)
            .unwrap();
        let second = fragment
            .file_open_state(&overflow, Some(projection.as_ref()), &config)
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(same_open_state(&first, &second));
        assert_eq!(cache.num_entries(), FragmentOpenCache::MAX_ENTRIES);
    }

    #[tokio::test]
    async fn test_open_cache_ignores_other_projections_and_manifests() {
        let (table, fragment, data_file) = first_file(1).await;
        let projection = Arc::new(table.schema().clone());
        let cache = Arc::new(FragmentOpenCache::new(
            table.manifest.clone(),
            projection.clone(),
        ));
        let config = FragReadConfig::default().with_open_cache(cache.clone());

        // An equal projection at another address is not the cache's projection.
        let equal_projection = Arc::new(table.schema().clone());
        let narrow_projection = Arc::new(table.schema().project(&["a"]).unwrap());
        for other in [&equal_projection, &narrow_projection] {
            let state = fragment
                .file_open_state(&data_file, Some(other.as_ref()), &config)
                .unwrap();
            assert!(same_open_state(
                &state,
                &uncached_open_state(&fragment, &data_file, other)
            ));
        }
        assert_eq!(cache.num_entries(), 0);

        // An equal table with another manifest.
        let (_, other_fragment, _) = first_file(1).await;
        other_fragment
            .file_open_state(&data_file, Some(projection.as_ref()), &config)
            .unwrap();
        assert_eq!(cache.num_entries(), 0);
    }
}
