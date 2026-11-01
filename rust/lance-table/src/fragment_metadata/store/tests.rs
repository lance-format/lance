// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use super::upload::split_ranges;
use super::*;

use crate::format::{
    DataFile, DeletionFile, DeletionFileType, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowIdMeta,
};
use crate::fragment_metadata::action;
use crate::fragment_metadata::support::{
    data_file_path, make_backfill_data_file, make_fragment, make_fragment_with_files,
    make_replacement_data_file,
};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use lance_core::utils::tempfile::TempObjDir;
use lance_file::version::ConcreteFileVersion;
use lance_io::scheduler::SchedulerConfig;
use lance_io::utils::tracking_store::IOTracker;
use object_store::{
    CopyOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStoreExt, PutMultipartOptions, PutResult, Result as OSResult,
};
use rstest::rstest;

fn column_lineage_fragment(id: u64) -> Fragment {
    let mut fragment = make_fragment(id);
    fragment.files.push(DataFile::new(
        format!("lineage-{id}.lance"),
        vec![
            ROW_ID_FIELD_ID,
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        ],
        vec![0, 1, 2],
        ConcreteFileVersion::V2_2,
        None,
        None,
    ));
    fragment.row_id_meta = Some(RowIdMeta::Column);
    fragment.created_at_version_meta = Some(RowDatasetVersionMeta::Column);
    fragment.last_updated_at_version_meta = Some(RowDatasetVersionMeta::Column);
    fragment
}

#[tokio::test]
async fn wrappers_share_one_binding_per_resolved_store() {
    let original = Arc::new(ObjectStore::memory());
    let primary_base = Path::from("primary");
    let mut store = NodeStore::new(
        original.clone(),
        primary_base.clone(),
        ScanScheduler::new(original.clone(), SchedulerConfig::default_for_testing()),
    );
    let foreign = Arc::new(ObjectStore::memory());
    let foreign_base = Path::from("inherited");
    let original_ordering = foreign.list_is_lexically_ordered;
    let mut different_settings = foreign.as_ref().clone();
    different_settings.list_is_lexically_ordered = !original_ordering;
    store.set_foreign_bases(HashMap::from([
        (1, (original, primary_base)),
        (2, (foreign.clone(), foreign_base.clone())),
        (3, (foreign.clone(), foreign_base.clone())),
        (4, (Arc::new(different_settings), foreign_base.clone())),
        (5, (Arc::new(ObjectStore::memory()), foreign_base.clone())),
    ]));
    let child = |base_id| pb::FragmentTreeChild {
        path: "_bt/node/one.node".into(),
        base_id: Some(base_id),
        ..Default::default()
    };
    let locations: Vec<_> = (1..=5)
        .map(|base_id| store.child_location(&child(base_id)).unwrap())
        .collect();
    assert_eq!(locations[1], locations[2]);
    assert_eq!(locations[1], locations[3]);
    assert_ne!(locations[1], locations[4]);
    for _ in 0..2 {
        let tracker = Arc::new(IOTracker::default());
        let mut primary = store.object_store.as_ref().clone();
        primary.apply_wrapper(tracker.as_ref());
        let primary = Arc::new(primary);
        let wrapper: Arc<dyn WrappingObjectStore> = tracker.clone();
        store.wrap_stores(primary.clone(), std::slice::from_ref(&wrapper));
        assert!(Arc::ptr_eq(&store.foreign_bases[&1].store, &primary));
        assert!(Arc::ptr_eq(
            &store.foreign_bases[&2].store,
            &store.foreign_bases[&3].store,
        ));
        assert_eq!(store.foreign_bases[&2].base, foreign_base);
        assert_eq!(
            store.foreign_bases[&2].store.list_is_lexically_ordered,
            original_ordering
        );
        assert_eq!(
            store.foreign_bases[&4].store.list_is_lexically_ordered,
            !original_ordering
        );
        assert_eq!(
            store
                .location_under(&foreign, &foreign_base, &child(2).path)
                .unwrap(),
            locations[1]
        );
        for (base_id, expected) in (1..=5).zip(&locations) {
            let child = child(base_id);
            assert_eq!(&store.child_location(&child).unwrap(), expected);
            let (owner, path) = store.resolve_child(&child).unwrap();
            owner.put(&path, b"node").await.unwrap();
            owner.inner.get(&path).await.unwrap().bytes().await.unwrap();
        }
        assert_eq!(tracker.stats().read_iops, 5);
    }
    let local = pb::FragmentTreeChild {
        base_id: None,
        ..child(1)
    };
    let previous = store.child_location(&local).unwrap();
    store.replace_store(Arc::new(ObjectStore::memory()));
    assert_ne!(store.child_location(&local).unwrap(), previous);
    assert_eq!(store.child_location(&child(2)).unwrap(), locations[1]);
}

#[rstest]
#[case::historical_numbers(0, 3)]
#[case::unknown_numbers(17, 42)]
#[tokio::test]
async fn leaf_preserves_data_file_version_numbers(#[case] major: u32, #[case] minor: u32) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let mut fragment = make_fragment(0);
    fragment.files[0].file_major_version = major;
    fragment.files[0].file_minor_version = minor;
    let written = store.write_leaf(&[fragment.clone()], 0).await.unwrap();
    assert_eq!(
        store.read_leaf(&written.child_ref).await.unwrap(),
        vec![fragment]
    );
}

#[tokio::test]
async fn interior_read_rejects_an_understated_object_size() {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let children = (0..2)
        .map(|id| {
            node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
        })
        .collect();
    let mut written = store.write_interior(children, Vec::new()).await.unwrap();
    written.child_ref.object_size = 1;
    let error = store.read_interior(&written.child_ref).await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("parent declares"), "{error}");
}

/// Answers every read with a response whose body fails, the only failure
/// `download_retry_count` governs.
#[derive(Debug)]
struct FailingBodyStore;

impl std::fmt::Display for FailingBodyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingBodyStore")
    }
}

#[async_trait]
impl OSObjectStore for FailingBodyStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> OSResult<PutResult> {
        unimplemented!("put_opts is not used by this test")
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        unimplemented!("put_multipart_opts is not used by this test")
    }

    async fn get_opts(&self, location: &Path, _options: GetOptions) -> OSResult<GetResult> {
        let body = stream::once(async {
            Err(object_store::Error::Generic {
                store: "FailingBodyStore",
                source: "connection reset".into(),
            })
        });
        Ok(GetResult {
            payload: GetResultPayload::Stream(body.boxed()),
            meta: ObjectMeta {
                location: location.clone(),
                last_modified: Default::default(),
                size: 1024,
                e_tag: None,
                version: None,
            },
            range: 0..1024,
            attributes: Default::default(),
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        unimplemented!("delete_stream is not used by this test")
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        unimplemented!("list is not used by this test")
    }

    async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> OSResult<ListResult> {
        unimplemented!("list_with_delimiter is not used by this test")
    }

    async fn copy_opts(&self, _from: &Path, _to: &Path, _opts: CopyOptions) -> OSResult<()> {
        unimplemented!("copy_opts is not used by this test")
    }
}

#[rstest]
#[case::no_retries(0, 1)]
#[case::default(3, 4)]
#[case::raised(10, 11)]
#[tokio::test]
async fn leaf_read_retries_a_partial_download_as_configured(
    #[case] download_retry_count: usize,
    #[case] expected_requests: u64,
) {
    let object_store = Arc::new(ObjectStore::new(
        Arc::new(FailingBodyStore),
        url::Url::parse("memory:///").unwrap(),
        None,
        None,
        false,
        false,
        1,
        download_retry_count,
        None,
    ));
    let scheduler =
        ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
    let store = NodeStore::new(object_store.clone(), Path::default(), scheduler);
    let child = pb::FragmentTreeChild {
        path: "_bt/leaf/broken.lance".into(),
        object_size: 1024,
        ..Default::default()
    };

    store.leaf_object(&child).unwrap().fetch().await;

    assert_eq!(
        object_store.io_stats_snapshot().read_iops,
        expected_requests
    );
}

/// Reading the bytes is what fails, so an operator gets the object to fetch.
#[rstest]
#[case::garbage_of_the_same_size(|bytes: &[u8]| vec![0xFF; bytes.len()])]
#[case::truncated(|bytes: &[u8]| bytes[..bytes.len() - 1].to_vec())]
#[tokio::test]
async fn damaged_nodes_and_roots_report_corrupt_file_at_their_object(
    #[case] damage: fn(&[u8]) -> Vec<u8>,
) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let children: Vec<_> = (0..2)
        .map(|id| {
            node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
        })
        .collect();
    let interior = store
        .write_interior(children.clone(), Vec::new())
        .await
        .unwrap()
        .child_ref;
    let (uuid, _) = store
        .write_root_base(&pb::FragmentTreeRoot {
            children,
            next_action_sequence: 1,
            ..Default::default()
        })
        .await
        .unwrap();
    let node_object = store.resolve_child(&interior).unwrap().1;
    let root_object = store.resolve_path(&root_path(&uuid).unwrap()).unwrap();
    for object in [&node_object, &root_object] {
        let bytes = get_whole(&store.object_store, object).await.unwrap();
        store
            .object_store
            .put(object, &damage(&bytes))
            .await
            .unwrap();
    }

    let node_error = store.read_interior(&interior).await.unwrap_err();
    let root_error = store.read_root_base(&uuid).await.unwrap_err();

    for (error, object) in [(node_error, node_object), (root_error, root_object)] {
        assert!(
            matches!(&error, Error::CorruptFile { path, .. } if *path == object),
            "{error}"
        );
    }
}

/// List validators and the count helpers see no object, so the reader
/// names the interior they rejected.
#[rstest]
#[case::child_list(
    |stored: &mut pb::FragmentTreeNode| stored.children[0].object_size = 0,
    "child reference"
)]
#[case::buffer(
    |stored: &mut pb::FragmentTreeNode| stored.buffer[0].action_sequence = 0,
    "sequence number 0"
)]
#[case::counts(
    |stored: &mut pb::FragmentTreeNode| stored.buffer[0].fragment_count_delta = -999,
    "aggregate underflow"
)]
#[tokio::test]
async fn interior_that_fails_its_checks_reports_its_object(
    #[case] tamper: fn(&mut pb::FragmentTreeNode),
    #[case] expected: &str,
) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let mut stored = pb::FragmentTreeNode {
        children: (0..2)
            .map(|id| {
                node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
            })
            .collect(),
        buffer: vec![pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::remove_fragment(0)),
            fragment_count_delta: -1,
            total_rows_delta: -1,
            visible_rows_delta: -1,
        }],
    };
    let mut child = store
        .write_interior(stored.children.clone(), stored.buffer.clone())
        .await
        .unwrap()
        .child_ref;
    tamper(&mut stored);
    let bytes = stored.encode_to_vec();
    child.object_size = bytes.len() as u64;
    let object = store.resolve_child(&child).unwrap().1;
    store.object_store.put(&object, &bytes).await.unwrap();

    let error = store.read_interior(&child).await.unwrap_err();

    assert!(
        matches!(&error, Error::CorruptFile { path, .. } if *path == object),
        "{error}"
    );
    assert!(error.to_string().contains(expected), "{error}");
}

#[rstest]
#[case::at_limit(false)]
#[case::over_limit(true)]
#[tokio::test]
async fn interior_write_respects_hard_capacity(#[case] over_limit: bool) {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    let children: Vec<_> = (0..2)
        .map(|id| {
            node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
        })
        .collect();
    let size = pb::FragmentTreeNode {
        children: children.clone(),
        buffer: Vec::new(),
    }
    .encoded_len() as u64;
    store.hard_capacity_bytes = size - u64::from(over_limit);
    let result = store.write_interior(children, Vec::new()).await;
    if over_limit {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("hard_capacity_bytes"), "{error}");
        assert!(
            store
                .object_store
                .inner
                .list(Some(&base))
                .try_next()
                .await
                .unwrap()
                .is_none()
        );
    } else {
        assert_eq!(result.unwrap().child_ref.object_size, size);
    }
}

#[rstest]
#[case::reader_cache(false)]
#[case::validation_scratch(true)]
#[tokio::test]
async fn cached_leaves_preserve_resolved_owner(#[case] retain: bool) {
    let mut bases = HashMap::new();
    let mut children = Vec::new();
    for id in 0..2 {
        let object_store = Arc::new(ObjectStore::memory());
        let source = NodeStore::new(
            object_store.clone(),
            Path::default(),
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing()),
        );
        let mut child = source
            .write_leaf(&[make_fragment_with_files(id, id as u32 + 1)], 0)
            .await
            .unwrap()
            .child_ref;
        let bytes = object_store
            .inner
            .get(&Path::from(child.path.clone()))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        child.path = "_bt/leaf/shared-name.lance".into();
        object_store
            .put(&Path::from(child.path.clone()), &bytes)
            .await
            .unwrap();
        child.base_id = Some(id as u32);
        bases.insert(id as u32, (object_store, Path::default()));
        children.push(child);
    }
    // Aliases resolve to the same bytes but stamp different inherited base IDs.
    bases.insert(2, bases[&0].clone());
    let mut alias = children[0].clone();
    alias.base_id = Some(2);
    children.push(alias);
    let object_store = Arc::new(ObjectStore::memory());
    let mut dest = NodeStore::new(
        object_store.clone(),
        Path::default(),
        ScanScheduler::new(object_store, SchedulerConfig::default_for_testing()),
    );
    dest.set_foreign_bases(bases);
    dest.set_cache(LanceCache::with_capacity(1024 * 1024));
    if retain {
        dest.retain_validation_reads().unwrap();
    }
    // First sighting, admission and a cache hit select the stamped record.
    for _ in 0..3 {
        for (child, id) in children.iter().zip([0, 1, 0]) {
            let mut expected = make_fragment_with_files(id, id as u32 + 1);
            for file in &mut expected.files {
                file.base_id = child.base_id;
            }
            let selected = dest
                .read_leaf_shared(child, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap()
                .select(&RoaringBitmap::from_iter([id as u32]));
            assert_eq!(selected, vec![expected.clone()]);
            assert_eq!(dest.read_leaf(child).await.unwrap(), vec![expected]);
        }
    }
}

#[rstest]
#[case::empty_leaf(&[], &[1], &[])]
#[case::nothing_requested(&[2, 4, 9], &[], &[])]
#[case::below_the_leaf(&[2, 4, 9], &[0, 1], &[])]
#[case::above_the_leaf(&[2, 4, 9], &[10, u32::MAX], &[])]
#[case::first_record(&[2, 4, 9], &[2], &[2])]
#[case::last_record(&[2, 4, 9], &[9], &[9])]
#[case::a_gap_inside_the_leaf(&[2, 4, 9], &[3, 5], &[])]
#[case::edges_and_ids_around_them(&[2, 4, 9], &[1, 2, 5, 9, 10], &[2, 9])]
#[case::whole_leaf(&[2, 4, 9], &[2, 4, 9], &[2, 4, 9])]
#[case::highest_id(&[7, u32::MAX as u64], &[u32::MAX], &[u32::MAX as u64])]
fn select_keeps_the_requested_records_of_cached_and_owned_leaves(
    #[case] leaf: &[u64],
    #[case] requested: &[u32],
    #[case] expected: &[u64],
) {
    let fragments: Vec<_> = leaf.iter().copied().map(make_fragment).collect();
    let requested = RoaringBitmap::from_iter(requested.iter().copied());
    let expected: Vec<_> = expected.iter().copied().map(make_fragment).collect();
    let cached = LeafRead::Cached(Arc::new(CheckedLeaf::new(fragments.clone())));
    assert_eq!(cached.select(&requested), expected);
    assert_eq!(LeafRead::Owned(fragments).select(&requested), expected);
}

fn interior_buffering(action_sequences: &[u64]) -> CheckedInterior {
    CheckedInterior::new(InteriorNode {
        children: Vec::new(),
        buffer: action_sequences
            .iter()
            .map(|action_sequence| pb::FragmentTreeMutation {
                action_sequence: *action_sequence,
                action: Some(action::remove_fragment(0)),
                ..Default::default()
            })
            .collect(),
    })
}

/// The sets show that a repeat exists. Only the buffer shows which one comes
/// first, and that is the one the error has to name.
#[rstest]
#[case::repeats_the_root(&[9, 5, 2, 7], 2)]
#[case::repeats_an_earlier_interior(&[9, 8, 3], 3)]
#[case::names_the_first_repeat_in_buffer_order(&[9, 4, 2], 4)]
fn cached_interior_admission_reports_what_buffer_admission_reports(
    #[case] repeating: &[u64],
    #[case] named: u64,
) {
    let root = interior_buffering(&[2]);
    let mut admitted = node::DecodedSequences::new(&root.node.buffer);
    admitted
        .admit_interior("earlier", &interior_buffering(&[4, 3]))
        .unwrap();
    let mut by_buffer = admitted.clone();
    let interior = interior_buffering(repeating);

    let error = admitted
        .admit_interior("later", &interior)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(&format!("Interior later buffers action sequence {named},")),
        "{error}"
    );
    let expected = by_buffer
        .admit("later", &interior.node.buffer)
        .unwrap_err()
        .to_string();
    assert_eq!(error, expected);
}

#[test]
fn cached_interior_weighs_its_sequence_set() {
    let interior = interior_buffering(&[9, 5, 2]);
    assert!(interior.sequences.serialized_size() > 0);
    assert_eq!(
        interior.deep_size_of_children(&mut Context::new()),
        interior
            .node
            .buffer
            .deep_size_of_children(&mut Context::new())
            + interior.sequences.serialized_size()
    );
}

/// A leaf cached for one store is never served for another store that
/// names an object at the same path through an equal reference.
#[tokio::test]
async fn cached_leaves_keep_stores_with_one_path_apart() {
    let cache = LanceCache::with_capacity(1024 * 1024);
    let node_store = |object_store: Arc<ObjectStore>| {
        let mut store = NodeStore::new(
            object_store.clone(),
            Path::from("table"),
            ScanScheduler::new(object_store, SchedulerConfig::default_for_testing()),
        );
        store.set_cache(cache.clone());
        store
    };
    let holder = node_store(Arc::new(ObjectStore::memory()));
    let child = holder
        .write_leaf(&[make_fragment(0)], 0)
        .await
        .unwrap()
        .child_ref;
    // A first sighting, its admission, then a cache hit.
    for _ in 0..3 {
        assert_eq!(holder.read_leaf(&child).await.unwrap().len(), 1);
    }
    let other = node_store(Arc::new(ObjectStore::memory()));
    let error = other.read_leaf(&child).await.unwrap_err();
    assert!(error.is_not_found(), "{error}");
}

#[tokio::test]
async fn root_read_preserves_unknown_protobuf_fields_without_writer_limits() {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let mut bytes = pb::FragmentTreeRoot::default().encode_to_vec();
    bytes.extend_from_slice(&[0xa2, 0x06, 0x80, 0x01]);
    bytes.extend_from_slice(&[0; 128]);
    let uuid = Uuid::new_v4();
    let path = root_path(uuid.as_bytes()).unwrap();
    store
        .object_store
        .put(&store.resolve_path(&path).unwrap(), &bytes)
        .await
        .unwrap();
    store.read_root_base(uuid.as_bytes()).await.unwrap();
}

#[rstest]
#[case::appended_byte(true)]
#[case::truncated_object(false)]
#[tokio::test]
async fn leaf_read_rejects_a_misstated_object_size(#[case] appended: bool) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let written = store.write_leaf(&[make_fragment(0)], 0).await.unwrap();
    let path = store.resolve_path(&written.child_ref.path).unwrap();
    let mut bytes = get_whole(&store.object_store, &path)
        .await
        .unwrap()
        .to_vec();
    if appended {
        bytes.push(0);
    } else {
        bytes.pop();
    }
    store.object_store.put(&path, &bytes).await.unwrap();
    let error = store.read_leaf(&written.child_ref).await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("parent declares"), "{error}");
}

fn test_store(base: Path) -> NodeStore {
    let object_store = Arc::new(ObjectStore::local());
    let scheduler =
        ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
    NodeStore::new(object_store, base, scheduler)
}

#[rstest]
#[case::encode_on_write(false)]
#[case::reuse_encoding(true)]
#[tokio::test]
async fn write_leaves_splits_at_the_limit_and_caps_singletons_at_hard_capacity(
    #[case] has_encoding: bool,
) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let config = node::FragmentTreeConfig::new(4096, 8)
        .with_max_leaf_bytes(8192)
        .with_hard_capacity_bytes(4 * 1024 * 1024);
    let fragments: Vec<_> = (0..256).map(make_fragment).collect();
    let encoded = if has_encoding {
        Some(store.encode_leaf(&fragments).await.unwrap())
    } else {
        None
    };
    let written = store
        .write_leaves(&fragments, encoded, 17, &config)
        .await
        .unwrap();
    assert!(written.len() > 1);
    let mut read = Vec::new();
    for leaf in &written {
        assert!(leaf.io_bytes <= config.max_leaf_bytes || leaf.child_ref.num_keys == 1);
        assert_eq!(leaf.child_ref.object_size, leaf.io_bytes);
        assert_eq!(leaf.child_ref.materialized_through_action_sequence, 17);
        read.extend(store.read_leaf(&leaf.child_ref).await.unwrap());
    }
    assert_eq!(read, fragments);

    let wide = make_fragment_with_files(257, 4096);
    let encoded = if has_encoding {
        Some(
            store
                .encode_leaf(std::slice::from_ref(&wide))
                .await
                .unwrap(),
        )
    } else {
        None
    };
    let written = store
        .write_leaves(std::slice::from_ref(&wide), encoded, 18, &config)
        .await
        .unwrap();
    assert_eq!(written.len(), 1);
    assert!(written[0].io_bytes > config.max_leaf_bytes);
    assert_eq!(
        store.read_leaf(&written[0].child_ref).await.unwrap(),
        vec![wide.clone()]
    );
    let before = store
        .object_store
        .read_dir_all(&store.base.clone().join("_bt").join("leaf"), None)
        .map_ok(|meta| meta.location)
        .try_collect::<std::collections::BTreeSet<_>>()
        .await
        .unwrap();
    let restricted = config.with_hard_capacity_bytes(16 * 1024);
    let encoded = if has_encoding {
        Some(
            store
                .encode_leaf(std::slice::from_ref(&wide))
                .await
                .unwrap(),
        )
    } else {
        None
    };
    let error = store
        .write_leaves(&[wide], encoded, 19, &restricted)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("fragment 257"));
    assert!(error.to_string().contains("hard_capacity_bytes=16384"));
    assert_eq!(
        store
            .object_store
            .read_dir_all(&store.base.clone().join("_bt").join("leaf"), None)
            .map_ok(|meta| meta.location)
            .try_collect::<std::collections::BTreeSet<_>>()
            .await
            .unwrap(),
        before
    );
}

/// `count` fragments with 16 files, where `wide` adjacent fragments in the
/// middle carry 512 files.
fn fragments_with_wide_cluster(count: u64, wide: u64) -> Vec<Fragment> {
    let wide_ids = (count - wide) / 2..(count + wide) / 2;
    (0..count)
        .map(|id| make_fragment_with_files(id, if wide_ids.contains(&id) { 512 } else { 16 }))
        .collect()
}

fn uniform_fragments(count: u64) -> Vec<Fragment> {
    (0..count).map(make_fragment).collect()
}

fn assert_ordered_cover(pieces: &[Range<usize>], range: &Range<usize>) {
    assert_eq!(pieces.first().map(|piece| piece.start), Some(range.start));
    assert_eq!(pieces.last().map(|piece| piece.end), Some(range.end));
    for piece in pieces {
        assert!(!piece.is_empty(), "{pieces:?}");
    }
    for pair in pieces.windows(2) {
        assert_eq!(pair[0].end, pair[1].start, "{pieces:?}");
    }
}

const TARGET: u64 = 1 << 20;

#[rstest]
#[case::uniform_4096(uniform_fragments(4096), 0..4096, 12 * TARGET, 16)]
#[case::uniform_1003(uniform_fragments(1100), 7..1010, 3 * TARGET + 1, 4)]
#[case::clustered_wide(fragments_with_wide_cluster(4096, 41), 0..4096, 12 * TARGET, 16)]
#[case::pieces_exceed_records(uniform_fragments(3), 0..3, 100 * TARGET, 3)]
#[case::dominant_first_fragment(
    std::iter::once(make_fragment_with_files(0, 4096))
        .chain((1..5).map(make_fragment))
        .collect(),
    0..5,
    3 * TARGET,
    2
)]
fn split_ranges_cover_the_range_in_order(
    #[case] fragments: Vec<Fragment>,
    #[case] range: Range<usize>,
    #[case] encoded_bytes: u64,
    #[case] expected_pieces: usize,
) {
    let pieces = split_ranges(&fragments, range.clone(), encoded_bytes, TARGET);
    assert_ordered_cover(&pieces, &range);
    assert_eq!(pieces.len(), expected_pieces, "{pieces:?}");
}

/// The pieces repeated halving at `floor(len / 2)` reaches after `depth` levels.
fn bisect(range: Range<usize>, depth: u32) -> Vec<Range<usize>> {
    if depth == 0 {
        return vec![range];
    }
    let middle = range.start + range.len() / 2;
    let mut pieces = bisect(range.start..middle, depth - 1);
    pieces.extend(bisect(middle..range.end, depth - 1));
    pieces
}

#[rstest]
#[case::pow2_4096_into_16(4096, 0..4096, 12 * TARGET, 4)]
#[case::odd_1003_into_4(1100, 7..1010, 3 * TARGET, 2)]
fn split_ranges_match_bisection_on_uniform_records(
    #[case] count: u64,
    #[case] range: Range<usize>,
    #[case] encoded_bytes: u64,
    #[case] depth: u32,
) {
    let fragments = uniform_fragments(count);
    assert_eq!(
        split_ranges(&fragments, range.clone(), encoded_bytes, TARGET),
        bisect(range, depth)
    );
}

#[rstest]
#[case::uniform(fragments_with_wide_cluster(4096, 0))]
#[case::clustered_wide(fragments_with_wide_cluster(4096, 41))]
#[tokio::test]
async fn oversized_range_encodes_each_record_at_most_twice_plus_resplits(
    #[case] fragments: Vec<Fragment>,
) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    // More than eight leaf limits, so halving would encode every record at
    // least five times, the probe and four levels. Sixteen-file records
    // keep the per-leaf file overhead small next to a limit, as in real leaves.
    let encoded_bytes = store.encode_leaf(&fragments).await.unwrap().len() as u64;
    let config = node::FragmentTreeConfig::new(4096, 8)
        .with_max_leaf_bytes(encoded_bytes / 12)
        .with_hard_capacity_bytes(64 * 1024 * 1024);
    store
        .encoded_records
        .store(0, std::sync::atomic::Ordering::Relaxed);
    let written = store
        .write_leaves(&fragments, None, 0, &config)
        .await
        .unwrap();
    let encoded_records = store
        .encoded_records
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        encoded_records * 2 <= fragments.len() * 5,
        "{encoded_records} record encodes for {} fragments in {} leaves",
        fragments.len(),
        written.len()
    );
    let mut read = Vec::new();
    for leaf in &written {
        assert!(leaf.io_bytes <= config.max_leaf_bytes || leaf.child_ref.num_keys == 1);
        read.extend(store.read_leaf(&leaf.child_ref).await.unwrap());
    }
    assert_eq!(read, fragments);
}

#[tokio::test]
async fn skewed_range_leaves_stay_above_quarter_limit() {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let fragments = fragments_with_wide_cluster(4096, 41);
    let config = node::FragmentTreeConfig::new(4096, 8)
        .with_max_leaf_bytes(256 * 1024)
        .with_hard_capacity_bytes(64 * 1024 * 1024);
    let written = store
        .write_leaves(&fragments, None, 0, &config)
        .await
        .unwrap();
    let sizes: Vec<_> = written.iter().map(|leaf| leaf.io_bytes).collect();
    assert!(written.len() > 8, "{sizes:?}");
    for leaf in &written {
        let oversized_singleton =
            leaf.child_ref.num_keys == 1 && leaf.io_bytes > config.max_leaf_bytes;
        assert!(
            oversized_singleton || leaf.io_bytes >= config.max_leaf_bytes / 4,
            "{sizes:?}"
        );
    }
}

#[tokio::test]
async fn leaf_round_trips_fragment_metadata_and_empty_fragment() {
    let mut fragment = make_fragment(7);
    fragment.deletion_file = Some(DeletionFile {
        read_version: 3,
        id: 11,
        file_type: DeletionFileType::Bitmap,
        num_deleted_rows: Some(1),
        base_id: Some(2),
        inline: None,
    });
    fragment
        .overlays
        .push(crate::format::overlay::DataOverlayFile {
            data_file: fragment.files[0].clone(),
            coverage: crate::format::overlay::OverlayCoverage::dense(
                roaring::RoaringBitmap::from_iter([0]),
            ),
            committed_version: 3,
        });
    fragment.row_id_meta = Some(RowIdMeta::Inline(vec![1, 2, 3, 4].into()));
    fragment.created_at_version_meta = Some(RowDatasetVersionMeta::Inline(Arc::from([5, 6, 7])));
    fragment.last_updated_at_version_meta =
        Some(RowDatasetVersionMeta::Inline(Arc::from([8, 9, 10])));

    let mut empty_fragment = Fragment::new(8);
    empty_fragment.physical_rows = Some(12);
    empty_fragment.row_id_meta = Some(RowIdMeta::Inline(vec![12, 13].into()));

    let expected = vec![fragment, empty_fragment, column_lineage_fragment(9)];
    let tempdir = TempObjDir::default();
    let store = test_store(tempdir.clone().join("fragment_metadata"));
    let written = store.write_leaf(&expected, 0).await.unwrap();
    let actual = store.read_leaf(&written.child_ref).await.unwrap();

    assert_eq!(actual, expected);
}
#[tokio::test]
async fn retained_validation_reads_keep_known_zero_counts_and_check_the_parent_reference() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    let mut fragment = make_fragment(0);
    fragment.physical_rows = Some(0);
    fragment.deletion_file = Some(DeletionFile {
        read_version: 1,
        id: 1,
        file_type: DeletionFileType::Bitmap,
        num_deleted_rows: Some(0),
        base_id: None,
        inline: None,
    });
    let written = store
        .write_leaf(std::slice::from_ref(&fragment), 0)
        .await
        .unwrap();
    store.retain_validation_reads().unwrap();
    // First read populates scratch; the second decodes the scratch record.
    for _ in 0..2 {
        assert_eq!(
            store.read_leaf(&written.child_ref).await.unwrap(),
            vec![fragment.clone()]
        );
    }
    let mut corrupt = written.child_ref.clone();
    corrupt.num_keys += 1;
    let error = store.read_leaf(&corrupt).await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("parent reference"), "{error}");
}

#[tokio::test]
async fn resolve_path_accepts_only_tree_relative_paths() {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let written = store.write_leaf(&[make_fragment(0)], 0).await.unwrap();
    assert!(written.child_ref.path.starts_with("_bt/leaf/"));
    assert!(!written.child_ref.path.contains("%2F"));
    let path = store.resolve_path(&written.child_ref.path).unwrap();
    assert!(store.object_store.inner.head(&path).await.is_ok());
    for path in [
        "../_bt/leaf/escape",
        "/_bt/leaf/absolute",
        "data/not-metadata",
    ] {
        assert!(store.resolve_path(path).is_err(), "{path}");
    }
}

/// A leaf decoded twice is kept, and then answers only for the parent
/// reference it was checked against, whole or by record. Reads that
/// succeed after the object is gone prove they came from the cache.
#[tokio::test]
async fn cached_leaf_serves_only_the_reference_it_checked() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let fragments: Vec<_> = (0..4).map(make_fragment).collect();
    let once = store.write_leaf(&fragments, 0).await.unwrap();
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    assert_eq!(store.read_leaf(&once.child_ref).await.unwrap(), fragments);
    for _ in 0..2 {
        let read = store.read_leaf(&written.child_ref).await.unwrap();
        assert_eq!(read, fragments);
    }

    let once_path = store.resolve_path(&once.child_ref.path).unwrap();
    store.object_store.inner.delete(&once_path).await.unwrap();
    assert!(store.read_leaf(&once.child_ref).await.is_err());

    let path = store.resolve_path(&written.child_ref.path).unwrap();
    store.object_store.inner.delete(&path).await.unwrap();
    assert_eq!(
        store.read_leaf(&written.child_ref).await.unwrap(),
        fragments
    );
    let selected = store
        .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
        .await
        .unwrap()
        .select(&RoaringBitmap::from_iter([1, 3, 9]));
    assert_eq!(selected, vec![fragments[1].clone(), fragments[3].clone()]);
    let point = store
        .read_leaf_fragment(&written.child_ref, 2, node::ROOT_EXCLUSIVE_END)
        .await;
    assert_eq!(point.unwrap().as_ref(), Some(&fragments[2]));
    let missing = store
        .read_leaf_fragment(&written.child_ref, 9, node::ROOT_EXCLUSIVE_END)
        .await;
    assert!(missing.unwrap().is_none());

    let mut changed = written.child_ref.clone();
    changed.visible_rows += 1;
    assert!(store.read_leaf(&changed).await.is_err());
    assert!(
        store
            .read_leaf_shared(&changed, node::ROOT_EXCLUSIVE_END)
            .await
            .is_err()
    );
    assert!(
        store
            .read_leaf_fragment(&changed, 2, node::ROOT_EXCLUSIVE_END)
            .await
            .is_err()
    );
    assert!(store.read_leaf_uncached(&written.child_ref).await.is_err());
}

#[derive(Clone, Copy)]
enum Touch {
    Point,
    Whole,
}

async fn touch(
    store: &NodeStore,
    child: &pb::FragmentTreeChild,
    fragments: &[Fragment],
    how: Touch,
) {
    match how {
        Touch::Point => {
            let point = store
                .read_leaf_fragment(child, fragments[1].id, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap();
            assert_eq!(point.as_ref(), Some(&fragments[1]));
        }
        Touch::Whole => assert_eq!(store.read_leaf(child).await.unwrap(), fragments),
    }
}

/// Points neither advance whole-read admission nor prevent its cache hits.
/// Reads after deleting the object prove whole reads admitted the leaf.
#[rstest]
#[case::point_then_point(Touch::Point, Touch::Point, 2)]
#[case::whole_then_point(Touch::Whole, Touch::Point, 1)]
#[case::point_then_whole(Touch::Point, Touch::Whole, 1)]
#[tokio::test]
async fn point_reads_preserve_whole_read_admission(
    #[case] first: Touch,
    #[case] second: Touch,
    #[case] additional_whole_reads: usize,
) {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let fragments: Vec<_> = (0..4).map(make_fragment).collect();
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    for how in [first, second] {
        touch(&store, &written.child_ref, &fragments, how).await;
    }
    let key = LeafKey::new(&store, &written.child_ref).unwrap();
    let leaves = store.cache.as_ref().unwrap();
    assert!(leaves.get_with_key(&key).await.is_none());
    for read in 0..additional_whole_reads {
        touch(&store, &written.child_ref, &fragments, Touch::Whole).await;
        assert_eq!(
            leaves.get_with_key(&key).await.is_some(),
            read + 1 == additional_whole_reads
        );
    }

    let path = store.resolve_path(&written.child_ref.path).unwrap();
    store.object_store.inner.delete(&path).await.unwrap();
    let point = store
        .read_leaf_fragment(&written.child_ref, 3, node::ROOT_EXCLUSIVE_END)
        .await;
    assert_eq!(point.unwrap().as_ref(), Some(&fragments[3]));
    let missing = store
        .read_leaf_fragment(&written.child_ref, 9, node::ROOT_EXCLUSIVE_END)
        .await;
    assert!(missing.unwrap().is_none());
    assert_eq!(
        store.read_leaf(&written.child_ref).await.unwrap(),
        fragments
    );

    let mut changed = written.child_ref.clone();
    changed.visible_rows += 1;
    assert!(
        store
            .read_leaf_fragment(&changed, 3, node::ROOT_EXCLUSIVE_END)
            .await
            .is_err()
    );
}

/// Repeated point reads still need the object until whole reads admit it.
#[tokio::test]
async fn point_reads_do_not_admit_leaves() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let fragments: Vec<_> = (0..4).map(make_fragment).collect();
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    for _ in 0..3 {
        touch(&store, &written.child_ref, &fragments, Touch::Point).await;
    }

    let path = store.resolve_path(&written.child_ref.path).unwrap();
    store.object_store.inner.delete(&path).await.unwrap();
    assert!(
        store
            .read_leaf_fragment(&written.child_ref, 2, node::ROOT_EXCLUSIVE_END)
            .await
            .is_err()
    );
}

/// Header checks pass a leaf whose other records carry files or overlays
/// that fail full validation. Selective reads validate the requested record;
/// every whole read rejects the corruption without admitting the leaf.
#[rstest]
#[case::file_mapping(|fragment: &mut Fragment| fragment.files[0].fields = Arc::from([0, 1, 2]))]
#[case::overlay_mapping(|fragment: &mut Fragment| {
    let mut data_file = fragment.files[0].clone();
    data_file.fields = Arc::from([0, 1, 2]);
    fragment.overlays.push(crate::format::overlay::DataOverlayFile {
        data_file,
        coverage: crate::format::overlay::OverlayCoverage::dense(
            roaring::RoaringBitmap::from_iter([0]),
        ),
        committed_version: 1,
    });
})]
#[tokio::test]
async fn point_reads_leave_unrequested_corruption_for_whole_reads(
    #[case] corrupt: fn(&mut Fragment),
) {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let mut fragments: Vec<_> = (0..4).map(make_fragment).collect();
    corrupt(&mut fragments[2]);
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    let data_file = &fragments[2].files[0].path;
    let names_leaf_and_file = |error: &Error| {
        let message = error.to_string();
        assert!(message.contains(&written.child_ref.path), "{message}");
        assert!(message.contains(data_file.as_str()), "{message}");
    };
    let whole = store
        .read_leaf_uncached(&written.child_ref)
        .await
        .unwrap_err();
    assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");
    names_leaf_and_file(&whole);

    for _ in 0..3 {
        let point = store
            .read_leaf_fragment(&written.child_ref, 0, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap();
        assert_eq!(point.as_ref(), Some(&fragments[0]));
        let point = store
            .read_leaf_fragment(&written.child_ref, 2, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap_err();
        assert!(matches!(point, Error::CorruptFile { .. }), "{point}");
        names_leaf_and_file(&point);
        let whole = store
            .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
            .await
            .err()
            .expect("a damaged leaf must fail a whole read");
        assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");
        names_leaf_and_file(&whole);
    }
    let key = LeafKey::new(&store, &written.child_ref).unwrap();
    let leaves = store.cache.as_ref().unwrap();
    assert!(leaves.get_with_key(&key).await.is_none());
}

#[tokio::test]
async fn header_reads_report_the_leaf_they_read() {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let written = store
        .write_leaf(&[make_fragment(u64::from(u32::MAX) + 1)], 0)
        .await
        .unwrap();
    let object = store.resolve_child(&written.child_ref).unwrap().1;
    let error = store
        .read_leaf_headers(&written.child_ref, node::ROOT_EXCLUSIVE_END)
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::CorruptFile { path, .. } if *path == object),
        "{error}"
    );
    assert!(
        error.to_string().contains("out of range or order"),
        "{error}"
    );
}

/// A set read decodes every record of an uncached leaf, so a malformed
/// record it did not request fails the sighting and the admitting read,
/// and the leaf is never admitted.
#[tokio::test]
async fn set_reads_validate_unrequested_records_on_every_uncached_read() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(1024 * 1024));
    let mut fragments: Vec<_> = (0..2).map(make_fragment).collect();
    fragments[1].files[0] = DataFile::new(
        "invalid.lance",
        vec![0],
        vec![],
        ConcreteFileVersion::V2_1,
        None,
        None,
    );
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    for _ in 0..3 {
        let error = store
            .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
            .await
            .err()
            .expect("a malformed unrequested record must fail the read");
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(
            error.to_string().contains("fewer column_indices"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn retained_validation_reads_bypass_the_cache_for_shared_reads() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(1024 * 1024));
    let fragments: Vec<_> = (0..2).map(make_fragment).collect();
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    for _ in 0..2 {
        assert_eq!(
            store.read_leaf(&written.child_ref).await.unwrap(),
            fragments
        );
    }
    store.retain_validation_reads().unwrap();
    let path = store.resolve_path(&written.child_ref.path).unwrap();
    store.object_store.inner.delete(&path).await.unwrap();
    let error = store
        .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
        .await
        .err()
        .expect("retained reads must fetch the deleted leaf");
    assert!(error.is_not_found(), "{error}");
    // Local-store errors use platform path separators.
    assert!(
        error
            .to_string()
            .replace('\\', "/")
            .contains(&written.child_ref.path),
        "{error}"
    );
}

/// The admitting read and later hits hand out the cached leaf itself, so
/// admission copies no record.
#[tokio::test]
async fn admission_shares_the_cached_leaf_with_its_reader() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    store.set_cache(LanceCache::with_capacity(1024 * 1024));
    let fragments: Vec<_> = (0..4).map(make_fragment).collect();
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    let child = &written.child_ref;
    let first = store
        .read_leaf_shared(child, node::ROOT_EXCLUSIVE_END)
        .await
        .unwrap();
    assert!(matches!(&first, LeafRead::Owned(read) if *read == fragments));
    let key = LeafKey::new(&store, child).unwrap();
    let leaves = store.cache.as_ref().unwrap();
    assert!(leaves.get_with_key(&key).await.is_none());
    for _ in 0..2 {
        let LeafRead::Cached(read) = store
            .read_leaf_shared(child, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap()
        else {
            panic!("the second and later reads must share the cached leaf");
        };
        let cached = leaves.get_with_key(&key).await.unwrap();
        assert!(Arc::ptr_eq(&read, &cached));
        assert_eq!(read.fragments, fragments);
    }
}

/// A point read decodes one record but checks every parent field a whole
/// read checks, so a tampered reference fails either way.
#[rstest]
#[case::extra_key(|child: &mut pb::FragmentTreeChild| child.num_keys += 1)]
#[case::extra_row(|child: &mut pb::FragmentTreeChild| child.total_rows += 1)]
#[case::extra_visible_row(|child: &mut pb::FragmentTreeChild| child.visible_rows += 1)]
#[case::fence_above_first_key(|child: &mut pb::FragmentTreeChild| child.min_key = 1)]
#[tokio::test]
async fn point_reads_reject_references_whole_reads_reject(
    #[case] tamper: fn(&mut pb::FragmentTreeChild),
) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let mut fragments: Vec<_> = (0..4).map(make_fragment).collect();
    fragments[2].deletion_file = Some(DeletionFile {
        read_version: 1,
        id: 1,
        file_type: DeletionFileType::Bitmap,
        num_deleted_rows: Some(0),
        base_id: None,
        inline: None,
    });
    let written = store.write_leaf(&fragments, 0).await.unwrap();
    for fragment in &fragments {
        let point = store
            .read_leaf_fragment(&written.child_ref, fragment.id, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap();
        assert_eq!(point.as_ref(), Some(fragment));
    }
    let missing = store
        .read_leaf_fragment(&written.child_ref, 4, node::ROOT_EXCLUSIVE_END)
        .await;
    assert!(missing.unwrap().is_none());

    let mut corrupt = written.child_ref.clone();
    tamper(&mut corrupt);
    let whole = store.read_leaf(&corrupt).await.unwrap_err();
    let point = store
        .read_leaf_fragment(&corrupt, 3, node::ROOT_EXCLUSIVE_END)
        .await
        .unwrap_err();
    assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");
    assert!(matches!(point, Error::CorruptFile { .. }), "{point}");
}

#[tokio::test]
async fn foreign_interior_buffer_inherits_dataset_on_file_refs() {
    let base = TempObjDir::default();
    let mut store = test_store(base.clone());
    let children = (0..2)
        .map(|id| {
            node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
        })
        .collect();
    let file = make_backfill_data_file(0, 0);
    let mutation = pb::FragmentTreeMutation {
        action_sequence: 1,
        action: Some(action::add_data_file(0, &file)),
        fragment_count_delta: 0,
        total_rows_delta: 0,
        visible_rows_delta: 0,
    };
    let mut written = store
        .write_interior(children, vec![mutation])
        .await
        .unwrap();
    let object_store = store.object_store.clone();
    let dataset_base = store.base.clone();
    store.set_foreign_bases(HashMap::from([(7, (object_store, dataset_base))]));
    written.child_ref.base_id = Some(7);
    let node = store.read_interior(&written.child_ref).await.unwrap();
    let Some(pb::fragment_action::Action::AddDataFile(add)) = node.buffer[0]
        .action
        .as_ref()
        .and_then(|a| a.action.as_ref())
    else {
        panic!("expected AddDataFile");
    };
    assert_eq!(add.file.as_ref().and_then(|file| file.base_id), Some(7));
}

#[tokio::test]
async fn foreign_interior_buffer_preserves_column_lineage_ownership() {
    let source_dir = TempObjDir::default();
    let source = test_store(source_dir.clone());
    let fragment = column_lineage_fragment(2);
    let mutation = pb::FragmentTreeMutation {
        action_sequence: 1,
        action: Some(action::upsert_fragment(&fragment).unwrap()),
        fragment_count_delta: 1,
        total_rows_delta: fragment.physical_rows.unwrap_or(0) as i64,
        visible_rows_delta: fragment.num_rows().unwrap_or(0) as i64,
    };
    let children = (0..2)
        .map(|id| {
            node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
        })
        .collect();
    let mut written = source
        .write_interior(children, vec![mutation])
        .await
        .unwrap();
    let dest_dir = TempObjDir::default();
    let mut dest = test_store(dest_dir.clone());
    dest.set_foreign_bases(HashMap::from([(
        7,
        (source.object_store.clone(), source.base.clone()),
    )]));
    written.child_ref.base_id = Some(7);
    let node = dest.read_interior(&written.child_ref).await.unwrap();
    let Some(pb::fragment_action::Action::UpsertFragment(encoded)) = node.buffer[0]
        .action
        .as_ref()
        .and_then(|a| a.action.as_ref())
    else {
        panic!("expected UpsertFragment");
    };
    let read = Fragment::try_from(encoded.clone()).unwrap();
    let mut expected = fragment;
    for file in &mut expected.files {
        file.base_id = Some(7);
    }
    assert_eq!(read, expected);
}

#[tokio::test]
async fn leaf_reads_share_field_list_allocations_across_leaves() {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let first = store.write_leaf(&[make_fragment(0)], 0).await.unwrap();
    let second = store.write_leaf(&[make_fragment(1)], 0).await.unwrap();
    let a = store.read_leaf(&first.child_ref).await.unwrap();
    let b = store.read_leaf(&second.child_ref).await.unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &a[0].files[0].fields,
        &b[0].files[0].fields
    ));
    assert!(std::sync::Arc::ptr_eq(
        &a[0].files[0].column_indices,
        &b[0].files[0].column_indices
    ));
}

fn deletion_file(num_deleted_rows: usize) -> DeletionFile {
    DeletionFile {
        read_version: 1,
        id: num_deleted_rows as u64,
        file_type: DeletionFileType::Array,
        num_deleted_rows: Some(num_deleted_rows),
        inline: None,
        base_id: None,
    }
}

/// Replays target fragment 1 in the first leaf. Every fragment of the
/// second leaf holds one more backfill file than the first leaf's, so any
/// file a replay leaves or adds on fragment 1 has a decoded twin in fragment
/// 5 with equal field lists.
async fn decoded_leaf_and_twin(store: &NodeStore) -> (BTreeMap<u64, Fragment>, Fragment) {
    let first: Vec<Fragment> = (0..4)
        .map(|id| {
            let mut fragment = make_fragment_with_files(id, 3);
            fragment.physical_rows = Some(10);
            fragment.deletion_file = Some(deletion_file(2));
            fragment
        })
        .collect();
    let second: Vec<Fragment> = (4..8).map(|id| make_fragment_with_files(id, 4)).collect();
    let first = store.write_leaf(&first, 0).await.unwrap();
    let second = store.write_leaf(&second, 0).await.unwrap();
    let leaf = store
        .read_leaf(&first.child_ref)
        .await
        .unwrap()
        .into_iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();
    let twin = store
        .read_leaf(&second.child_ref)
        .await
        .unwrap()
        .swap_remove(1);
    assert_eq!(twin.id, 5);
    (leaf, twin)
}

fn assert_shares_field_lists(fragment: &Fragment, twin: &Fragment) {
    for file in &fragment.files {
        let decoded = twin
            .files
            .iter()
            .find(|decoded| decoded.fields == file.fields)
            .unwrap_or_else(|| panic!("fragment {} has no twin for {}", twin.id, file.path));
        assert!(
            Arc::ptr_eq(&file.fields, &decoded.fields),
            "fields of {} are a private copy",
            file.path
        );
        assert!(
            Arc::ptr_eq(&file.column_indices, &decoded.column_indices),
            "column_indices of {} are a private copy",
            file.path
        );
    }
}

fn replay_of_fragment_one(
    action: pb::FragmentAction,
    visible_rows_delta: i64,
) -> pb::FragmentTreeMutation {
    pb::FragmentTreeMutation {
        action_sequence: 1,
        action: Some(action),
        fragment_count_delta: 0,
        total_rows_delta: 0,
        visible_rows_delta,
    }
}

#[rstest]
#[case::add_deletion_file(action::add_deletion_file(1, &deletion_file(3)).unwrap(), -1)]
#[case::clear_deletion_file(action::clear_deletion_file(1), 2)]
#[case::remove_data_file(action::remove_data_file(1, data_file_path(1, 1)), 0)]
#[case::replace_data_file(
    action::replace_data_file(1, &data_file_path(1, 1), &make_replacement_data_file(1, 0)),
    0
)]
#[tokio::test]
async fn replay_keeps_leaf_field_lists_for_actions_without_data_files(
    #[case] action: pb::FragmentAction,
    #[case] visible_rows_delta: i64,
) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let (mut leaf, twin) = decoded_leaf_and_twin(&store).await;
    let actions = vec![replay_of_fragment_one(action, visible_rows_delta)];
    let mut expected = leaf.clone();
    crate::fragment_metadata::content::apply_verified(&mut expected, actions.clone()).unwrap();

    store.apply_verified(&mut leaf, actions).unwrap();

    assert_eq!(leaf, expected);
    assert_shares_field_lists(&leaf[&1], &twin);
}

#[rstest]
#[case::upsert_fragment(action::upsert_fragment(&{
    let mut fragment = make_fragment_with_files(1, 4);
    fragment.physical_rows = Some(10);
    fragment.deletion_file = Some(deletion_file(2));
    fragment
}).unwrap())]
#[case::add_data_file(action::add_data_file(1, &make_backfill_data_file(1, 2)))]
#[tokio::test]
async fn replay_shares_field_lists_of_new_data_files(#[case] action: pb::FragmentAction) {
    let base = TempObjDir::default();
    let store = test_store(base.clone());
    let (mut leaf, twin) = decoded_leaf_and_twin(&store).await;
    let actions = vec![replay_of_fragment_one(action, 0)];
    let mut expected = leaf.clone();
    crate::fragment_metadata::content::apply_verified(&mut expected, actions.clone()).unwrap();

    store.apply_verified(&mut leaf, actions).unwrap();

    assert_eq!(leaf, expected);
    assert_eq!(leaf[&1].files.len(), 4);
    assert_shares_field_lists(&leaf[&1], &twin);
}

#[tokio::test]
async fn foreign_leaf_preserves_column_lineage_ownership() {
    let source_dir = TempObjDir::default();
    let source = test_store(source_dir.clone());
    let first = column_lineage_fragment(0);
    let mut second = column_lineage_fragment(1);
    // A chained clone already names the original carrier's owner.
    second.files[1].base_id = Some(9);
    let fragments = vec![first, second];
    let written = source.write_leaf(&fragments, 0).await.unwrap();

    let dest_dir = TempObjDir::default();
    let mut dest = test_store(dest_dir.clone());
    dest.set_foreign_bases(HashMap::from([(
        4,
        (source.object_store.clone(), source.base.clone()),
    )]));
    dest.set_cache(LanceCache::with_capacity(1024 * 1024));
    let mut child = written.child_ref.clone();
    child.base_id = Some(4);
    let mut expected = fragments;
    for fragment in &mut expected {
        for file in &mut fragment.files {
            if file.base_id.is_none() {
                file.base_id = Some(4);
            }
        }
    }
    // First sighting, admission and a cache hit keep the stamped owners.
    for _ in 0..3 {
        let selected = dest
            .read_leaf_shared(&child, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap()
            .select(&RoaringBitmap::from_iter([0, 1, 9]));
        assert_eq!(selected, expected);
    }
    let read = dest.read_leaf(&child).await.unwrap();
    assert_eq!(read, expected);
    // Point reads must apply the same inherited ownership as whole leaves.
    for fragment in expected {
        assert_eq!(
            dest.read_leaf_fragment(&child, fragment.id, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap(),
            Some(fragment)
        );
    }
}
#[rstest]
#[case(0)]
#[case(15)]
#[case(17)]
fn rejects_invalid_root_uuid_length(#[case] length: usize) {
    let result = root_path(&vec![0; length]);
    assert!(matches!(result, Err(Error::CorruptFile { .. })));
}

#[test]
fn root_uuid_uses_display_order() {
    let uuid = uuid::Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").unwrap();
    assert_eq!(
        root_path(uuid.as_bytes()).unwrap(),
        "_bt/root/00112233-4455-6677-8899-aabbccddeeff.root"
    );
}
