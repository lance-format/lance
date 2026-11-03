// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod index_over_committed_sp;
mod tagged_mutations;

use super::*;
use crate::dataset::builder::DatasetBuilder;
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::{
    build_frag_reuse_index_metadata, build_new_frag_reuse_index, decode_frag_reuse_ledger,
    load_raw_frag_reuse_content,
};
use crate::index::frag_reuse_reader::tests::{field, persist_fixture, prepare, prepare_partition};
use crate::session::Session;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
use arrow_array::Int32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::Schema as ArrowSchema;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::frag_reuse::{
    FRAG_REUSE_INDEX_NAME, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_index::scalar::ScalarIndexParams;
use lance_table::format::Fragment;
use lance_table::format::pb::fragment_reuse_index_details::{
    FragmentDigest, StablePartition, Transition, transition,
};
use lance_table::system_index::frag_reuse::FragDigest;
use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};

async fn ram_fixture(frag_count: u32, rows_per_fragment: u32) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(
            FragmentCount::from(frag_count),
            FragmentRowCount::from(rows_per_fragment),
        )
        .await
        .unwrap()
}

async fn disk_fixture(uri: &str, frag_count: u32, rows_per_fragment: u32) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            uri,
            FragmentCount::from(frag_count),
            FragmentRowCount::from(rows_per_fragment),
        )
        .await
        .unwrap()
}

async fn fresh_session(uri: &str) -> Dataset {
    DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap()
}

async fn reserve(dataset: &mut Dataset, num_fragments: u32) {
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::ReserveFragments { num_fragments },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
}

/// The rewrite's complete entry is built against `dataset`, the
/// version the rewrite reads.
async fn tagged_rewrite(
    dataset: &Dataset,
    old_fragments: Vec<Fragment>,
    new_fragments: Vec<Fragment>,
    transitions: Vec<Transition>,
) -> Operation {
    Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments,
            new_fragments,
        }],
        rewritten_indices: vec![],
        frag_reuse_index: Some(
            crate::index::frag_reuse::frag_reuse_entry_appending(dataset, transitions)
                .await
                .unwrap(),
        ),
    }
}

async fn commit_sp(dataset: &Dataset, read_version: u64, operation: Operation) -> Result<Dataset> {
    CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(Transaction::new(read_version, operation, None))
        .await
}

async fn fri_ledger(dataset: &Dataset) -> (IndexMetadata, FragReuseLedger) {
    let stored = crate::index::loading::load_all_indices(dataset)
        .await
        .unwrap();
    let entry = stored
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .unwrap()
        .clone();
    let ledger = decode_frag_reuse_ledger(dataset, &entry).await.unwrap();
    (entry, ledger)
}

/// Row-map ids of the entry's stable-partition transitions.
async fn sp_map_ids(dataset: &Dataset) -> std::collections::HashSet<String> {
    let (_, ledger) = fri_ledger(dataset).await;
    ledger
        .transitions()
        .iter()
        .filter_map(|transition| match transition.mapping() {
            Mapping::StablePartition(partition) => Some(partition.map_id.clone()),
            _ => None,
        })
        .collect()
}

/// (stable-partition, ordered-compaction) transition counts.
fn count_mappings(ledger: &FragReuseLedger) -> (usize, usize) {
    ledger
        .transitions()
        .iter()
        .fold((0, 0), |(sp, oc), t| match t.mapping() {
            Mapping::StablePartition(_) => (sp + 1, oc),
            Mapping::OrderedCompaction(_) => (sp, oc + 1),
        })
}

async fn sorted_values(dataset: &Dataset) -> Vec<i32> {
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let mut values: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    values.sort_unstable();
    values
}

/// Copy the rows of `source_ids` into one fresh fragment and commit
/// the deferred-compaction Rewrite exactly as the v0 writer does: the
/// legacy FRI entry rides the in-memory `frag_reuse_index` field.
async fn commit_v0_compaction(dataset: &mut Dataset, source_ids: &[u64], dest_id: u64) {
    let transaction = v0_compaction(dataset, source_ids, dest_id).await;
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
}

async fn v0_compaction(dataset: &Dataset, source_ids: &[u64], dest_id: u64) -> Transaction {
    let old_fragments: Vec<Fragment> = source_ids
        .iter()
        .map(|id| {
            dataset
                .fragments()
                .iter()
                .find(|f| f.id == *id)
                .unwrap()
                .clone()
        })
        .collect();
    let batch = {
        let mut scan = dataset.scan();
        scan.with_fragments(old_fragments.clone());
        scan.try_into_batch().await.unwrap()
    };
    let txn = InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![batch])
        .await
        .unwrap();
    let Operation::Append { mut fragments } = txn.operation else {
        unreachable!()
    };
    assert_eq!(fragments.len(), 1);
    fragments[0].id = dest_id;
    let mut changed_row_addrs = RoaringTreemap::new();
    for frag in &old_fragments {
        for offset in 0..frag.physical_rows.unwrap() as u64 {
            changed_row_addrs.insert((frag.id << 32) + offset);
        }
    }
    let mut serialized = Vec::new();
    changed_row_addrs.serialize_into(&mut serialized).unwrap();
    let entry = build_new_frag_reuse_index(
        &mut dataset.clone(),
        vec![FragReuseGroup {
            changed_row_addrs: serialized,
            old_frags: old_fragments.iter().map(FragDigest::from).collect(),
            new_frags: fragments.iter().map(FragDigest::from).collect(),
        }],
        RoaringBitmap::from_iter([dest_id as u32]),
        dataset.manifest.version,
    )
    .await
    .unwrap();
    assert_eq!(entry.index_version, 0);
    Transaction::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments,
                new_fragments: fragments,
            }],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        },
        None,
    )
}

/// With `external`, a history whose details live in their own file, so a
/// session that did not build an entry has to read it from storage.
async fn reuse_fixture(uri: &str, external: bool) -> (Dataset, usize) {
    use crate::utils::test::{commit_padding_reuse_history, inline_padding_capacity};
    let mut dataset = disk_fixture(uri, 2, 4).await;
    let padding = if external {
        inline_padding_capacity() + 1
    } else {
        0
    };
    if external {
        commit_padding_reuse_history(&mut dataset, padding).await;
    }
    reserve(&mut dataset, 5).await;
    (dataset, padding)
}

fn take_reuse_details_reads(dataset: &Dataset) -> usize {
    dataset
        .object_store
        .io_stats_incremental()
        .requests
        .iter()
        .filter(|request| {
            request.method.starts_with("get")
                && request
                    .path
                    .as_ref()
                    .ends_with(lance_index::frag_reuse::FRAG_REUSE_DETAILS_FILE_NAME)
        })
        .count()
}

#[rstest::rstest]
#[tokio::test]
async fn v0_compaction_built_in_another_session_commits(#[values(false, true)] external: bool) {
    let dir = TempStrDir::default();
    let (dataset, padding) = reuse_fixture(dir.as_str(), external).await;
    let dest_id = dataset.manifest.max_fragment_id.unwrap() as u64;
    let compaction = v0_compaction(&dataset, &[0], dest_id).await;

    let mut committer = fresh_session(dir.as_str()).await;
    take_reuse_details_reads(&committer);
    committer
        .apply_commit(compaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    assert_eq!(take_reuse_details_reads(&committer) > 0, external);
    let history = fresh_session(dir.as_str())
        .await
        .frag_reuse_index()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(history.details.versions.len(), padding + 1);
    assert_eq!(history.details.versions[padding].old_frag_ids(), vec![0]);
}

#[rstest::rstest]
#[tokio::test]
async fn v0_compaction_built_from_an_older_snapshot_is_refused(
    #[values(false, true)] external: bool,
) {
    let dir = TempStrDir::default();
    let (dataset, padding) = reuse_fixture(dir.as_str(), external).await;
    let dest_id = dataset.manifest.max_fragment_id.unwrap() as u64;
    let stale = v0_compaction(&dataset, &[0], dest_id - 1).await;
    let mut other = fresh_session(dir.as_str()).await;
    commit_v0_compaction(&mut other, &[1], dest_id).await;

    let mut latest = fresh_session(dir.as_str()).await;
    let error = latest
        .apply_commit(stale, &Default::default(), &Default::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("plus one version of its own"),
        "{error}"
    );
    let history = fresh_session(dir.as_str())
        .await
        .frag_reuse_index()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(history.details.versions.len(), padding + 1);
    assert_eq!(history.details.versions[padding].old_frag_ids(), vec![1]);
}

/// Matrix row 1: append, an unrelated delete and a fragment
/// reservation land between the rewrite's read version and its
/// commit; the retry reassembles against the latest entry and every
/// transaction survives.
#[tokio::test]
async fn sp_lands_over_unrelated_concurrent_ops() {
    let mut dataset = ram_fixture(2, 4).await;
    reserve(&mut dataset, 20).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&dataset).await;
    let read_version = dataset.manifest.version;

    // Writer A: append four rows, delete two of them, reserve ids.
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let appended = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from_iter_values(100..104))],
    )
    .unwrap();
    let mut writer_a = InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![appended])
        .await
        .unwrap();
    writer_a.delete("i >= 102").await.unwrap();
    reserve(&mut writer_a, 5).await;

    // Writer B: the stable-partition rewrite reads the old version.
    let committed = commit_sp(
        &dataset,
        read_version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    let live_ids: Vec<u64> = committed.fragments().iter().map(|f| f.id).collect();
    assert_eq!(live_ids.len(), 3);
    assert_eq!(&live_ids[..2], &[10, 11]);
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..102);
    assert_eq!(sorted_values(&committed).await, expected);
    let (entry, ledger) = fri_ledger(&committed).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (1, 0));
}

/// Matrix row 2 (same session): a v0 deferred compaction of disjoint
/// fragments lands first; the stable-partition retry lifts its legacy
/// entry to index_version 1 and appends, so both records survive.
#[tokio::test]
async fn sp_reassembles_over_concurrent_v0_compaction() {
    let mut dataset = ram_fixture(2, 4).await;
    reserve(&mut dataset, 30).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&dataset).await;
    let read_version = dataset.manifest.version;

    // Writer A: append a fragment, then v0-compact it.
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let appended = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from_iter_values(100..104))],
    )
    .unwrap();
    let mut writer_a = InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![appended])
        .await
        .unwrap();
    let appended_id = writer_a.fragments().last().unwrap().id;
    reserve(&mut writer_a, 5).await;
    let dest_id = writer_a.manifest.max_fragment_id.unwrap() as u64;
    commit_v0_compaction(&mut writer_a, &[appended_id], dest_id).await;
    let (their_entry, their_ledger) = fri_ledger(&writer_a).await;
    assert_eq!(their_entry.index_version, 0);
    assert_eq!(count_mappings(&their_ledger), (0, 1));
    let their_content = load_raw_frag_reuse_content(&writer_a, &their_entry)
        .await
        .unwrap();

    let committed = commit_sp(
        &dataset,
        read_version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    let (entry, ledger) = fri_ledger(&committed).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (1, 1));
    assert!(ledger.consumer(appended_id as u32).is_some());
    assert!(ledger.consumer(0).is_some());
    let content = load_raw_frag_reuse_content(&committed, &entry)
        .await
        .unwrap();
    assert!(content.starts_with(&their_content));
    let live_ids: Vec<u64> = committed.fragments().iter().map(|f| f.id).collect();
    assert_eq!(live_ids, vec![10, 11, dest_id]);
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&committed).await, expected);
}

/// Matrix row 2 (fresh session): the retrying writer cannot see the
/// committed compaction's in-memory `frag_reuse_index` (a fresh
/// session decodes the transaction file, where the field is None);
/// its legacy record must be picked up from the CURRENT manifest
/// entry during reassembly.
#[tokio::test]
async fn sp_reassembles_over_concurrent_v0_compaction_fresh_session() {
    let dir = TempStrDir::default();
    let mut writer_a = disk_fixture(dir.as_str(), 2, 4).await;
    reserve(&mut writer_a, 30).await;

    let writer_b = fresh_session(dir.as_str()).await;
    let old_fragments: Vec<Fragment> = writer_b.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&writer_b).await;
    let read_version = writer_b.manifest.version;

    let schema = Arc::new(ArrowSchema::from(writer_a.schema()));
    let appended = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from_iter_values(100..104))],
    )
    .unwrap();
    let mut writer_a = InsertBuilder::new(Arc::new(writer_a.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![appended])
        .await
        .unwrap();
    let appended_id = writer_a.fragments().last().unwrap().id;
    reserve(&mut writer_a, 5).await;
    let dest_id = writer_a.manifest.max_fragment_id.unwrap() as u64;
    commit_v0_compaction(&mut writer_a, &[appended_id], dest_id).await;
    let (their_entry, _) = fri_ledger(&writer_a).await;
    let their_content = load_raw_frag_reuse_content(&writer_a, &their_entry)
        .await
        .unwrap();

    let committed = commit_sp(
        &writer_b,
        read_version,
        tagged_rewrite(&writer_b, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    // Verify through yet another session: everything below comes from
    // the committed manifest, not anyone's cache.
    let verify = fresh_session(dir.as_str()).await;
    assert_eq!(verify.manifest.version, committed.manifest.version);
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (1, 1));
    let content = load_raw_frag_reuse_content(&verify, &entry).await.unwrap();
    assert!(content.starts_with(&their_content));
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&verify).await, expected);
}

/// Matrix row 3 (fresh session): the committed rewrite consumed our
/// transition sources; the conflict must be detected from its
/// transaction file alone and rejected outright: the intent can never
/// succeed once its fragments are gone.
#[tokio::test]
async fn sp_double_consumption_is_rejected_fresh_session() {
    let dir = TempStrDir::default();
    let mut writer_a = disk_fixture(dir.as_str(), 2, 4).await;
    reserve(&mut writer_a, 40).await;

    let writer_b = fresh_session(dir.as_str()).await;
    let old_fragments: Vec<Fragment> = writer_b.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&writer_b).await;
    let read_version = writer_b.manifest.version;

    // Writer A rewrites the same fragments (a plain compaction).
    let batch = writer_a.scan().try_into_batch().await.unwrap();
    let txn = InsertBuilder::new(Arc::new(writer_a.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![batch])
        .await
        .unwrap();
    let Operation::Append { mut fragments } = txn.operation else {
        unreachable!()
    };
    fragments[0].id = 30;
    writer_a
        .apply_commit(
            Transaction::new(
                writer_a.manifest.version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: writer_a.fragments().iter().cloned().collect(),
                        new_fragments: fragments,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    let error = commit_sp(
        &writer_b,
        read_version,
        tagged_rewrite(&writer_b, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, Error::IncompatibleTransaction { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("consumed fragments"), "{error}");
}

/// Matrix row 3, destination direction: a committed rewrite that
/// consumed the fragments our transitions PRODUCE is double
/// consumption too. Unit-level because a real writer cannot commit a
/// rewrite of fragments that do not exist yet; the rule still has to
/// hold for replayed/raced transactions that claim them.
#[tokio::test]
async fn sp_double_consumption_covers_destinations() {
    let dataset = test_dataset(8, 2).await;
    let transitions = vec![Transition {
        sources: vec![FragmentDigest {
            id: 0,
            physical_rows: 4,
            num_deleted_rows: 0,
        }],
        destinations: vec![FragmentDigest {
            id: 10,
            physical_rows: 4,
            num_deleted_rows: 0,
        }],
        mapping: Some(transition::Mapping::StablePartition(StablePartition {
            map_id: Uuid::new_v4().to_string(),
            map_size_bytes: 1,
            base_id: None,
        })),
    }];
    let ours = Transaction::new(
        1,
        tagged_rewrite(
            &dataset,
            vec![dataset.fragments()[0].clone()],
            vec![Fragment::new(10)],
            transitions,
        )
        .await,
        None,
    );
    let mut rebase = TransactionRebase::try_new(&dataset, ours, None)
        .await
        .unwrap();

    let theirs = Transaction::new(
        0,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![Fragment::new(10)],
                new_fragments: vec![Fragment::new(11)],
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
        None,
    );
    let error = rebase.check_txn(&theirs, 2).unwrap_err();
    assert!(
        matches!(error, Error::IncompatibleTransaction { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("consumed fragments"), "{error}");
}

/// Matrix row 4: a v0 FRI trim (cleanup CreateIndex) lands first; the
/// stable-partition retry reloads the trimmed entry from the current
/// manifest, re-appends its transition and revalidates, so the
/// trimmed-away record stays gone and the surviving one is kept.
#[tokio::test]
async fn sp_rebases_over_concurrent_fri_trim() {
    let mut dataset = ram_fixture(2, 4).await;

    let digest = |id: u64| FragDigest {
        id,
        physical_rows: 4,
        num_deleted_rows: 0,
    };
    let legacy_version = |dataset_version: u64, old_id: u64, new_id: u64| {
        let mut addrs = RoaringTreemap::new();
        for offset in 0..4u64 {
            addrs.insert((old_id << 32) + offset);
        }
        let mut serialized = Vec::new();
        addrs.serialize_into(&mut serialized).unwrap();
        FragReuseVersion {
            dataset_version,
            groups: vec![FragReuseGroup {
                changed_row_addrs: serialized,
                old_frags: vec![digest(old_id)],
                new_frags: vec![digest(new_id)],
            }],
        }
    };
    let full = FragReuseIndexDetails {
        versions: vec![legacy_version(1, 100, 110), legacy_version(2, 200, 210)],
    };
    let entry = build_frag_reuse_index_metadata(
        &dataset,
        None,
        full,
        RoaringBitmap::from_iter([110u32, 210]),
    )
    .await
    .unwrap();
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![entry],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    reserve(&mut dataset, 20).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&dataset).await;
    let read_version = dataset.manifest.version;

    // Writer A trims the oldest legacy version, as v0 cleanup does.
    let mut writer_a = dataset.clone();
    let stored = crate::index::loading::load_all_indices(&writer_a)
        .await
        .unwrap();
    let old_entry = stored
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .unwrap()
        .clone();
    let trimmed = FragReuseIndexDetails {
        versions: vec![legacy_version(2, 200, 210)],
    };
    let new_entry = build_frag_reuse_index_metadata(
        &writer_a,
        Some(&old_entry),
        trimmed,
        RoaringBitmap::from_iter([210u32]),
    )
    .await
    .unwrap();
    writer_a
        .apply_commit(
            Transaction::new(
                writer_a.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![new_entry],
                    removed_indices: vec![old_entry],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    let committed = commit_sp(
        &dataset,
        read_version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    let (entry, ledger) = fri_ledger(&committed).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(ledger.transitions().len(), 2);
    assert_eq!(count_mappings(&ledger), (1, 1));
    // The trimmed record stayed gone; the survivor and ours remain.
    assert!(ledger.consumer(100).is_none());
    assert!(ledger.consumer(200).is_some());
    assert!(ledger.consumer(0).is_some());
}

/// Matrix row 5: a user reindex (CreateIndex over the fragments the
/// rewrite consumes) is compatible: the rewrite defers remapping, the
/// index keeps its retired coverage as provenance, and afterwards the
/// tagged trim retains the record that index still needs.
#[tokio::test]
async fn sp_lands_over_concurrent_user_index_and_blocks_v0_trim() {
    let mut dataset = ram_fixture(2, 4).await;
    reserve(&mut dataset, 20).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&dataset).await;
    let read_version = dataset.manifest.version;

    let mut writer_a = dataset.clone();
    writer_a
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let mut committed = commit_sp(
        &dataset,
        read_version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    let stored = crate::index::loading::load_all_indices(&committed)
        .await
        .unwrap();
    let scalar = stored.iter().find(|idx| idx.name == "i_idx").unwrap();
    assert_eq!(
        scalar.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );
    let (entry, ledger) = fri_ledger(&committed).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (1, 0));
    // The index still answers through the translated coverage.
    assert_eq!(
        committed
            .count_rows(Some("i = 3".to_string()))
            .await
            .unwrap(),
        1
    );
    // The tagged trim must not release the history out from under
    // the not-yet-caught-up index: everything is retained, so the
    // cleanup is a no-op (no commit, entry untouched).
    let version = committed.manifest.version;
    crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut committed)
        .await
        .unwrap();
    assert_eq!(committed.manifest.version, version);
    let (entry_after, _) = fri_ledger(&committed).await;
    assert_eq!(entry_after.uuid, entry.uuid);
}

/// A distributed compaction plans at V and its driver holds a handle
/// at V+N. Building the complete entry from that later handle folds
/// the transitions other writers appended in between into this
/// rewrite's own additions; the commit path refuses such an entry
/// with the contract error instead of misreading it (the gatekeeper's
/// probe, kept as the proof of the check).
#[tokio::test]
async fn rewrite_entry_built_from_a_later_handle_is_refused_with_the_contract_error() {
    let dir = TempStrDir::default();
    let mut main = disk_fixture(dir.as_str(), 4, 4).await;
    main.create_index(
        &["i"],
        IndexType::Scalar,
        Some("i_idx".into()),
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    reserve(&mut main, 40).await;
    let planned = fresh_session(dir.as_str()).await;
    let read_version = planned.manifest.version;
    let ours_old: Vec<Fragment> = planned
        .fragments()
        .iter()
        .filter(|fragment| fragment.id < 2)
        .cloned()
        .collect();
    let (ours_transition, ours_destinations) = prepare_partition(&planned, &[0, 1], 10).await;

    let a_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|fragment| fragment.id >= 2)
        .cloned()
        .collect();
    let (a_transition, a_destinations) = prepare_partition(&main, &[2, 3], 20).await;
    let a_version = main.manifest.version;
    let main = commit_sp(
        &main,
        a_version,
        tagged_rewrite(&main, a_old, a_destinations, vec![a_transition]).await,
    )
    .await
    .unwrap();

    // The task read V; the entry is built from the latest handle.
    let operation = tagged_rewrite(&main, ours_old, ours_destinations, vec![ours_transition]).await;
    let err = commit_sp(&main, read_version, operation)
        .await
        .expect_err("an entry built from a later snapshot is refused");
    assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    assert!(
        err.to_string().contains("built from version"),
        "unexpected error: {err}"
    );
    // Nothing landed: the concurrent rewrite's history is intact and
    // every row is still served from it.
    let latest = fresh_session(dir.as_str()).await;
    assert_eq!(latest.manifest.version, main.manifest.version);
    assert_eq!(sorted_values(&latest).await, (0..16).collect::<Vec<_>>());
}

/// The same race with the entry built from the read-version snapshot,
/// as the contract requires: both disjoint rewrites land, the history
/// holds both transitions and every row stays served.
#[tokio::test]
async fn rewrite_entry_built_at_the_read_version_merges_with_a_disjoint_concurrent_rewrite() {
    let dir = TempStrDir::default();
    let mut main = disk_fixture(dir.as_str(), 4, 4).await;
    main.create_index(
        &["i"],
        IndexType::Scalar,
        Some("i_idx".into()),
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    reserve(&mut main, 40).await;
    let planned = fresh_session(dir.as_str()).await;
    let read_version = planned.manifest.version;
    let ours_old: Vec<Fragment> = planned
        .fragments()
        .iter()
        .filter(|fragment| fragment.id < 2)
        .cloned()
        .collect();
    let (ours_transition, ours_destinations) = prepare_partition(&planned, &[0, 1], 10).await;
    // Built from the snapshot the task read, before the other writer
    // lands.
    let operation =
        tagged_rewrite(&planned, ours_old, ours_destinations, vec![ours_transition]).await;

    let a_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|fragment| fragment.id >= 2)
        .cloned()
        .collect();
    let (a_transition, a_destinations) = prepare_partition(&main, &[2, 3], 20).await;
    let a_version = main.manifest.version;
    let main = commit_sp(
        &main,
        a_version,
        tagged_rewrite(&main, a_old, a_destinations, vec![a_transition]).await,
    )
    .await
    .unwrap();

    let committed = commit_sp(&main, read_version, operation).await.unwrap();
    assert_eq!(sorted_values(&committed).await, (0..16).collect::<Vec<_>>());
    let (_, ledger) = fri_ledger(&committed).await;
    assert_eq!(count_mappings(&ledger), (2, 0));
    assert_eq!(committed.fragments().len(), 4);
    let indexed = committed
        .scan()
        .filter("i = 5")
        .unwrap()
        .use_scalar_index(true)
        .try_into_batch()
        .await
        .unwrap();
    let scanned = committed
        .scan()
        .filter("i = 5")
        .unwrap()
        .use_scalar_index(false)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(indexed, scanned);
    assert_eq!(indexed.num_rows(), 1);
}

/// A retry re-reads the same intent. The rewrite's entry stays what
/// the caller supplied across attempts, so a second assembly against
/// a manifest that gained more concurrent transitions still adds
/// exactly this rewrite's transition: nothing another writer
/// recorded is ever taken for our own addition, and nothing of ours
/// is recorded twice.
#[tokio::test]
async fn sp_retry_reassembles_the_same_intent_without_absorbing_others() {
    let dir = TempStrDir::default();
    let mut main = disk_fixture(dir.as_str(), 6, 4).await;
    main.create_index(
        &["i"],
        IndexType::Scalar,
        Some("i_idx".into()),
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    reserve(&mut main, 60).await;
    let read_version_dataset = fresh_session(dir.as_str()).await;
    let read_version = read_version_dataset.manifest.version;

    // Ours: F0, F1 -> F10, F11, built against the read version.
    let ours_old: Vec<Fragment> = read_version_dataset
        .fragments()
        .iter()
        .filter(|f| f.id < 2)
        .cloned()
        .collect();
    let (our_transition, our_destinations) =
        prepare_partition(&read_version_dataset, &[0, 1], 10).await;
    let ours = Transaction::new(
        read_version,
        tagged_rewrite(
            &read_version_dataset,
            ours_old,
            our_destinations,
            vec![our_transition.clone()],
        )
        .await,
        None,
    );
    let Operation::Rewrite {
        frag_reuse_index: Some(intent),
        ..
    } = &ours.operation
    else {
        unreachable!()
    };
    let intent_uuid = intent.uuid;

    // Concurrent writer A lands F2, F3 -> F20, F21.
    let a_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|f| f.id == 2 || f.id == 3)
        .cloned()
        .collect();
    let (a_transition, a_destinations) = prepare_partition(&main, &[2, 3], 20).await;
    let a_version = main.manifest.version;
    let main = commit_sp(
        &main,
        a_version,
        tagged_rewrite(&main, a_old, a_destinations, vec![a_transition.clone()]).await,
    )
    .await
    .unwrap();

    // First attempt: merged onto A's entry, transaction untouched.
    let rebase = TransactionRebase::try_new(&read_version_dataset, ours.clone(), None)
        .await
        .unwrap();
    let (after_first, assembly) = rebase.finish_with_tagged_rewrite(&main).await.unwrap();
    let assembly = assembly.expect("a tagged rewrite assembles an entry");
    assert_eq!(
        after_first, ours,
        "the transaction keeps the caller's entry"
    );
    let Operation::Rewrite {
        frag_reuse_index: Some(kept),
        ..
    } = &after_first.operation
    else {
        unreachable!()
    };
    assert_eq!(kept.uuid, intent_uuid);
    let ids = |ledger: &FragReuseLedger| -> Vec<Vec<u64>> {
        ledger
            .transitions()
            .iter()
            .map(|transition| transition.sources().iter().map(|d| d.id).collect())
            .collect()
    };
    let ledger = decode_frag_reuse_ledger(&main, &assembly.entry)
        .await
        .unwrap();
    assert_eq!(ids(&ledger), vec![vec![2, 3], vec![0, 1]]);

    // Concurrent writer C lands F4, F5 -> F30, F31 before a retry.
    let c_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|f| f.id == 4 || f.id == 5)
        .cloned()
        .collect();
    let (c_transition, c_destinations) = prepare_partition(&main, &[4, 5], 30).await;
    let c_version = main.manifest.version;
    let main = commit_sp(
        &main,
        c_version,
        tagged_rewrite(&main, c_old, c_destinations, vec![c_transition]).await,
    )
    .await
    .unwrap();

    // Second attempt from the same transaction: A's and C's
    // transitions are the current entry's, ours is added once.
    let rebase = TransactionRebase::try_new(&read_version_dataset, after_first, None)
        .await
        .unwrap();
    let (after_second, assembly) = rebase.finish_with_tagged_rewrite(&main).await.unwrap();
    assert_eq!(after_second, ours);
    let ledger = decode_frag_reuse_ledger(&main, &assembly.unwrap().entry)
        .await
        .unwrap();
    assert_eq!(ids(&ledger), vec![vec![2, 3], vec![4, 5], vec![0, 1]]);

    // And the real commit lands the same way.
    let committed = commit_sp(&read_version_dataset, read_version, ours.operation)
        .await
        .unwrap();
    let (_, ledger) = fri_ledger(&committed).await;
    assert_eq!(ids(&ledger), vec![vec![2, 3], vec![4, 5], vec![0, 1]]);
    assert_eq!(sorted_values(&committed).await, (0..24).collect::<Vec<_>>());
}

/// Matrix row 6 (fresh session): two DISJOINT stable-partition
/// rewrites race and BOTH land. The second writer cannot see the
/// first through its transaction file; its in-loop reassembly reads
/// the committed entry, appends its own transition, and the merged
/// ledger validates. Overlap is rejected earlier and non-retryably
/// by the double-consumption rule, so what reaches reassembly is
/// proven disjoint.
#[tokio::test]
async fn sp_vs_sp_disjoint_merges_fresh_session() {
    let dir = TempStrDir::default();
    let mut writer_a = disk_fixture(dir.as_str(), 4, 4).await;
    writer_a
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    reserve(&mut writer_a, 40).await;
    let base_version = writer_a.manifest.version;

    let writer_b = fresh_session(dir.as_str()).await;
    let b_old: Vec<Fragment> = writer_b
        .fragments()
        .iter()
        .filter(|f| f.id < 2)
        .cloned()
        .collect();
    let (b_transition, b_destinations) = prepare_partition(&writer_b, &[0, 1], 10).await;
    let read_version = writer_b.manifest.version;
    assert_eq!(read_version, base_version);

    let a_old: Vec<Fragment> = writer_a
        .fragments()
        .iter()
        .filter(|f| f.id >= 2)
        .cloned()
        .collect();
    let (a_transition, a_destinations) = prepare_partition(&writer_a, &[2, 3], 20).await;
    let a_version = writer_a.manifest.version;
    commit_sp(
        &writer_a,
        a_version,
        tagged_rewrite(&writer_a, a_old, a_destinations, vec![a_transition.clone()]).await,
    )
    .await
    .unwrap();

    // No caller-visible error: the in-loop rebase merges.
    let committed = commit_sp(
        &writer_b,
        read_version,
        tagged_rewrite(&writer_b, b_old, b_destinations, vec![b_transition.clone()]).await,
    )
    .await
    .unwrap();
    // Exactly two commits landed on top of the shared base.
    assert_eq!(committed.manifest.version, base_version + 2);

    // Verified through yet another session: both transitions and
    // both row maps are in the committed entry.
    let verify = fresh_session(dir.as_str()).await;
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (2, 0));
    let map_id = |transition: &Transition| match transition.mapping.as_ref().unwrap() {
        transition::Mapping::StablePartition(partition) => partition.map_id.clone(),
        _ => unreachable!(),
    };
    let merged = sp_map_ids(&verify).await;
    assert!(merged.contains(&map_id(&a_transition)));
    assert!(merged.contains(&map_id(&b_transition)));
    let live_ids: Vec<u64> = verify.fragments().iter().map(|f| f.id).collect();
    assert_eq!(live_ids, vec![10, 11, 20, 21]);

    // Both row maps translate: full scan and translated index
    // queries stay value-identical across both partitions.
    assert_eq!(sorted_values(&verify).await, (0..16).collect::<Vec<_>>());
    let filtered = |predicate: &'static str| {
        let verify = verify.clone();
        async move {
            let mut scan = verify.scan();
            scan.filter(predicate).unwrap();
            let batch = scan.try_into_batch().await.unwrap();
            let mut values: Vec<i32> = batch["i"]
                .as_primitive::<Int32Type>()
                .iter()
                .map(|v| v.unwrap())
                .collect();
            values.sort_unstable();
            values
        }
    };
    assert_eq!(filtered("i = 1").await, vec![1]);
    assert_eq!(filtered("i = 9").await, vec![9]);
    assert_eq!(filtered("i >= 14").await, vec![14, 15]);
    let stored = crate::index::loading::load_all_indices(&verify)
        .await
        .unwrap();
    let scalar = stored.iter().find(|idx| idx.name == "i_idx").unwrap();
    assert_eq!(
        scalar.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1, 2, 3])
    );
}

/// Indexed and index-disabled results for `predicate` agree; the
/// indexed batch is returned.
async fn assert_indexed_agrees(dataset: &Dataset, predicate: &str) -> RecordBatch {
    let indexed = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .use_scalar_index(true)
        .try_into_batch()
        .await
        .unwrap();
    let scanned = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .use_scalar_index(false)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(indexed, scanned, "{predicate}");
    indexed
}

/// A rewrite of two groups: fragments 0 and 1 partitioned onto 10
/// and 11 with their transition recorded, fragments 2 and 3 moved
/// onto 20 and 21 with no transition (a bare group).
async fn mixed_rewrite(dataset: &Dataset) -> Operation {
    let old = |ids: &[u64]| -> Vec<Fragment> {
        dataset
            .fragments()
            .iter()
            .filter(|fragment| ids.contains(&fragment.id))
            .cloned()
            .collect()
    };
    let (transition, covered_destinations) = prepare_partition(dataset, &[0, 1], 10).await;
    let (_, bare_destinations) = prepare_partition(dataset, &[2, 3], 20).await;
    Operation::Rewrite {
        groups: vec![
            RewriteGroup {
                old_fragments: old(&[0, 1]),
                new_fragments: covered_destinations,
            },
            RewriteGroup {
                old_fragments: old(&[2, 3]),
                new_fragments: bare_destinations,
            },
        ],
        rewritten_indices: vec![],
        frag_reuse_index: Some(
            crate::index::frag_reuse::frag_reuse_entry_appending(dataset, vec![transition])
                .await
                .unwrap(),
        ),
    }
}

/// Four fragments of four rows, `i_idx` over all of them.
async fn covered_fixture() -> Dataset {
    let mut dataset = ram_fixture(4, 4).await;
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    reserve(&mut dataset, 40).await;
    dataset
}

/// Two fragments indexed by `i_idx`, then fragments 2 and 3 (values
/// 8..16) appended after the index was built: nothing covers them.
async fn uncovered_fixture() -> Dataset {
    let mut dataset = ram_fixture(2, 4).await;
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let dataset = append_rows(&dataset, 8..12).await;
    let mut dataset = append_rows(&dataset, 12..16).await;
    reserve(&mut dataset, 40).await;
    dataset
}

/// Attach a MemWAL index (no fragment bitmap by design).
async fn with_mem_wal(dataset: Dataset) -> Dataset {
    use lance_index::mem_wal::{MEM_WAL_INDEX_NAME, MemWalIndexDetails};
    let version = dataset.manifest.version;
    let mem_wal = lance_table::system_index::mem_wal::new_mem_wal_index_meta(
        version,
        MemWalIndexDetails::default(),
    )
    .unwrap();
    let dataset = commit_sp(
        &dataset,
        version,
        Operation::CreateIndex {
            new_indices: vec![mem_wal],
            removed_indices: vec![],
        },
    )
    .await
    .unwrap();
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    let entry = stored
        .iter()
        .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
        .unwrap();
    assert!(entry.fragment_bitmap.is_none());
    dataset
}

fn i_idx(indices: &[IndexMetadata]) -> &IndexMetadata {
    indices.iter().find(|idx| idx.name == "i_idx").unwrap()
}

/// The mixed rewrite on `dataset` is refused whole: nothing lands,
/// `i_idx` is as it was and still agrees with the scan.
async fn assert_mixed_rewrite_refused(mut dataset: Dataset) {
    let version = dataset.manifest.version;
    let stored_before = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(
        i_idx(&stored_before).fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1, 2, 3])
    );

    let operation = mixed_rewrite(&dataset).await;
    let error = commit_sp(&dataset, version, operation).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    assert!(error.to_string().contains("i_idx"), "{error}");

    assert_eq!(dataset.latest_version_id().await.unwrap(), version);
    dataset.checkout_latest().await.unwrap();
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(stored.len(), stored_before.len());
    assert_eq!(i_idx(&stored).uuid, i_idx(&stored_before).uuid);
    assert_eq!(
        i_idx(&stored).fragment_bitmap,
        i_idx(&stored_before).fragment_bitmap
    );
    assert_eq!(sorted_values(&dataset).await, (0..16).collect::<Vec<_>>());
    assert_eq!(
        assert_indexed_agrees(&dataset, "i = 13").await.num_rows(),
        1
    );
}

/// The mixed rewrite on `dataset` lands: the transition is recorded,
/// `i_idx` translates onto the covered destinations, the bare
/// destinations are scanned.
async fn assert_mixed_rewrite_lands(dataset: Dataset) {
    let version = dataset.manifest.version;
    let stored_before = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(
        i_idx(&stored_before).fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );

    let operation = mixed_rewrite(&dataset).await;
    let committed = commit_sp(&dataset, version, operation).await.unwrap();
    assert_eq!(
        committed
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![10, 11, 20, 21]
    );
    let (_, ledger) = fri_ledger(&committed).await;
    assert_eq!(ledger.transitions().len(), 1);
    let stored = crate::index::loading::load_all_indices(&committed)
        .await
        .unwrap();
    assert_eq!(stored.len(), stored_before.len() + 1, "the entry was added");
    assert_eq!(i_idx(&stored).uuid, i_idx(&stored_before).uuid);
    assert_eq!(
        i_idx(&stored).fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1]),
        "stored provenance stays the retired sources"
    );
    let derived = committed.load_indices().await.unwrap();
    assert_eq!(
        i_idx(&derived).fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11]),
        "derived coverage is the covered destinations"
    );
    assert_eq!(sorted_values(&committed).await, (0..16).collect::<Vec<_>>());
    assert_eq!(
        assert_indexed_agrees(&committed, "i = 5").await.num_rows(),
        1
    );
    assert_eq!(
        assert_indexed_agrees(&committed, "i = 13").await.num_rows(),
        1
    );
    assert_eq!(
        assert_indexed_agrees(&committed, "i >= 6").await.num_rows(),
        10
    );
}

/// A rewrite carrying a transition for one group and none for another
/// is validated group by group: the bare group must pass the rule of a
/// bare rewrite (its old fragments outside every index bitmap). Here
/// `i_idx` covers the bare group's sources, so the whole transaction
/// is refused and nothing partial is published: without this the
/// index silently loses coverage of the moved rows.
#[tokio::test]
async fn mixed_rewrite_with_a_bare_group_over_covered_fragments_is_refused_whole() {
    assert_mixed_rewrite_refused(covered_fixture().await).await;
}

/// The same shape where no index covers the bare group's sources
/// lands: the transition is recorded, the index translates onto the
/// covered destinations, the bare destinations are scanned.
#[tokio::test]
async fn mixed_rewrite_with_a_bare_group_over_uncovered_fragments_lands() {
    assert_mixed_rewrite_lands(uncovered_fixture().await).await;
}

/// A MemWAL index carries no fragment bitmap by design; it is not
/// unknown coverage, so it neither blocks the uncovered bare group
/// nor excuses the covered one.
#[tokio::test]
async fn mixed_rewrite_beside_a_mem_wal_index_follows_the_bare_group_rule() {
    assert_mixed_rewrite_lands(with_mem_wal(uncovered_fixture().await).await).await;
    assert_mixed_rewrite_refused(with_mem_wal(covered_fixture().await).await).await;
}

/// A second writer's index over `fragment_ids`, committed after
/// `version`.
async fn concurrent_index(dataset: &Dataset, fragment_ids: Vec<u32>) -> Dataset {
    let mut writer_b = dataset.clone();
    writer_b
        .create_index_builder(&["i"], IndexType::Scalar, &ScalarIndexParams::default())
        .name("late_idx".into())
        .fragments(fragment_ids)
        .await
        .unwrap();
    assert_eq!(writer_b.manifest.version, dataset.manifest.version + 1);
    writer_b
}

fn late_idx(indices: &[IndexMetadata]) -> &IndexMetadata {
    indices.iter().find(|idx| idx.name == "late_idx").unwrap()
}

/// The bare group is checked against the index list of the attempt,
/// not of the read version: an index a concurrent writer built over
/// the bare group's fragments refuses the rebased rewrite.
#[tokio::test]
async fn bare_group_covered_by_a_concurrent_index_is_refused_on_retry() {
    let mut writer_a = uncovered_fixture().await;
    let version = writer_a.manifest.version;
    let operation = mixed_rewrite(&writer_a).await;
    let writer_b = concurrent_index(&writer_a, vec![2, 3]).await;
    let stored_b = crate::index::loading::load_all_indices(&writer_b)
        .await
        .unwrap();
    assert_eq!(
        late_idx(&stored_b).fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([2u32, 3])
    );

    let error = commit_sp(&writer_a, version, operation).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    assert!(error.to_string().contains("late_idx"), "{error}");

    assert_eq!(
        writer_a.latest_version_id().await.unwrap(),
        writer_b.manifest.version,
        "nothing from writer A landed"
    );
    writer_a.checkout_latest().await.unwrap();
    assert_eq!(
        writer_a
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let stored = crate::index::loading::load_all_indices(&writer_a)
        .await
        .unwrap();
    assert_eq!(stored.len(), stored_b.len());
    assert_eq!(late_idx(&stored).uuid, late_idx(&stored_b).uuid);
    assert_eq!(
        late_idx(&stored).fragment_bitmap,
        late_idx(&stored_b).fragment_bitmap
    );
    assert_eq!(i_idx(&stored).uuid, i_idx(&stored_b).uuid);
    assert_eq!(sorted_values(&writer_a).await, (0..16).collect::<Vec<_>>());
    assert_eq!(
        assert_indexed_agrees(&writer_a, "i = 13").await.num_rows(),
        1
    );
    assert_eq!(
        assert_indexed_agrees(&writer_a, "i >= 6").await.num_rows(),
        10
    );
}

/// The mirror: the concurrent index covers only the transition
/// group's sources, so the rebased rewrite lands and that index
/// translates onto the covered destinations.
#[tokio::test]
async fn bare_group_uncovered_by_a_concurrent_index_lands_on_retry() {
    let writer_a = uncovered_fixture().await;
    let version = writer_a.manifest.version;
    let operation = mixed_rewrite(&writer_a).await;
    let writer_b = concurrent_index(&writer_a, vec![0, 1]).await;
    let stored_b = crate::index::loading::load_all_indices(&writer_b)
        .await
        .unwrap();

    let committed = commit_sp(&writer_a, version, operation).await.unwrap();
    assert_eq!(committed.manifest.version, writer_b.manifest.version + 1);
    assert_eq!(
        committed
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![10, 11, 20, 21]
    );
    let (_, ledger) = fri_ledger(&committed).await;
    assert_eq!(ledger.transitions().len(), 1);
    let stored = crate::index::loading::load_all_indices(&committed)
        .await
        .unwrap();
    assert_eq!(stored.len(), stored_b.len() + 1, "the entry was added");
    for (name, pick) in [
        (
            "late_idx",
            late_idx as fn(&[IndexMetadata]) -> &IndexMetadata,
        ),
        ("i_idx", i_idx),
    ] {
        assert_eq!(pick(&stored).uuid, pick(&stored_b).uuid, "{name}");
        assert_eq!(
            pick(&stored).fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1]),
            "{name}: stored provenance stays the retired sources"
        );
    }
    let derived = committed.load_indices().await.unwrap();
    assert_eq!(
        late_idx(&derived).fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11])
    );
    assert_eq!(sorted_values(&committed).await, (0..16).collect::<Vec<_>>());
    assert_eq!(
        assert_indexed_agrees(&committed, "i = 5").await.num_rows(),
        1
    );
    assert_eq!(
        assert_indexed_agrees(&committed, "i = 13").await.num_rows(),
        1
    );
    assert_eq!(
        assert_indexed_agrees(&committed, "i >= 6").await.num_rows(),
        10
    );
}

/// A tagged table with one stable-partition transition (fragments
/// 10 and 11) built through the real commit path.
async fn make_tagged(mut dataset: Dataset) -> Dataset {
    reserve(&mut dataset, 40).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&dataset).await;
    let version = dataset.manifest.version;
    commit_sp(
        &dataset,
        version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap()
}

async fn tagged_fixture() -> Dataset {
    make_tagged(ram_fixture(2, 4).await).await
}

/// An on-disk tagged fixture (fragments 10 and 11, four rows each,
/// values 0..8) plus two two-row appended fragments (100..102 and
/// 102..104): with `target_rows_per_fragment: 4` only the appended
/// fragments are compaction candidates, so a real `compact_files`
/// stays disjoint from a stable-partition rewrite of 10 and 11.
async fn disk_tagged_fixture_with_small_fragments(uri: &str) -> Dataset {
    // Order matters: the two-row fragments are appended and indexed
    // BEFORE the tagging rewrite, so deferred compaction of them
    // must record a transition (uncovered fragments would commit a
    // plain rewrite instead; see
    // `uncovered_deferred_compaction_commits_plain_rewrite`).
    let dataset = disk_fixture(uri, 2, 4).await;
    let dataset = append_rows(&dataset, 100..102).await;
    let mut dataset = append_rows(&dataset, 102..104).await;
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    reserve(&mut dataset, 40).await;
    let old_fragments: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|f| f.id < 2)
        .cloned()
        .collect();
    let (transition, destinations) = prepare_partition(&dataset, &[0, 1], 10).await;
    let version = dataset.manifest.version;
    let mut dataset = commit_sp(
        &dataset,
        version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();
    reserve(&mut dataset, 20).await;
    dataset
}

fn deferred_compaction_options() -> crate::dataset::optimize::CompactionOptions {
    crate::dataset::optimize::CompactionOptions {
        target_rows_per_fragment: 4,
        defer_index_remap: true,
        ..Default::default()
    }
}

async fn append_rows(dataset: &Dataset, values: std::ops::Range<i32>) -> Dataset {
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(values))]).unwrap();
    InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

/// The transition and destination a tagged deferred compaction of
/// `source_ids` would carry: the same digests and changed_row_addrs
/// the writer records, with an ordered-compaction mapping.
async fn oc_parts(
    dataset: &Dataset,
    source_ids: &[u64],
    dest_id: u64,
) -> (Transition, Vec<Fragment>) {
    let old_fragments: Vec<Fragment> = source_ids
        .iter()
        .map(|id| {
            dataset
                .fragments()
                .iter()
                .find(|f| f.id == *id)
                .unwrap()
                .clone()
        })
        .collect();
    let batch = {
        let mut scan = dataset.scan();
        scan.with_fragments(old_fragments.clone());
        scan.try_into_batch().await.unwrap()
    };
    let txn = InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![batch])
        .await
        .unwrap();
    let Operation::Append { mut fragments } = txn.operation else {
        unreachable!()
    };
    assert_eq!(fragments.len(), 1);
    fragments[0].id = dest_id;
    let mut changed_row_addrs = RoaringTreemap::new();
    for frag in &old_fragments {
        for offset in 0..frag.physical_rows.unwrap() as u64 {
            changed_row_addrs.insert((frag.id << 32) + offset);
        }
    }
    let mut serialized = Vec::new();
    changed_row_addrs.serialize_into(&mut serialized).unwrap();
    let digests = |fragments: &[Fragment]| {
        fragments
            .iter()
            .map(|f| FragmentDigest::from(&FragDigest::from(f)))
            .collect::<Vec<_>>()
    };
    let transition = Transition {
        sources: digests(&old_fragments),
        destinations: digests(&fragments),
        mapping: Some(transition::Mapping::OrderedCompaction(
            lance_table::format::pb::fragment_reuse_index_details::OrderedCompaction {
                changed_row_addrs: serialized,
            },
        )),
    };
    (transition, fragments)
}

/// Tagged-compaction row, direction 1: a stable-partition rewrite
/// lands while a tagged deferred compaction of disjoint fragments is
/// in flight; the compaction rebases onto the appended entry (its
/// transitions are ordered compactions, so the SP-vs-SP diff must
/// not fire) and both records survive.
#[tokio::test]
async fn tagged_compaction_rebases_over_concurrent_sp() {
    let dataset = tagged_fixture().await;
    let mut dataset = append_rows(&dataset, 100..104).await;
    reserve(&mut dataset, 20).await;
    let appended_id = dataset.fragments().last().unwrap().id;

    let b_old = vec![dataset.fragments().last().unwrap().clone()];
    let (b_transition, b_destinations) = oc_parts(&dataset, &[appended_id], 55).await;
    let read_version = dataset.manifest.version;

    let writer_a = dataset.clone();
    let a_old: Vec<Fragment> = writer_a
        .fragments()
        .iter()
        .filter(|f| f.id != appended_id)
        .cloned()
        .collect();
    let (a_transition, a_destinations) = prepare_partition(&writer_a, &[10, 11], 50).await;
    let a_version = writer_a.manifest.version;
    commit_sp(
        &writer_a,
        a_version,
        tagged_rewrite(&writer_a, a_old, a_destinations, vec![a_transition]).await,
    )
    .await
    .unwrap();

    let committed = commit_sp(
        &dataset,
        read_version,
        tagged_rewrite(&dataset, b_old, b_destinations, vec![b_transition]).await,
    )
    .await
    .unwrap();
    let (entry, ledger) = fri_ledger(&committed).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (2, 1));
    assert!(ledger.consumer(appended_id as u32).is_some());
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&committed).await, expected);
}

/// Tagged-compaction row, direction 2: a tagged deferred compaction
/// lands while a stable-partition rewrite of disjoint fragments is
/// in flight; the rewrite rebases onto the entry containing the
/// compaction's transition and both records survive.
#[tokio::test]
async fn sp_rebases_over_concurrent_tagged_compaction() {
    let dataset = tagged_fixture().await;
    let mut dataset = append_rows(&dataset, 100..104).await;
    reserve(&mut dataset, 20).await;
    let appended_id = dataset.fragments().last().unwrap().id;

    let b_old: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|f| f.id != appended_id)
        .cloned()
        .collect();
    let (b_transition, b_destinations) = prepare_partition(&dataset, &[10, 11], 50).await;
    let read_version = dataset.manifest.version;

    let writer_a = dataset.clone();
    let a_old = vec![writer_a.fragments().last().unwrap().clone()];
    let (a_transition, a_destinations) = oc_parts(&writer_a, &[appended_id], 55).await;
    let a_version = writer_a.manifest.version;
    commit_sp(
        &writer_a,
        a_version,
        tagged_rewrite(&writer_a, a_old, a_destinations, vec![a_transition]).await,
    )
    .await
    .unwrap();

    let committed = commit_sp(
        &dataset,
        read_version,
        tagged_rewrite(&dataset, b_old, b_destinations, vec![b_transition]).await,
    )
    .await
    .unwrap();
    let (entry, ledger) = fri_ledger(&committed).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (2, 1));
    assert!(ledger.consumer(10).is_some());
    assert!(ledger.consumer(appended_id as u32).is_some());
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&committed).await, expected);
}

/// A trim racing a concurrent stable-partition rewrite replaced
/// transition X with transition Y between the read version and the
/// current manifest (same count, different identity). Reassembly
/// merges onto the CURRENT entry: the finished rewrite carries Y
/// plus ours, and the trimmed-away X stays gone.
#[tokio::test]
async fn sp_merges_onto_swapped_transition_with_equal_count() {
    let mut dataset = ram_fixture(2, 4).await;
    reserve(&mut dataset, 40).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = prepare(&dataset).await;
    let version = dataset.manifest.version;
    let mut dataset = commit_sp(
        &dataset,
        version,
        tagged_rewrite(
            &dataset,
            old_fragments,
            destinations,
            vec![transition.clone()],
        )
        .await,
    )
    .await
    .unwrap();
    let read_version = dataset.manifest.version;

    // Swap the committed transition's row-map identity directly in
    // the manifest; trim is not implemented yet, so this simulates
    // its effect netting a zero count change.
    let mut swapped = transition;
    let Some(transition::Mapping::StablePartition(mapping)) = &mut swapped.mapping else {
        unreachable!()
    };
    mapping.map_id = Uuid::new_v4().to_string();
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    let indices: Vec<IndexMetadata> = stored
        .iter()
        .map(|idx| {
            if idx.name == FRAG_REUSE_INDEX_NAME {
                let mut replaced = idx.clone();
                replaced.uuid = Uuid::new_v4();
                replaced.index_details = Some(Arc::new(prost_types::Any {
                    type_url: "/lance.table.FragmentReuseIndexDetails".into(),
                    value: field(1, &field(2, &swapped.encode_to_vec())),
                }));
                replaced
            } else {
                idx.clone()
            }
        })
        .collect();
    persist_fixture(&mut dataset, indices).await;

    let read_dataset = dataset.checkout_version(read_version).await.unwrap();
    let read_ids = sp_map_ids(&read_dataset).await;
    let current_ids = sp_map_ids(&dataset).await;
    // Same count, different identity: the swap is real.
    assert_eq!(read_ids.len(), current_ids.len());
    assert_ne!(read_ids, current_ids);

    let b_old: Vec<Fragment> = read_dataset.fragments().iter().cloned().collect();
    let (b_transition, b_destinations) = prepare_partition(&read_dataset, &[10, 11], 30).await;
    let b_map_id = match b_transition.mapping.as_ref().unwrap() {
        transition::Mapping::StablePartition(partition) => partition.map_id.clone(),
        _ => unreachable!(),
    };
    // The intent is built against the read version (transition X
    // recorded); the current entry holds Y instead.
    let rebase = TransactionRebase::try_new(
        &read_dataset,
        Transaction::new(
            read_version,
            tagged_rewrite(&read_dataset, b_old, b_destinations, vec![b_transition]).await,
            None,
        ),
        None,
    )
    .await
    .unwrap();
    let (_, assembly) = rebase.finish_with_tagged_rewrite(&dataset).await.unwrap();
    let entry = &assembly.expect("expected an assembled entry").entry;
    let ledger = decode_frag_reuse_ledger(&dataset, entry).await.unwrap();
    let merged: std::collections::HashSet<String> = ledger
        .transitions()
        .iter()
        .filter_map(|transition| match transition.mapping() {
            Mapping::StablePartition(partition) => Some(partition.map_id.clone()),
            _ => None,
        })
        .collect();
    // The merge kept the concurrent transition (Y) and appended
    // ours; the trimmed-away X did not resurface.
    let mut expected = current_ids.clone();
    expected.insert(b_map_id);
    assert_eq!(merged, expected);
    assert!(read_ids.iter().all(|x| !merged.contains(x)));
}

/// FIX 3a, order 1: a real deferred `compact_files` planned before a
/// concurrent stable-partition rewrite landed (fresh sessions, on
/// disk) rebases onto the appended entry and both records survive.
#[tokio::test]
async fn real_deferred_compaction_rebases_over_concurrent_sp_fresh_session() {
    let dir = TempStrDir::default();
    disk_tagged_fixture_with_small_fragments(dir.as_str()).await;

    // The compactor opens BEFORE the rewrite commits: its plan and
    // its commit read-version anchor at the pre-rewrite snapshot.
    let mut compactor = fresh_session(dir.as_str()).await;

    let sp_writer = fresh_session(dir.as_str()).await;
    let b_old: Vec<Fragment> = sp_writer
        .fragments()
        .iter()
        .filter(|f| f.id == 10 || f.id == 11)
        .cloned()
        .collect();
    let (b_transition, b_destinations) = prepare_partition(&sp_writer, &[10, 11], 50).await;
    let read_version = sp_writer.manifest.version;
    commit_sp(
        &sp_writer,
        read_version,
        tagged_rewrite(&sp_writer, b_old, b_destinations, vec![b_transition]).await,
    )
    .await
    .unwrap();

    let metrics = crate::dataset::optimize::compact_files(
        &mut compactor,
        deferred_compaction_options(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);

    let verify = fresh_session(dir.as_str()).await;
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (2, 1));
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&verify).await, expected);
}

/// FIX 3a, order 2: a stable-partition rewrite whose read version
/// predates a real deferred `compact_files` commit (fresh sessions,
/// on disk) rebases onto the entry holding the compaction's
/// transition and both records survive.
#[tokio::test]
async fn sp_rebases_over_real_deferred_compaction_fresh_session() {
    let dir = TempStrDir::default();
    disk_tagged_fixture_with_small_fragments(dir.as_str()).await;

    let sp_writer = fresh_session(dir.as_str()).await;
    let b_old: Vec<Fragment> = sp_writer
        .fragments()
        .iter()
        .filter(|f| f.id == 10 || f.id == 11)
        .cloned()
        .collect();
    let (b_transition, b_destinations) = prepare_partition(&sp_writer, &[10, 11], 50).await;
    let read_version = sp_writer.manifest.version;

    let mut compactor = fresh_session(dir.as_str()).await;
    crate::dataset::optimize::compact_files(&mut compactor, deferred_compaction_options(), None)
        .await
        .unwrap();

    let committed = commit_sp(
        &sp_writer,
        read_version,
        tagged_rewrite(&sp_writer, b_old, b_destinations, vec![b_transition]).await,
    )
    .await
    .unwrap();

    let verify = fresh_session(dir.as_str()).await;
    assert_eq!(verify.manifest.version, committed.manifest.version);
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (2, 1));
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&verify).await, expected);
}

/// FIX 3b, disjoint: a real deferred `compact_files` and a
/// tagged-compaction rewrite of other fragments both land; the entry
/// holds both ordered-compaction transitions.
#[tokio::test]
async fn tagged_compactions_disjoint_both_land_fresh_session() {
    let dir = TempStrDir::default();
    disk_tagged_fixture_with_small_fragments(dir.as_str()).await;

    let oc_writer = fresh_session(dir.as_str()).await;
    let b_old: Vec<Fragment> = oc_writer
        .fragments()
        .iter()
        .filter(|f| f.id == 10 || f.id == 11)
        .cloned()
        .collect();
    let (b_transition, b_destinations) = oc_parts(&oc_writer, &[10, 11], 55).await;
    let read_version = oc_writer.manifest.version;

    let mut compactor = fresh_session(dir.as_str()).await;
    crate::dataset::optimize::compact_files(&mut compactor, deferred_compaction_options(), None)
        .await
        .unwrap();

    let committed = commit_sp(
        &oc_writer,
        read_version,
        tagged_rewrite(&oc_writer, b_old, b_destinations, vec![b_transition]).await,
    )
    .await
    .unwrap();

    let verify = fresh_session(dir.as_str()).await;
    assert_eq!(verify.manifest.version, committed.manifest.version);
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(count_mappings(&ledger), (1, 2));
    assert!(ledger.consumer(10).is_some());
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    assert_eq!(sorted_values(&verify).await, expected);
}

/// FIX 3b, overlapping: two real deferred `compact_files` racing over
/// the same candidates; the loser is rejected outright through the
/// double-consumption rule (its plan can never succeed as-is).
#[tokio::test]
async fn tagged_compactions_overlapping_rejected_fresh_session() {
    let dir = TempStrDir::default();
    disk_tagged_fixture_with_small_fragments(dir.as_str()).await;

    let mut loser = fresh_session(dir.as_str()).await;
    let mut winner = fresh_session(dir.as_str()).await;
    crate::dataset::optimize::compact_files(&mut winner, deferred_compaction_options(), None)
        .await
        .unwrap();

    let error =
        crate::dataset::optimize::compact_files(&mut loser, deferred_compaction_options(), None)
            .await
            .unwrap_err();
    assert!(
        matches!(error, Error::IncompatibleTransaction { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("consumed fragments"), "{error}");
}

/// FIX 4a: a CreateIndex mixing an FRI replacement with user indices
/// is a shape the resolver does not reason about; a stable-partition
/// rewrite refuses it as incompatible, same as the v0 arm.
#[tokio::test]
async fn sp_rejects_mixed_fri_create_index() {
    let dataset = test_dataset(8, 2).await;
    let index_meta = |name: &str| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let ours = Transaction::new(
        1,
        tagged_rewrite(
            &dataset,
            vec![dataset.fragments()[0].clone()],
            vec![Fragment::new(10)],
            vec![Transition {
                sources: vec![FragmentDigest {
                    id: 0,
                    physical_rows: 4,
                    num_deleted_rows: 0,
                }],
                destinations: vec![FragmentDigest {
                    id: 10,
                    physical_rows: 4,
                    num_deleted_rows: 0,
                }],
                mapping: Some(transition::Mapping::StablePartition(StablePartition {
                    map_id: Uuid::new_v4().to_string(),
                    map_size_bytes: 1,
                    base_id: None,
                })),
            }],
        )
        .await,
        None,
    );
    let mut rebase = TransactionRebase::try_new(&dataset, ours, None)
        .await
        .unwrap();
    let theirs = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![index_meta(FRAG_REUSE_INDEX_NAME), index_meta("u_idx")],
            removed_indices: vec![index_meta(FRAG_REUSE_INDEX_NAME)],
        },
        None,
    );
    let error = rebase.check_txn(&theirs, 2).unwrap_err();
    assert!(error.to_string().contains("incompatible"), "{error}");
}

/// ITEM 6: a real deferred `compact_files` and a fragment-reuse
/// rewrite whose sources are exactly the compaction candidates. The
/// compaction lands first; the rewrite is rejected outright (its
/// transitions and row map reference consumed fragments and can
/// never commit as-is).
#[tokio::test]
async fn overlapping_compaction_and_sp_rejected_fresh_session() {
    let dir = TempStrDir::default();
    disk_tagged_fixture_with_small_fragments(dir.as_str()).await;

    let sp_writer = fresh_session(dir.as_str()).await;
    let small_ids: Vec<u64> = sp_writer
        .fragments()
        .iter()
        .filter(|f| f.physical_rows.unwrap() < 4)
        .map(|f| f.id)
        .collect();
    assert_eq!(small_ids.len(), 2);
    let b_old: Vec<Fragment> = sp_writer
        .fragments()
        .iter()
        .filter(|f| small_ids.contains(&f.id))
        .cloned()
        .collect();
    let (b_transition, b_destinations) = prepare_partition(&sp_writer, &small_ids, 60).await;
    let read_version = sp_writer.manifest.version;

    let mut compactor = fresh_session(dir.as_str()).await;
    crate::dataset::optimize::compact_files(&mut compactor, deferred_compaction_options(), None)
        .await
        .unwrap();

    let error = commit_sp(
        &sp_writer,
        read_version,
        tagged_rewrite(&sp_writer, b_old, b_destinations, vec![b_transition]).await,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, Error::IncompatibleTransaction { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("consumed fragments"), "{error}");
}

/// Three disjoint writers race: a stable-partition rewrite, a real
/// deferred compaction and another stable-partition rewrite, the
/// latter two reading a version predating the first. All three land
/// through in-loop reassembly and the ledger holds all three
/// records.
#[tokio::test]
async fn sp_compaction_sp_three_way_disjoint_merge() {
    let dir = TempStrDir::default();
    let dataset = disk_tagged_fixture_with_small_fragments(dir.as_str()).await;
    // A fourth region for the second stable-partition writer.
    let mut dataset = append_rows(&dataset, 200..204).await;
    reserve(&mut dataset, 100).await;
    let extra_id = dataset.fragments().last().unwrap().id;

    // Both racing writers open before the first rewrite commits.
    let mut compactor = fresh_session(dir.as_str()).await;
    let sp_writer = fresh_session(dir.as_str()).await;
    let w3_old = vec![
        sp_writer
            .fragments()
            .iter()
            .find(|f| f.id == extra_id)
            .unwrap()
            .clone(),
    ];
    let (w3_transition, w3_destinations) = prepare_partition(&sp_writer, &[extra_id], 90).await;
    let w3_read_version = sp_writer.manifest.version;

    // Writer 1: stable-partition rewrite of fragments 10 and 11.
    let w1_old: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|f| f.id == 10 || f.id == 11)
        .cloned()
        .collect();
    let (w1_transition, w1_destinations) = prepare_partition(&dataset, &[10, 11], 80).await;
    let w1_version = dataset.manifest.version;
    commit_sp(
        &dataset,
        w1_version,
        tagged_rewrite(&dataset, w1_old, w1_destinations, vec![w1_transition]).await,
    )
    .await
    .unwrap();

    // Writer 2: real deferred compaction of the indexed two-row
    // fragments (2 and 3), planned before writer 1 landed.
    crate::dataset::optimize::compact_files(&mut compactor, deferred_compaction_options(), None)
        .await
        .unwrap();

    // Writer 3: stable-partition rewrite of the appended region,
    // rebasing over BOTH concurrent commits.
    commit_sp(
        &sp_writer,
        w3_read_version,
        tagged_rewrite(&sp_writer, w3_old, w3_destinations, vec![w3_transition]).await,
    )
    .await
    .unwrap();

    let verify = fresh_session(dir.as_str()).await;
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    // Fixture rewrite + writers 1 and 3, plus the compaction.
    assert_eq!(count_mappings(&ledger), (3, 1));
    assert!(ledger.consumer(10).is_some());
    assert!(ledger.consumer(2).is_some());
    assert!(ledger.consumer(extra_id as u32).is_some());
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    expected.extend(200..204);
    assert_eq!(sorted_values(&verify).await, expected);
}

/// A v0 table with one committed legacy reuse version and two
/// indexed two-row fragments left as compaction candidates (the
/// four-row fragments are not candidates at target 4).
async fn disk_v0_fixture_with_history_and_small_fragments(uri: &str) -> Dataset {
    let dataset = disk_fixture(uri, 2, 4).await;
    let dataset = append_rows(&dataset, 100..102).await;
    let mut dataset = append_rows(&dataset, 102..104).await;
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    // Real v0 deferred compaction of the first pair: the legacy
    // history the later lift must preserve.
    crate::dataset::optimize::compact_files(&mut dataset, deferred_compaction_options(), None)
        .await
        .unwrap();
    let (entry, ledger) = fri_ledger(&dataset).await;
    assert_eq!(entry.index_version, 0);
    assert_eq!(count_mappings(&ledger), (0, 1));
    // Fresh candidates, re-indexed (replace) so a deferred
    // compaction of them must record reuse.
    let dataset = append_rows(&dataset, 200..202).await;
    let mut dataset = append_rows(&dataset, 202..204).await;
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    reserve(&mut dataset, 60).await;
    dataset
}

fn small_fragment_ids(dataset: &Dataset) -> Vec<u64> {
    dataset
        .fragments()
        .iter()
        .filter(|f| f.physical_rows.unwrap() < 4)
        .map(|f| f.id)
        .collect()
}

fn expected_fixture_values() -> Vec<i32> {
    let mut expected: Vec<i32> = (0..8).collect();
    expected.extend(100..104);
    expected.extend(200..204);
    expected
}

/// Round 9 (1): a deferred compaction shaped for the v0 table loses
/// the race to a concurrent stable-partition lift. The retry
/// converts its legacy version into an ordered-compaction transition
/// and commits with no caller-visible error; the lifted legacy
/// history is preserved and the compaction lands as a transition,
/// not a legacy version.
#[tokio::test]
async fn v0_compaction_converts_over_concurrent_sp_lift() {
    let dir = TempStrDir::default();
    // Boxed for CI clippy `large_futures`: the fixture future grew past 16 KiB.
    let main = Box::pin(disk_v0_fixture_with_history_and_small_fragments(
        dir.as_str(),
    ))
    .await;

    // The compactor opens while the table is still v0.
    let mut compactor = fresh_session(dir.as_str()).await;
    let small_ids = small_fragment_ids(&compactor);
    assert_eq!(small_ids.len(), 2);

    // Concurrent stable-partition lift of the four-row originals.
    let sp_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|f| f.id < 2)
        .cloned()
        .collect();
    let (transition, destinations) = prepare_partition(&main, &[0, 1], 30).await;
    let version = main.manifest.version;
    commit_sp(
        &main,
        version,
        tagged_rewrite(&main, sp_old, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    // The stale writer still decides the v0 shape and must convert
    // during the in-loop rebase.
    let metrics = crate::dataset::optimize::compact_files(
        &mut compactor,
        deferred_compaction_options(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);

    let verify = fresh_session(dir.as_str()).await;
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    // Lifted legacy compaction + converted compaction + the lift.
    assert_eq!(ledger.transitions().len(), 3);
    assert_eq!(count_mappings(&ledger), (1, 2));
    for id in small_ids {
        assert!(ledger.consumer(id as u32).is_some());
    }
    assert_eq!(sorted_values(&verify).await, expected_fixture_values());
}

/// Round 9 (2): the same conversion when the commit that tagged the
/// table was itself a tagged compaction (an ordered-compaction
/// rewrite lifting the entry), not a stable-partition rewrite.
#[tokio::test]
async fn v0_compaction_converts_over_concurrent_tagged_compaction() {
    let dir = TempStrDir::default();
    // Boxed for CI clippy `large_futures`: the fixture future grew past 16 KiB.
    let main = Box::pin(disk_v0_fixture_with_history_and_small_fragments(
        dir.as_str(),
    ))
    .await;

    let mut compactor = fresh_session(dir.as_str()).await;
    let small_ids = small_fragment_ids(&compactor);

    // The intervening tagged compaction consumes the four-row
    // originals through an ordered-compaction transition.
    let oc_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|f| f.id < 2)
        .cloned()
        .collect();
    let (oc_transition, oc_destinations) = oc_parts(&main, &[0, 1], 40).await;
    let version = main.manifest.version;
    commit_sp(
        &main,
        version,
        tagged_rewrite(&main, oc_old, oc_destinations, vec![oc_transition]).await,
    )
    .await
    .unwrap();

    crate::dataset::optimize::compact_files(&mut compactor, deferred_compaction_options(), None)
        .await
        .unwrap();

    let verify = fresh_session(dir.as_str()).await;
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 1);
    assert_eq!(ledger.transitions().len(), 3);
    assert_eq!(count_mappings(&ledger), (0, 3));
    for id in small_ids {
        assert!(ledger.consumer(id as u32).is_some());
    }
    assert_eq!(sorted_values(&verify).await, expected_fixture_values());
}

/// Round 9 (3): the concurrent lift consumed the compaction's own
/// candidates; the overlap conflicts before any conversion. The
/// class is the v0-legacy retryable conflict: at check time the
/// resolver has no dataset state and both committed shapes are
/// plain rewrites in their transaction files, so a v0-shaped loser
/// cannot be classified differently from a pure v0 race without
/// breaking v0 invariance (see the v0 x v0 guard below).
#[tokio::test]
async fn v0_compaction_overlap_with_sp_still_conflicts() {
    let dir = TempStrDir::default();
    // Boxed for CI clippy `large_futures`: the fixture future grew past 16 KiB.
    let main = Box::pin(disk_v0_fixture_with_history_and_small_fragments(
        dir.as_str(),
    ))
    .await;

    let mut compactor = fresh_session(dir.as_str()).await;
    let small_ids = small_fragment_ids(&compactor);

    // The lift rewrites exactly the compaction's candidates.
    let sp_old: Vec<Fragment> = main
        .fragments()
        .iter()
        .filter(|f| small_ids.contains(&f.id))
        .cloned()
        .collect();
    let (transition, destinations) = prepare_partition(&main, &small_ids, 30).await;
    let version = main.manifest.version;
    commit_sp(
        &main,
        version,
        tagged_rewrite(&main, sp_old, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    let error = crate::dataset::optimize::compact_files(
        &mut compactor,
        deferred_compaction_options(),
        None,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
}

/// Round 9 (4): a pure v0 x v0 compaction race keeps legacy
/// behavior exactly -- the loser fails with the legacy retryable
/// conflict, no conversion runs, and the entry stays v0.
#[tokio::test]
async fn v0_compaction_race_keeps_legacy_behavior() {
    let dir = TempStrDir::default();
    // Boxed for CI clippy `large_futures`: the fixture future grew past 16 KiB.
    Box::pin(disk_v0_fixture_with_history_and_small_fragments(
        dir.as_str(),
    ))
    .await;

    let mut loser = fresh_session(dir.as_str()).await;
    let mut winner = fresh_session(dir.as_str()).await;
    crate::dataset::optimize::compact_files(&mut winner, deferred_compaction_options(), None)
        .await
        .unwrap();

    let error =
        crate::dataset::optimize::compact_files(&mut loser, deferred_compaction_options(), None)
            .await
            .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );

    let verify = fresh_session(dir.as_str()).await;
    let (entry, ledger) = fri_ledger(&verify).await;
    assert_eq!(entry.index_version, 0);
    assert_eq!(count_mappings(&ledger), (0, 2));
}
