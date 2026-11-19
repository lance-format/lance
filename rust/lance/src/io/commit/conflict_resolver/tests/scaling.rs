// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How the conflict check's cost grows with the table, rather than with what
//! the transactions touch. Ignored: it reports timings, it does not assert
//! them. Run with
//!
//! ```text
//! cargo test -p lance --release --lib conflict_resolver::tests::scaling -- --ignored --nocapture
//! ```

use std::time::Instant;

use lance_table::format::{DeletionFile, DeletionFileType};
use lance_table::transaction::action::{
    Action as TxnAction, AddFragment, AddFragmentParts, AddIndexSegment, AddIndexSegmentParts,
    CompositeOperation, FieldRef, Ref as ActionRef, SetDeletionFile, UserStep,
};

use super::*;

const CONCURRENT_COMMITS: u64 = 10;

fn manifest_with_fragments(read: &Manifest, fragments: Vec<Fragment>, version: u64) -> Manifest {
    let mut manifest = read.clone();
    manifest.max_fragment_id = fragments.iter().map(|f| f.id as u32).max();
    manifest.fragments = Arc::new(fragments);
    manifest.version = version;
    manifest
}

fn action_txn(read_version: u64, actions: Vec<TxnAction>) -> Transaction {
    Transaction::new_from_version(
        read_version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "step", actions,
        )])),
    )
}

fn action_append(read_version: u64) -> Transaction {
    action_txn(
        read_version,
        vec![TxnAction::AddFragment(AddFragment::new_unchecked(
            AddFragmentParts {
                id: ActionRef::Local(0),
                physical_rows: 1,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            },
        ))],
    )
}

/// What the scenario commits on top of the concurrent appends.
#[derive(Clone, Copy)]
enum Pending {
    /// A delete of one row in fragment 5.
    Delete,
    /// An index segment covering every fragment.
    IndexEveryFragment,
}

/// A pending action-set delete of one row in fragment 5 against
/// `CONCURRENT_COMMITS` committed appends, alternating legacy and action-set
/// form, on a table of `fragments` fragments. Returns the seconds spent
/// setting up the rebase and checking every committed transaction.
async fn time_check(fragments: u64, pending: Pending) -> f64 {
    let dataset = test_dataset(10, 1).await;
    let template = dataset.fragments()[0].clone();
    let fragment = |id: u64| {
        let mut fragment = template.clone();
        fragment.id = id;
        fragment
    };
    let read_version = dataset.manifest.version;
    let read = manifest_with_fragments(
        &dataset.manifest,
        (0..fragments).map(fragment).collect(),
        read_version,
    );
    let mut history = vec![Arc::new(read.clone())];
    let mut committed = Vec::new();
    for i in 1..=CONCURRENT_COMMITS {
        let mut next = history.last().unwrap().fragments.as_ref().clone();
        let added = fragment(fragments + i);
        next.push(added.clone());
        history.push(Arc::new(manifest_with_fragments(
            &read,
            next,
            read_version + i,
        )));
        let transaction = if i % 2 == 0 {
            action_append(read_version + i - 1)
        } else {
            Transaction::new_from_version(
                read_version + i - 1,
                Operation::Append {
                    fragments: vec![added],
                },
            )
        };
        committed.push((read_version + i, transaction));
    }

    let mut dataset = dataset.clone();
    dataset.manifest = Arc::new(read);
    let delete = || {
        vec![TxnAction::SetDeletionFile(
            SetDeletionFile::try_new_unchecked(
                5,
                DeletionFile {
                    read_version,
                    id: 1,
                    file_type: DeletionFileType::Array,
                    num_deleted_rows: Some(1),
                    base_id: None,
                    inline: None,
                },
                None,
                true,
            )
            .unwrap(),
        )]
    };
    let index = || {
        vec![TxnAction::AddIndexSegment(
            AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
                uuid: Uuid::new_v4(),
                name: "by_a".into(),
                fields: vec![FieldRef::Committed(0)],
                covering_fields: Vec::new(),
                index_details: None,
                index_version: 1,
                covered_fragments: Some((0..fragments).map(ActionRef::Committed).collect()),
                files: Vec::new(),
                base: None,
                created_at: None,
                dataset_version: None,
                data_change: false,
            })
            .unwrap(),
        )]
    };
    let pending = action_txn(
        read_version,
        match pending {
            Pending::Delete => delete(),
            Pending::IndexEveryFragment => index(),
        },
    );

    let start = Instant::now();
    let mut rebase = TransactionRebase::try_new(&dataset, pending, None)
        .await
        .unwrap();
    for (index, (version, transaction)) in committed.iter().enumerate() {
        if rebase.needs_committed_manifests(transaction) {
            rebase.record_committed(*version, transaction, &history[index]);
        }
    }
    if let (Some(since), Some(latest)) = (history.first(), history.last()) {
        rebase.record_window(since.clone(), latest.clone());
    }
    let committed: Vec<(u64, &Transaction)> = committed
        .iter()
        .map(|(version, transaction)| (*version, transaction))
        .collect();
    rebase.check_txns(&committed).unwrap();
    start.elapsed().as_secs_f64()
}

#[tokio::test]
#[ignore = "reports timings; run by hand"]
#[allow(clippy::print_stdout)]
async fn conflict_check_cost_by_table_size() {
    for (label, pending) in [
        ("delete", Pending::Delete),
        ("index_every_fragment", Pending::IndexEveryFragment),
    ] {
        for fragments in [1_000, 10_000, 100_000] {
            let seconds = time_check(fragments, pending).await;
            println!(
                "SCALING pending={label} fragments={fragments} \
                 concurrent_commits={CONCURRENT_COMMITS} check_seconds={seconds:.4}"
            );
        }
    }
}
