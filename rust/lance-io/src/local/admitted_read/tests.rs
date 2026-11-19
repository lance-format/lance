// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::ErrorKind;
use std::os::unix::fs::FileExt;
use std::sync::mpsc;
use std::time::Duration;

use futures::poll;
use lance_core::utils::allocation::AllocationBudget;
use object_store::path::Path;
use rstest::rstest;
use tokio::sync::oneshot;
use tokio::time::timeout;

use super::*;
use crate::local::LocalObjectReader;
use crate::traits::Reader;

const DEADLINE: Duration = Duration::from_secs(2);

fn cold_reader(path: &std::path::Path) -> LocalObjectReader {
    LocalObjectReader {
        file: Arc::new(File::open(path).unwrap()),
        admitted_read: ReadGate::default(),
        path: Path::from_filesystem_path(path).unwrap(),
        size: OnceCell::new(),
        block_size: 4096,
        io_tracker: Default::default(),
    }
}

async fn reader(path: &std::path::Path) -> LocalObjectReader {
    let reader = cold_reader(path);
    // Complete lazy identity lookup so polling below observes the read queue,
    // rather than merely waiting for the first metadata dispatch.
    reader.admitted_read.get(reader.file.clone()).await.unwrap();
    reader
}

#[tokio::test]
async fn ordinary_and_rejected_reads_do_not_initialize_gate() {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"abcd").unwrap();
    let reader = cold_reader(file.path());
    assert_eq!(reader.get_range(0..4).await.unwrap(), b"abcd"[..]);
    assert!(reader.admitted_read.gate.get().is_none());
    let budget = AllocationBudget::new(4);
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let error = reader.get_range_into(0..3, output).await.unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("read range does not match reserved payload")
    );
    assert_eq!(budget.reserved(), 0);
    assert!(reader.admitted_read.gate.get().is_none());
    let outputs = reader.get_ranges_into(Vec::new()).await.unwrap();
    assert!(outputs.is_empty());
    assert!(reader.admitted_read.gate.get().is_none());
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let output = reader.get_range_into(0..4, output).await.unwrap();
    assert_eq!(output.as_ref(), b"abcd");
    assert!(reader.admitted_read.gate.get().is_some());
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::single(1)]
#[case::batch(2)]
#[tokio::test]
async fn detached_read_keeps_inode_gate_and_credit_but_other_files_progress(#[case] count: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data");
    let link = dir.path().join("hard-link");
    let other = dir.path().join("other");
    std::fs::write(&path, b"abcdefgh").unwrap();
    std::fs::hard_link(&path, &link).unwrap();
    std::fs::write(&other, b"12345678").unwrap();
    let first = reader(&path).await;
    let same = reader(&path).await;
    let linked = reader(&link).await;
    let different = reader(&other).await;
    let held_bytes = count * 4;
    let budget = AllocationBudget::new(held_bytes + 12);
    let held_outputs = (0..count)
        .map(|_| budget.try_reserve(4).unwrap().allocate().unwrap())
        .collect::<Vec<_>>();
    let (started_tx, started_rx) = oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut held = Box::pin(first.admitted_read.run(first.file.clone(), move |file| {
        started_tx.send(()).unwrap();
        release_rx.recv_timeout(DEADLINE).unwrap();
        for mut output in held_outputs {
            file.read_exact_at(output.as_mut(), 0)?;
            assert_eq!(output.as_ref(), b"abcd");
        }
        Ok(())
    }));
    assert!(poll!(&mut held).is_pending());
    timeout(DEADLINE, started_rx).await.unwrap().unwrap();
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let mut queued = same.get_range_into(0..4, output);
    assert!(poll!(&mut queued).is_pending());
    assert_eq!(budget.reserved(), held_bytes + 4);
    drop(queued);
    assert_eq!(budget.reserved(), held_bytes);

    let left = budget.try_reserve(4).unwrap().allocate().unwrap();
    let right = budget.try_reserve(4).unwrap().allocate().unwrap();
    let mut batch = linked.get_ranges_into(vec![(4..8, left), (0..4, right)]);
    assert!(poll!(&mut batch).is_pending());
    // Dropping the waiter must not admit another same-inode read while the
    // detached blocking closure still owns and uses the first allocation.
    drop(held);
    assert!(poll!(&mut batch).is_pending());
    assert_eq!(budget.reserved(), held_bytes + 8);

    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let output = timeout(DEADLINE, different.get_range_into(2..6, output))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.as_ref(), b"3456");
    drop(output);
    // The ordinary scanner's local read API never waits on the owned-read gate.
    assert_eq!(
        timeout(DEADLINE, same.get_range(1..5))
            .await
            .unwrap()
            .unwrap(),
        b"bcde"[..]
    );
    release_tx.send(()).unwrap();
    let outputs = timeout(DEADLINE, batch).await.unwrap().unwrap();
    assert_eq!(outputs[0].as_ref(), b"efgh");
    assert_eq!(outputs[1].as_ref(), b"abcd");
    assert_eq!(budget.reserved(), 8);
    drop(outputs);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn replaced_path_uses_new_inode_while_open_reader_keeps_old_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data");
    let replacement = dir.path().join("replacement");
    std::fs::write(&path, b"old!").unwrap();
    let old = reader(&path).await;
    let old_gate = old.admitted_read.get(old.file.clone()).await.unwrap();
    let guard = old_gate.lock().await;
    std::fs::write(&replacement, b"new!").unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    let new = reader(&path).await;
    let budget = AllocationBudget::new(4);
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let output = timeout(DEADLINE, new.get_range_into(0..4, output))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.as_ref(), b"new!");
    drop((output, guard));
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let output = timeout(DEADLINE, old.get_range_into(0..4, output))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.as_ref(), b"old!");
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::io_error(false)]
#[case::panic(true)]
#[tokio::test]
async fn failed_blocking_operation_releases_gate_and_buffer(#[case] panic: bool) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"abcd").unwrap();
    let reader = reader(file.path()).await;
    let budget = AllocationBudget::new(4);
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let error = reader
        .admitted_read
        .run(reader.file.clone(), move |_| {
            let _output = output;
            if panic {
                panic!("injected owned read panic");
            }
            Err::<(), _>(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                "injected read EOF",
            ))
        })
        .await
        .unwrap_err();
    if panic {
        assert!(matches!(error, object_store::Error::JoinError { .. }));
        assert!(error.to_string().contains("injected owned read panic"));
    } else {
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(error.to_string().contains("injected read EOF"));
    }
    assert_eq!(budget.reserved(), 0);
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let output = timeout(DEADLINE, reader.get_range_into(0..4, output))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.as_ref(), b"abcd");
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn registry_reclaims_closed_readers_without_splitting_live_identity() {
    let mut registry = Registry::default();
    let live = registry.get((1, 1));
    for inode in 2..100 {
        drop(registry.get((1, inode)));
    }
    let again = registry.get((1, 1));
    assert!(Arc::ptr_eq(&live, &again));
    assert_eq!(registry.gates.len(), 1);
    drop((live, again));
    let next = registry.get((2, 1));
    assert_eq!(registry.gates.len(), 1);
    assert_eq!(Arc::strong_count(&next), 1);
}
