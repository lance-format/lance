// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
    thread::ThreadId,
};

use lance_core::utils::{allocation::Reservation, tokio::spawn_cpu};

use super::*;
use crate::{
    compression::{PerValueDecodeBudget, VariablePerValueDecompressor},
    data::VariableWidthBlock,
    encodings::logical::primitive::fullzip_decode::PerValueDecompressor,
};

#[derive(Debug)]
struct ObservedCodec {
    inner: Arc<dyn VariablePerValueDecompressor>,
    caller: ThreadId,
    calls: Arc<AtomicUsize>,
    outcome: &'static str,
}

impl VariablePerValueDecompressor for ObservedCodec {
    fn decompress(&self, _: VariableWidthBlock) -> Result<DataBlock> {
        panic!("ordinary decode must not be called")
    }
    fn visit_binary_values(
        &self,
        data: &VariableWidthBlock,
        visitor: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        self.inner.visit_binary_values(data, visitor)
    }
    fn plan_decode(&self, data: &VariableWidthBlock) -> Result<PerValueDecodeBudget> {
        self.inner.plan_decode(data)
    }
    fn try_decompress_in(
        &self,
        data: &VariableWidthBlock,
        work: &mut Reservation,
    ) -> Result<Option<DataBlock>> {
        assert_ne!(std::thread::current().id(), self.caller);
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.outcome {
            "refuse" => Ok(None),
            "error" => Err(Error::io("injected CPU codec error")),
            _ => self.inner.try_decompress_in(data, work),
        }
    }
}

fn observe(f: &mut Fixture, outcome: &'static str) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    let details = Arc::get_mut(&mut f.scheduler.details).unwrap();
    let PerValueDecompressor::Variable(inner) = &details.value_decompressor else {
        panic!("variable fixture required")
    };
    details.value_decompressor = PerValueDecompressor::Variable(Arc::new(ObservedCodec {
        inner: inner.clone(),
        caller: std::thread::current().id(),
        calls: calls.clone(),
        outcome,
    }));
    calls
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn array_cpu_dispatch_preserves_selected_values(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
    #[values("flat", "nullable")] kind: &str,
) {
    let mut f = fixture(bits, fsst, kind);
    let source = source(&mut f);
    let calls = observe(&mut f, "complete");
    let selected = f.values[1..4]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if kind == "nullable" && i == 0 {
                None
            } else {
                Some(v.as_slice())
            }
        })
        .collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from(selected).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from(selected).to_data(),
        )
    };
    let budget = AllocationBudget::new(262144);
    let reads = Arc::new(Reads::default());
    let caller = std::thread::current().id();
    let mut read = reader(source, reads.clone(), Fault::None);
    let (work, array) = Arc::new(f.scheduler)
        .try_read_array_range_on_cpu(
            1..4,
            ty,
            budget.try_reserve(262144).unwrap(),
            move |range, output| {
                assert_eq!(std::thread::current().id(), caller, "I/O stays async");
                read(range, output)
            },
        )
        .await
        .unwrap();
    let array = array.unwrap();
    assert_eq!(array.to_data(), expected);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(reads.0.lock().unwrap().len(), 3);
    let alias = array.slice(1, 1);
    drop((array, work));
    let retained = budget.reserved();
    assert!(retained > 0);
    let peer = budget.try_reserve(262144 - retained).unwrap();
    assert!(budget.try_reserve(1).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), peer.len());
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn queued_cpu_job_owns_credit_after_caller_cancellation(
    #[values("complete", "drop", "refuse", "error", "undelivered")] outcome: &'static str,
) {
    let mut f = fixture(32, true, "flat");
    let source = source(&mut f);
    let calls = observe(&mut f, outcome);
    let budget = AllocationBudget::new(262144);
    let queued = RefCell::new(None);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    // A controllable dispatch queue tests cancellation at the exact ownership
    // boundary. No worker is blocked waiting for the test, even with one CPU.
    let mut future = Box::pin(Arc::new(f.scheduler).read_array_range_with(
        0..4,
        DataType::Binary,
        budget.try_reserve(262144).unwrap(),
        reader(source, Arc::default(), Fault::None),
        |job| {
            *queued.borrow_mut() = Some(job);
            async move { receiver.await.unwrap() }
        },
    ));
    assert!(poll!(future.as_mut()).is_pending());
    let job = queued.borrow_mut().take().unwrap();
    assert_eq!(budget.reserved(), 262144);
    assert!(budget.try_reserve(1).is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    if outcome == "undelivered" {
        let result = spawn_cpu(move || job.decode()).await;
        sender.send(result).unwrap();
        assert!(budget.reserved() > 0, "unpolled output remains owned");
        drop(future);
    } else {
        drop(future);
        assert_eq!(
            budget.reserved(),
            262144,
            "queued job keeps its full workspace"
        );
        assert!(budget.try_reserve(1).is_none());
        if outcome == "drop" {
            drop(job);
        } else {
            let result = spawn_cpu(move || job.decode()).await;
            match outcome {
                "error" => assert!(result.unwrap_err().to_string().contains("CPU codec error")),
                "refuse" => assert!(result.unwrap().1.is_none()),
                _ => {
                    let (work, array) = result.unwrap();
                    assert_eq!(array.as_ref().unwrap().len(), 4);
                    drop((work, array));
                }
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }
    assert_eq!(budget.reserved(), 0);
    let peer = budget.try_reserve(262144).unwrap();
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}
