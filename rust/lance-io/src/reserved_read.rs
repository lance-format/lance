// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in reads into admitted application-owned payloads. Transport/Reader buffers
//! are outside this budget: even a one-byte transport chunk can alias a larger
//! opaque allocation. Do not use this API's charge as a process/RSS bound.

use std::{num::NonZeroUsize, ops::Range, sync::Arc};

use bytes::Bytes;
use futures::{FutureExt, StreamExt, TryStreamExt, future::BoxFuture, stream};
use lance_core::{Error, Result, utils::allocation::Reservation};

use crate::traits::Reader;

/// Admit all ranges together before starting any read, preserving range order.
///
/// Returns `None` without I/O/allocation if the workspace is too small. The caller
/// must drain or choose a smaller working set, never interpret this as EOF. Credit
/// for output/rep-def/scratch can be left in `workspace` before admitting inputs.
/// No waiting for additional credit occurs while this operation holds payloads.
///
/// Copying the streamed body into owned allocations avoids retaining an opaque
/// transport owner in returned slices. This changes copying/GET costs and must be
/// qualified before product adoption; existing readers/schedulers are unchanged.
///
/// ```no_run
/// # async fn example(reader: std::sync::Arc<dyn lance_io::traits::Reader>) -> lance_core::Result<()> {
/// use lance_core::utils::allocation::AllocationBudget;
/// use lance_io::reserved_read::try_read_ranges;
/// let budget = AllocationBudget::new(8192);
/// let mut workspace = budget.try_reserve(8192).unwrap();
/// let inputs = try_read_ranges(reader, vec![0..4096], &mut workspace, 1.try_into().unwrap())?
///     .unwrap().await?;
/// // 4096 bytes remain reserved for downstream work while inputs are alive.
/// drop(inputs);
/// # Ok(()) }
/// ```
pub fn try_read_ranges(
    reader: Arc<dyn Reader>,
    ranges: Vec<Range<usize>>,
    workspace: &mut Reservation,
    concurrency: NonZeroUsize,
) -> Result<Option<BoxFuture<'static, Result<Vec<Bytes>>>>> {
    let total = ranges.iter().try_fold(0_usize, |sum, range| {
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|len| *len <= isize::MAX as usize)
            .ok_or_else(|| {
                Error::invalid_input(format!("invalid reserved read range {range:?}"))
            })?;
        sum.checked_add(len)
            .ok_or_else(|| Error::invalid_input("reserved read size overflow"))
    })?;
    let Some(mut admitted) = workspace.try_split(total) else {
        return Ok(None);
    };
    // These child credits total exactly `total`; constructing the future performs
    // no read or payload allocation. Dropping an unpolled future returns the credit.
    let requests = ranges
        .into_iter()
        .map(|range| {
            let reservation = admitted.try_split(range.len()).ok_or_else(|| {
                Error::internal("validated read ranges exceed their total reservation")
            })?;
            Ok((range, reservation))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(
        async move {
            stream::iter(requests)
                .map(|(range, reservation)| {
                    let reader = reader.clone();
                    async move {
                        let mut output = reservation.allocate()?;
                        if range.is_empty() {
                            return Ok(output.freeze());
                        }
                        let mut body = reader.get_range_stream(range.clone()).await?;
                        let mut offset = 0;
                        while let Some(chunk) = body.next().await {
                            let chunk = chunk?;
                            let remaining = range.len() - offset;
                            if chunk.len() > remaining {
                                return Err(Error::io(format!(
                                    "body for {} range {range:?} exceeds requested length",
                                    reader.path()
                                )));
                            }
                            output.as_mut()[offset..offset + chunk.len()].copy_from_slice(&chunk);
                            offset += chunk.len();
                        }
                        if offset != range.len() {
                            return Err(Error::io(format!(
                                "body for {} range {range:?} has {offset} bytes, expected {}",
                                reader.path(),
                                range.len()
                            )));
                        }
                        Ok(output.freeze())
                    }
                })
                .buffered(concurrency.get())
                .try_collect()
                .await
        }
        .boxed(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{object_reader::CloudObjectReader, traits::ByteStream};
    use arrow::buffer::Buffer;
    use arrow_array::{Array, UInt8Array};
    use futures::poll;
    use lance_core::{
        cache::{Context, DeepSizeOf},
        utils::allocation::AllocationBudget,
    };
    use object_store::{ObjectStore, memory::InMemory, path::Path};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn slow_input_does_not_prevent_other_admitted_work_from_finishing() {
        let budget = AllocationBudget::new(16);
        let slow = Arc::new(TestReader {
            path: Path::from("slow"),
            calls: AtomicUsize::new(0),
            mode: Mode::Pending,
        });
        let mut slow_workspace = budget.try_reserve(8).unwrap();
        let mut slow_read =
            try_read_ranges(slow, vec![0..8], &mut slow_workspace, 1.try_into().unwrap())
                .unwrap()
                .unwrap();
        assert!(poll!(&mut slow_read).is_pending());
        let store = Arc::new(InMemory::new());
        let path = Path::from("fast");
        store
            .put(&path, Bytes::from_static(b"01234567").into())
            .await
            .unwrap();
        let fast = Arc::new(CloudObjectReader::new(store, path, 4096, Some(8), 0).unwrap());
        // A finite succession of independent reads completes while the slow read
        // holds half the shared budget. No operation waits for additional credit.
        for _ in 0..3 {
            let mut workspace = budget.try_reserve(8).unwrap();
            let bytes = try_read_ranges(
                fast.clone(),
                vec![0..8],
                &mut workspace,
                1.try_into().unwrap(),
            )
            .unwrap()
            .unwrap()
            .await
            .unwrap();
            assert_eq!(budget.reserved(), 16);
            assert_eq!(&bytes[0][..], b"01234567");
            assert!(budget.try_reserve(1).is_none());
            drop(bytes);
            assert_eq!(budget.reserved(), 8);
        }
        drop(slow_read);
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn ranges_share_credit_through_arrow_and_reallocation() {
        let store = Arc::new(InMemory::new());
        let path = Path::from("input");
        store
            .put(
                &path,
                Bytes::from((0..=255).cycle().take(65536).collect::<Vec<u8>>()).into(),
            )
            .await
            .unwrap();
        let reader = Arc::new(CloudObjectReader::new(store, path, 4096, Some(65536), 0).unwrap());
        let budget = AllocationBudget::new(24);
        let mut workspace = budget.try_reserve(24).unwrap();
        let buffers = try_read_ranges(
            reader,
            vec![2..10, 250..258],
            &mut workspace,
            2.try_into().unwrap(),
        )
        .unwrap()
        .unwrap()
        .await
        .unwrap();
        assert_eq!(&buffers[0][..], &[2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(&buffers[1][..], &[250, 251, 252, 253, 254, 255, 0, 1]);
        assert_eq!(workspace.len(), 8);
        // Actual Arrow conversion must retain the Bytes owner, not reconstruct a
        // reservation from Arrow's reported visible length.
        let array = UInt8Array::new(Buffer::from(buffers[0].clone()).into(), None);
        let sibling = array.slice(0, 1);
        drop(array);
        drop(buffers);
        assert_eq!(budget.reserved(), 16);
        let replacement = workspace.try_split(8).unwrap().allocate().unwrap().freeze();
        drop(workspace);
        assert_eq!(budget.reserved(), 16);
        drop(sibling);
        assert_eq!(budget.reserved(), 8);
        drop(replacement);
        assert_eq!(budget.reserved(), 0);
    }

    #[derive(Debug)]
    struct TestReader {
        path: Path,
        calls: AtomicUsize,
        mode: Mode,
    }
    #[derive(Debug, Clone, Copy)]
    enum Mode {
        Pending,
        Short,
        Long,
        Error,
    }
    impl DeepSizeOf for TestReader {
        fn deep_size_of_children(&self, _: &mut Context) -> usize {
            0
        }
    }
    impl Reader for TestReader {
        fn path(&self) -> &Path {
            &self.path
        }
        fn block_size(&self) -> usize {
            1
        }
        fn io_parallelism(&self) -> usize {
            1
        }
        fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
            async { Ok(8) }.boxed()
        }
        fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
            panic!("no full body collection")
        }
        fn get_range(&self, _: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
            panic!("use streaming read")
        }
        fn get_range_stream(
            &self,
            _: Range<usize>,
        ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mode = self.mode;
            async move {
                let stream: ByteStream = match mode {
                    Mode::Pending => stream::pending().boxed(),
                    Mode::Short => stream::iter([Ok(Bytes::from_static(b"x"))]).boxed(),
                    Mode::Long => stream::iter([Ok(Bytes::from_static(b"0123456789"))]).boxed(),
                    Mode::Error => stream::iter([Err(object_store::Error::Generic {
                        store: "test",
                        source: "injected body failure".into(),
                    })])
                    .boxed(),
                };
                Ok(stream)
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn refusal_and_cancellation_do_not_start_or_leak_reads() {
        let reader = Arc::new(TestReader {
            path: Path::from("input"),
            calls: AtomicUsize::new(0),
            mode: Mode::Pending,
        });
        let budget = AllocationBudget::new(16);
        let mut workspace = budget.try_reserve(16).unwrap();
        assert!(
            try_read_ranges(
                reader.clone(),
                vec![0..17],
                &mut workspace,
                1.try_into().unwrap()
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(workspace.len(), 16);
        let future = try_read_ranges(
            reader.clone(),
            vec![0..8, 8..16],
            &mut workspace,
            1.try_into().unwrap(),
        )
        .unwrap()
        .unwrap();
        drop(future);
        assert_eq!(reader.calls.load(Ordering::SeqCst), 0);
        assert_eq!(budget.reserved(), 0);
        let mut workspace = budget.try_reserve(16).unwrap();
        let mut future = try_read_ranges(
            reader.clone(),
            vec![0..8, 8..16],
            &mut workspace,
            1.try_into().unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(poll!(&mut future).is_pending());
        assert_eq!(reader.calls.load(Ordering::SeqCst), 1);
        assert!(budget.try_reserve(1).is_none());
        drop(future);
        assert_eq!(budget.reserved(), 0);
        assert!(budget.try_reserve(16).is_some());
    }

    #[rstest::rstest]
    #[case(Mode::Short, "expected 8")]
    #[case(Mode::Long, "exceeds requested length")]
    #[case(Mode::Error, "injected body failure")]
    #[tokio::test]
    async fn invalid_bodies_release_all_credit(#[case] mode: Mode, #[case] message: &str) {
        let reader = Arc::new(TestReader {
            path: Path::from("input"),
            calls: AtomicUsize::new(0),
            mode,
        });
        let budget = AllocationBudget::new(16);
        let mut workspace = budget.try_reserve(16).unwrap();
        let error = try_read_ranges(
            reader,
            vec![0..8, 8..16],
            &mut workspace,
            2.try_into().unwrap(),
        )
        .unwrap()
        .unwrap()
        .await
        .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(budget.reserved(), 0);
    }
}
