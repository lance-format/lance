// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded whole-stream reads without Arrow payload materialization.

use super::*;
use crate::scalar::IndexBinaryReader;

// Preserve complete term streams and ordinary WAND execution. An explicit zero
// retains the Arrow fallback for diagnosing supported current-layout reads.
// Unsupported encodings and frozen legacy layouts keep their existing readers.
static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("LANCE_FTS_DIRECT_POSITIONS").map_or(true, |value| value == "1")
});

// Bound staging for one complete value without changing ordinary posting I/O.
// The final stream and concurrent values still count separately from staging.
const PAYLOAD_READ_BYTES: usize = 2 * 1024 * 1024;
const MAX_PENDING_PAYLOAD_READS: usize = 8;

pub(super) fn enabled() -> bool {
    *ENABLED
}

pub(super) async fn try_read(
    reader: &dyn IndexReader,
    token_id: u32,
    codec: PositionStreamCodec,
) -> Result<Option<SharedPositionStream>> {
    let Some(value) = reader
        .try_prepare_binary_value(COMPRESSED_POSITION_COL, u64::from(token_id))
        .await?
    else {
        // Unsupported encodings keep the ordinary Arrow reader.
        return Ok(None);
    };
    let row = token_id as usize;
    let (batch, bytes) = futures::try_join!(
        reader.read_range(row..row + 1, Some(&[POSITION_BLOCK_OFFSET_COL])),
        read_payload(value.as_ref(), PAYLOAD_READ_BYTES),
    )?;
    let offsets = batch
        .column_by_name(POSITION_BLOCK_OFFSET_COL)
        .ok_or_else(|| Error::index("whole positions read lost its block offsets column"))?
        .as_list::<i32>()
        .value(0);
    let offsets = offsets.as_primitive::<UInt32Type>().values().to_vec();
    if offsets.first().is_some_and(|offset| *offset != 0)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || offsets
            .last()
            .is_some_and(|offset| *offset as usize > bytes.len())
    {
        return Err(Error::index(format!(
            "invalid block offsets for whole positions token {token_id}, payload {} bytes",
            bytes.len(),
        )));
    }
    Ok(Some(SharedPositionStream::new(codec, offsets, bytes)))
}

async fn read_payload(value: &dyn IndexBinaryReader, chunk_bytes: usize) -> Result<bytes::Bytes> {
    if chunk_bytes == 0 {
        return Err(Error::invalid_input(
            "whole positions read chunk size must be positive",
        ));
    }
    let expected = value.len();
    if expected <= chunk_bytes as u64 {
        return Ok(compact_payload(
            read_payload_range(value, 0..expected).await?,
        ));
    }
    let len = usize::try_from(expected).map_err(|_| {
        Error::index(format!(
            "whole positions payload of {expected} bytes exceeds the address space",
        ))
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|error| {
        Error::io(format!(
            "cannot reserve {len} bytes for whole positions payload: {error}",
        ))
    })?;
    // Separate requests prevent the scheduler from coalescing adjacent chunks
    // back into a large read. Consume completion order so a stalled first read
    // neither retains ready chunks nor hides an error from a later read.
    // Dropping this stream also drops its pending, unspawned read futures.
    let mut reads = futures::stream::iter((0..expected).step_by(chunk_bytes).map(|start| {
        let end = start.saturating_add(chunk_bytes as u64).min(expected);
        async move {
            let bytes = read_payload_range(value, start..end).await?;
            Result::Ok((start as usize, bytes))
        }
    }))
    .buffer_unordered(MAX_PENDING_PAYLOAD_READS);
    let mut initialized = 0usize;
    while let Some((start, bytes)) = reads.try_next().await? {
        let end = start + bytes.len();
        // Keep the vector length at zero while filling disjoint ranges. A late
        // prefix must not force us to zero and touch pages before their reads
        // finish. No spare-capacity byte is exposed or read here.
        let destination = output
            .spare_capacity_mut()
            .get_mut(start..end)
            .ok_or_else(|| Error::index("whole positions read exceeds reserved output"))?;
        for (destination, byte) in destination.iter_mut().zip(bytes.iter()) {
            destination.write(*byte);
        }
        initialized = initialized
            .checked_add(bytes.len())
            .filter(|initialized| *initialized <= len)
            .ok_or_else(|| Error::index("whole positions initialized byte count overflow"))?;
    }
    if initialized != len {
        return Err(Error::index(format!(
            "whole positions initialized {initialized} bytes, expected {len}",
        )));
    }
    // SAFETY: try_reserve_exact reserved at least len bytes. The step_by ranges
    // cover 0..len exactly once, and read_payload_range validates every result's
    // length before its bytes are written. Exhausting the stream successfully
    // therefore initializes every byte in 0..len, including a partial tail.
    // Error and cancellation leave the vector length at zero and publish nothing.
    unsafe {
        output.set_len(len);
    }
    Ok(compact_payload(bytes::Bytes::from(output)))
}

async fn read_payload_range(
    value: &dyn IndexBinaryReader,
    range: Range<u64>,
) -> Result<bytes::Bytes> {
    let expected = range.end - range.start;
    let mut buffers = value.read_slices(std::slice::from_ref(&range)).await?;
    let buffer = match buffers.len() {
        1 => buffers.remove(0),
        count => {
            return Err(Error::index(format!(
                "whole positions read returned {count} buffers, expected 1",
            )));
        }
    };
    if buffer.len() as u64 != expected {
        return Err(Error::index(format!(
            "whole positions read returned {} bytes, expected {expected}",
            buffer.len(),
        )));
    }
    Ok(buffer)
}

fn compact_payload(buffer: bytes::Bytes) -> bytes::Bytes {
    // Consuming Bytes recovers a standard allocation, copying custom/shared owners.
    // Vec exposes the full allocation capacity, including a hidden prefix:
    // uniqueness or BytesMut's remaining capacity alone cannot certify this.
    let owned = Vec::<u8>::from(buffer);
    if owned.capacity() == owned.len() {
        bytes::Bytes::from(owned)
    } else {
        bytes::Bytes::copy_from_slice(&owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    struct Backing {
        bytes: Vec<u8>,
        _lifetime: Arc<()>,
    }

    impl AsRef<[u8]> for Backing {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }

    #[derive(Debug)]
    struct BinaryValue {
        len: u64,
        buffers: usize,
        short: bool,
        lifetime: std::sync::Mutex<Option<std::sync::Weak<()>>>,
    }

    #[async_trait]
    impl IndexBinaryReader for BinaryValue {
        fn len(&self) -> u64 {
            self.len
        }

        async fn read_slices(&self, slices: &[Range<u64>]) -> Result<Vec<bytes::Bytes>> {
            assert_eq!(slices, &[0..self.len]);
            let lifetime = Arc::new(());
            *self.lifetime.lock().unwrap() = Some(Arc::downgrade(&lifetime));
            let backing = bytes::Bytes::from_owner(Backing {
                bytes: vec![7; 16384],
                _lifetime: lifetime,
            });
            Ok(vec![
                backing.slice(
                    128..128 + self.len as usize - usize::from(self.short)
                );
                self.buffers
            ])
        }
    }

    #[rstest::rstest]
    #[case::empty(0)]
    #[case::custom_owner(1024)]
    #[tokio::test]
    async fn selective_phrase_whole_payload_releases_coalesced_backing(#[case] len: u64) {
        let value = BinaryValue {
            len,
            buffers: 1,
            short: false,
            lifetime: Default::default(),
        };
        let payload = read_payload(&value, PAYLOAD_READ_BYTES).await.unwrap();
        assert_eq!(payload.as_ref(), vec![7; len as usize]);
        assert!(
            value
                .lifetime
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .upgrade()
                .is_none()
        );
    }

    #[derive(Debug)]
    struct OwnedValue {
        bytes: std::sync::Mutex<Option<bytes::Bytes>>,
        len: u64,
    }

    #[async_trait]
    impl IndexBinaryReader for OwnedValue {
        fn len(&self) -> u64 {
            self.len
        }

        async fn read_slices(&self, slices: &[Range<u64>]) -> Result<Vec<bytes::Bytes>> {
            assert_eq!(slices, &[0..self.len]);
            Ok(vec![self.bytes.lock().unwrap().take().unwrap()])
        }
    }

    #[rstest::rstest]
    #[case::reconstructed(0, 0, false, true)]
    #[case::hidden_prefix(1024, 0, false, false)]
    #[case::spare_capacity(0, 1024, false, false)]
    #[case::shared(0, 0, true, false)]
    #[tokio::test]
    async fn selective_phrase_whole_payload_preserves_only_compact_ownership(
        #[case] prefix: usize,
        #[case] spare: usize,
        #[case] shared: bool,
        #[case] reuse: bool,
    ) {
        let len = 128;
        let mut owned = Vec::with_capacity(prefix + len + spare);
        owned.extend((0..prefix + len).map(|value| value as u8));
        let bytes = bytes::Bytes::from(owned).slice(prefix..prefix + len);
        let original = bytes.as_ptr();
        let expected = bytes.to_vec();
        let alias = shared.then(|| bytes.clone());
        let value = OwnedValue {
            bytes: std::sync::Mutex::new(Some(bytes)),
            len: len as u64,
        };
        let payload = read_payload(&value, PAYLOAD_READ_BYTES).await.unwrap();
        assert_eq!(payload.as_ref(), expected);
        assert_eq!(payload.as_ptr() == original, reuse);
        if let Some(alias) = alias {
            assert_eq!(alias.as_ref(), expected);
        }
        // Consuming the result exposes its complete standard allocation, not
        // the remaining span of a sliced mutable view.
        let owned = Vec::<u8>::from(payload);
        assert_eq!(owned.capacity(), owned.len());
    }

    #[rstest::rstest]
    #[case::missing(0, false, "buffers")]
    #[case::extra(2, false, "buffers")]
    #[case::short(1, true, "bytes")]
    #[tokio::test]
    async fn selective_phrase_whole_payload_rejects_invalid_reads(
        #[case] buffers: usize,
        #[case] short: bool,
        #[case] message: &str,
    ) {
        let value = BinaryValue {
            len: 8,
            buffers,
            short,
            lifetime: Default::default(),
        };
        let error = read_payload(&value, PAYLOAD_READ_BYTES).await.unwrap_err();
        assert!(matches!(error, Error::Index { .. }), "{error:?}");
        assert!(error.to_string().contains(message), "{error}");
        assert!(
            value
                .lifetime
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .upgrade()
                .is_none()
        );
    }

    #[derive(Debug)]
    struct PendingValue(Arc<AtomicBool>);

    struct ActiveRead(Arc<AtomicBool>);

    impl Drop for ActiveRead {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl IndexBinaryReader for PendingValue {
        fn len(&self) -> u64 {
            8
        }

        async fn read_slices(&self, _: &[Range<u64>]) -> Result<Vec<bytes::Bytes>> {
            self.0.store(true, Ordering::SeqCst);
            let _active = ActiveRead(self.0.clone());
            futures::future::pending().await
        }
    }

    #[tokio::test]
    async fn selective_phrase_whole_payload_cancellation_drops_pending_io() {
        let active = Arc::new(AtomicBool::new(false));
        let value = PendingValue(active.clone());
        let mut read = Box::pin(read_payload(&value, PAYLOAD_READ_BYTES));
        assert!(futures::poll!(read.as_mut()).is_pending());
        assert!(active.load(Ordering::SeqCst));
        drop(read);
        assert!(!active.load(Ordering::SeqCst));
    }

    #[derive(Debug)]
    struct RangedValue {
        len: u64,
        gates: Vec<tokio::sync::Notify>,
        active: Arc<AtomicUsize>,
        peak: AtomicUsize,
        requests: std::sync::Mutex<Vec<Range<u64>>>,
        owners: std::sync::Mutex<Vec<std::sync::Weak<()>>>,
        failure: Option<(usize, bool)>,
    }

    impl RangedValue {
        fn new() -> Self {
            Self {
                len: 83,
                gates: (0..11).map(|_| tokio::sync::Notify::new()).collect(),
                active: Arc::new(AtomicUsize::new(0)),
                peak: AtomicUsize::new(0),
                requests: Default::default(),
                owners: Default::default(),
                failure: None,
            }
        }
    }

    struct PendingRange(Arc<AtomicUsize>);

    impl Drop for PendingRange {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl IndexBinaryReader for RangedValue {
        fn len(&self) -> u64 {
            self.len
        }

        async fn read_slices(&self, slices: &[Range<u64>]) -> Result<Vec<bytes::Bytes>> {
            let [range] = slices else {
                panic!("each bounded read must submit one slice");
            };
            assert!(range.end > range.start && range.end - range.start <= 8);
            self.requests.lock().unwrap().push(range.clone());
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            let _pending = PendingRange(self.active.clone());
            self.peak.fetch_max(active, Ordering::SeqCst);
            let ordinal = range.start as usize / 8;
            self.gates[ordinal].notified().await;
            let mut end = range.end;
            if let Some((failure, short)) = self.failure
                && failure == ordinal
            {
                if !short {
                    return Err(Error::index("injected range read failure"));
                }
                end -= 1;
            }
            let owner = Arc::new(());
            self.owners.lock().unwrap().push(Arc::downgrade(&owner));
            Ok(vec![bytes::Bytes::from_owner(Backing {
                bytes: (range.start..end)
                    .map(|offset| (offset % 251) as u8)
                    .collect(),
                _lifetime: owner,
            })])
        }
    }

    #[tokio::test]
    async fn selective_phrase_whole_payload_orders_bounded_reads_and_partial_tail() {
        let value = RangedValue::new();
        let mut read = Box::pin(read_payload(&value, 8));
        assert!(futures::poll!(read.as_mut()).is_pending());
        assert_eq!(value.active.load(Ordering::SeqCst), 8);
        // Complete later reads first. Consume their bytes and admit more work
        // without waiting for the first read, preserving final byte order.
        for gate in &value.gates[1..8] {
            gate.notify_one();
        }
        assert!(futures::poll!(read.as_mut()).is_pending());
        assert_eq!(value.active.load(Ordering::SeqCst), 4);
        assert_eq!(value.requests.lock().unwrap().len(), 11);
        assert!(
            value
                .owners
                .lock()
                .unwrap()
                .iter()
                .all(|owner| owner.upgrade().is_none())
        );
        value.gates[0].notify_one();
        assert!(futures::poll!(read.as_mut()).is_pending());
        for gate in &value.gates[8..] {
            gate.notify_one();
        }
        let payload = read.await.unwrap();
        assert_eq!(payload.as_ref(), (0..83).collect::<Vec<u8>>());
        assert_eq!(value.requests.lock().unwrap().len(), 11);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        assert!(value.peak.load(Ordering::SeqCst) <= 8);
        assert!(
            value
                .owners
                .lock()
                .unwrap()
                .iter()
                .all(|owner| owner.upgrade().is_none())
        );
        let owned = Vec::<u8>::from(payload);
        assert_eq!(owned.capacity(), owned.len());
    }

    #[rstest::rstest]
    #[case::reverse((0..11).rev().collect())]
    #[case::alternating(vec![1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 10])]
    #[case::tail_before_prefix(vec![7, 6, 5, 4, 3, 2, 1, 0, 10, 9, 8])]
    #[tokio::test]
    async fn selective_phrase_whole_payload_preserves_bytes_across_completion_orders(
        #[case] order: Vec<usize>,
    ) {
        let value = RangedValue::new();
        let mut read = Box::pin(read_payload(&value, 8));
        assert!(futures::poll!(read.as_mut()).is_pending());
        let mut payload = None;
        for (step, ordinal) in order.iter().enumerate() {
            value.gates[*ordinal].notify_one();
            if let std::task::Poll::Ready(result) = futures::poll!(read.as_mut()) {
                assert_eq!(step + 1, order.len(), "published before all reads finished");
                payload = Some(result.unwrap());
                break;
            }
        }
        let payload = match payload {
            Some(payload) => payload,
            None => read.await.unwrap(),
        };
        assert_eq!(payload.as_ref(), (0..83).collect::<Vec<u8>>());
        assert_eq!(value.requests.lock().unwrap().len(), 11);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        assert!(value.peak.load(Ordering::SeqCst) <= 8);
        assert!(
            value
                .owners
                .lock()
                .unwrap()
                .iter()
                .all(|owner| owner.upgrade().is_none())
        );
    }

    #[tokio::test]
    async fn selective_phrase_whole_payload_cancel_drops_ready_and_pending_chunks() {
        let value = RangedValue::new();
        let mut read = Box::pin(read_payload(&value, 8));
        assert!(futures::poll!(read.as_mut()).is_pending());
        for gate in &value.gates[1..8] {
            gate.notify_one();
        }
        assert!(futures::poll!(read.as_mut()).is_pending());
        assert_eq!(value.active.load(Ordering::SeqCst), 4);
        assert_eq!(value.owners.lock().unwrap().len(), 7);
        assert!(
            value
                .owners
                .lock()
                .unwrap()
                .iter()
                .all(|owner| owner.upgrade().is_none())
        );
        drop(read);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        assert_eq!(value.requests.lock().unwrap().len(), 11);
        assert!(
            value
                .owners
                .lock()
                .unwrap()
                .iter()
                .all(|owner| owner.upgrade().is_none())
        );
    }

    #[tokio::test]
    async fn selective_phrase_whole_payload_later_error_bypasses_stalled_first_read() {
        let mut value = RangedValue::new();
        value.failure = Some((1, false));
        value.gates[1].notify_one();
        let mut read = Box::pin(read_payload(&value, 8));
        let result = futures::poll!(read.as_mut());
        let std::task::Poll::Ready(Err(error)) = result else {
            panic!("a completed read error must not wait for an earlier pending read");
        };
        assert!(matches!(error, Error::Index { .. }), "{error:?}");
        assert!(error.to_string().contains("injected"), "{error}");
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        assert!(value.peak.load(Ordering::SeqCst) <= 8);
    }

    #[rstest::rstest]
    #[case::first_error(0, false)]
    #[case::middle_short_read(3, true)]
    #[case::tail_error(10, false)]
    #[tokio::test]
    async fn selective_phrase_whole_payload_failure_drops_other_chunks(
        #[case] ordinal: usize,
        #[case] short: bool,
    ) {
        let mut value = RangedValue::new();
        value.failure = Some((ordinal, short));
        for gate in &value.gates[..=ordinal] {
            gate.notify_one();
        }
        let error = read_payload(&value, 8).await.unwrap_err();
        assert!(matches!(error, Error::Index { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains(if short { "bytes" } else { "injected" }),
            "{error}"
        );
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        assert!(value.peak.load(Ordering::SeqCst) <= 8);
        assert!(
            value
                .owners
                .lock()
                .unwrap()
                .iter()
                .all(|owner| owner.upgrade().is_none())
        );
    }

    #[tokio::test]
    async fn selective_phrase_whole_payload_rejects_unaddressable_allocation_before_io() {
        let mut value = RangedValue::new();
        value.len = u64::MAX;
        let error = read_payload(&value, 8).await.unwrap_err();
        assert!(
            matches!(error, Error::IO { .. } | Error::Index { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("bytes"), "{error}");
        assert!(value.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn selective_phrase_whole_payload_rejects_zero_chunk_before_io() {
        let value = RangedValue::new();
        let error = read_payload(&value, 0).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("positive"), "{error}");
        assert!(value.requests.lock().unwrap().is_empty());
    }
}
