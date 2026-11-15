// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in whole-stream reads without Arrow payload materialization.

use super::*;
use crate::scalar::IndexBinaryReader;

// Independent of selective windows: this preserves one complete term stream
// and ordinary WAND execution. Qualification must include large-term RSS.
static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("LANCE_FTS_DIRECT_POSITIONS").is_ok_and(|value| value == "1"));

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
        read_payload(value.as_ref()),
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

async fn read_payload(value: &dyn IndexBinaryReader) -> Result<bytes::Bytes> {
    let expected = value.len();
    let mut buffers = value.read_slices(&[0..expected]).await?;
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
    // Split large I/O ranges are already reconstructed into an owned Vec.
    // Consuming Bytes recovers that allocation, copying custom/shared owners.
    // Vec exposes the full allocation capacity, including a hidden prefix:
    // uniqueness or BytesMut's remaining capacity alone cannot certify this.
    let owned = Vec::<u8>::from(buffer);
    Ok(if owned.capacity() == owned.len() {
        bytes::Bytes::from(owned)
    } else {
        bytes::Bytes::copy_from_slice(&owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::AtomicBool;

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
        let payload = read_payload(&value).await.unwrap();
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
        let payload = read_payload(&value).await.unwrap();
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
        let error = read_payload(&value).await.unwrap_err();
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
        let mut read = Box::pin(read_payload(&value));
        assert!(futures::poll!(read.as_mut()).is_pending());
        assert!(active.load(Ordering::SeqCst));
        drop(read);
        assert!(!active.load(Ordering::SeqCst));
    }
}
