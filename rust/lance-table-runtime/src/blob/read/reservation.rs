// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blob::BlobMaterializationReservation;
use bytes::Bytes;
use lance_io::traits::Reader;
use object_store::path::Path;
use std::sync::Arc;

/// Scheduled reader futures retain both reservations even if the consumer drops
/// its stream while backend I/O is still in flight.
#[derive(Debug)]
pub struct ReadReservation {
    _materialization: Option<Arc<BlobMaterializationReservation>>,
    _io: Option<Arc<BlobMaterializationReservation>>,
}

impl ReadReservation {
    /// Keep both admissions charged until the scheduled reads complete.
    pub fn new(
        materialization: Option<Arc<BlobMaterializationReservation>>,
        io: Option<Arc<BlobMaterializationReservation>>,
    ) -> Self {
        Self {
            _materialization: materialization,
            _io: io,
        }
    }
}

#[derive(Debug)]
pub(super) struct ReservedReader {
    pub(super) reader: Arc<dyn Reader>,
    pub(super) reservation: Arc<ReadReservation>,
}

impl lance_core::deepsize::DeepSizeOf for ReservedReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.reader.deep_size_of_children(context)
    }
}

impl Reader for ReservedReader {
    fn path(&self) -> &Path {
        self.reader.path()
    }
    fn block_size(&self) -> usize {
        self.reader.block_size()
    }
    fn io_parallelism(&self) -> usize {
        self.reader.io_parallelism()
    }
    fn size(&self) -> futures::future::BoxFuture<'_, object_store::Result<usize>> {
        self.reader.size()
    }

    fn get_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> futures::future::BoxFuture<'static, object_store::Result<Bytes>> {
        let future = self.reader.get_range(range);
        let reservation = self.reservation.clone();
        Box::pin(async move {
            let result = future.await;
            drop(reservation);
            result
        })
    }

    fn get_all(&self) -> futures::future::BoxFuture<'_, object_store::Result<Bytes>> {
        self.reader.get_all()
    }
}
