// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded scheduling for native RaBitQ plane prewarming.

use super::{IvfQuantizationStorage, plane_cache_entry};
use crate::vector::quantizer::Quantization;
use futures::{Future, StreamExt, TryStreamExt, stream};
use lance_arrow::DataTypeExt;
use lance_core::cache::WeakLanceCache;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, Result};
use tokio::sync::Semaphore;

// A per-prewarm staging target, independent of cache admission. Account for a
// whole partition's fixed-width values even when warming only one plane. This
// is not a process-memory cap: decoder buffers, cache residency and concurrent
// prewarm calls are separate. One oversized partition must still make progress.
const PREWARM_VALUE_BYTES: u32 = 64 * 1024 * 1024;

impl<Q: Quantization> IvfQuantizationStorage<Q> {
    /// Warm every plane entry a full-precision scan reads, overlapping partitions
    /// up to the CPU parallelism and a 64 MiB estimated-value staging target.
    /// A partition larger than the target is loaded alone. Cache backends still
    /// enforce their own byte admission budgets; this is not an RSS limit.
    ///
    /// Gated backends finish all sign planes before checking which partitions
    /// still have a resident sign plane and warming their ex planes. Other
    /// backends warm both planes of every partition, sign first, including
    /// entries that can only be retained by a persistent tier.
    pub async fn prewarm_planes(&self, cache: &WeakLanceCache) -> Result<()> {
        let row_bytes = self
            .schema()
            .fields()
            .iter()
            .try_fold(0_usize, |sum, field| {
                sum.checked_add(field.data_type().byte_width_opt()?)
            });
        let partitions = || {
            (0..self.num_partitions()).map(|part_id| {
                // Unknown or overflowing sizes conservatively run alone.
                let bytes = row_bytes
                    .and_then(|width| width.checked_mul(self.partition_size(part_id)))
                    .unwrap_or(PREWARM_VALUE_BYTES as usize);
                (part_id, bytes)
            })
        };
        let parallelism = get_num_compute_intensive_cpus();
        let load = |part_id, plane| async move {
            cache
                .get_or_insert_with_key(self.plane_key(part_id, plane), || async move {
                    Ok(plane_cache_entry(
                        self.read_plane_entry(part_id, plane, None).await?,
                    ))
                })
                .await?;
            Ok(())
        };
        if !cache.plane_admission_gated() {
            return run(
                partitions(),
                parallelism,
                PREWARM_VALUE_BYTES,
                |part_id| async move {
                    load(part_id, 0).await?;
                    load(part_id, 1).await
                },
            )
            .await;
        }
        for plane in [0, 1] {
            run(
                partitions(),
                parallelism,
                PREWARM_VALUE_BYTES,
                |part_id| async move {
                    // Probe after the byte reservation, so queued work does not
                    // use residency observed before earlier admissions/evictions.
                    if plane == 0
                        || cache
                            .peek_resident_with_key(&self.plane_key(part_id, 0))
                            .await
                    {
                        load(part_id, plane).await?;
                    }
                    Ok(())
                },
            )
            .await?;
        }
        Ok(())
    }
}

// Keep the reservation through cache admission, and borrow futures rather than
// spawning tasks: errors and caller cancellation drop outstanding reservations
// and loaders. The cache's existing single-flight boundary owns deduplication.
async fn run<I, F, Fut>(partitions: I, parallelism: usize, value_bytes: u32, load: F) -> Result<()>
where
    I: IntoIterator<Item = (usize, usize)>,
    F: Fn(usize) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let budget = Semaphore::new(value_bytes as usize);
    stream::iter(partitions)
        .map(Ok)
        .try_for_each_concurrent(parallelism, |(part_id, bytes)| {
            let budget = &budget;
            let load = &load;
            async move {
                let _reservation = budget
                    .acquire_many(bytes.min(value_bytes as usize).max(1) as u32)
                    .await
                    .map_err(|e| {
                        Error::internal(format!("prewarm byte reservation failed: {e}"))
                    })?;
                load(part_id).await
            }
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::run;
    use futures::FutureExt;
    use lance_core::Error;
    use std::sync::{Arc, Mutex};
    use tokio::sync::Semaphore;

    #[tokio::test]
    async fn prewarm_reserves_bytes_and_isolates_oversized_partitions() {
        let releases: Vec<_> = (0..4).map(|_| Semaphore::new(0)).collect();
        let starts = Mutex::new(Vec::new());
        let warm = run([(0, 4), (1, 4), (2, 1), (3, 16)], 3, 8, |part| {
            let releases = &releases;
            let starts = &starts;
            async move {
                starts.lock().unwrap().push(part);
                releases[part].acquire().await.unwrap().forget();
                Ok(())
            }
        });
        tokio::pin!(warm);
        assert!(futures::poll!(&mut warm).is_pending());
        assert_eq!(*starts.lock().unwrap(), [0, 1]);
        releases[0].add_permits(1);
        assert!(futures::poll!(&mut warm).is_pending());
        assert_eq!(*starts.lock().unwrap(), [0, 1, 2]);
        releases[1].add_permits(1);
        assert!(futures::poll!(&mut warm).is_pending());
        assert_eq!(*starts.lock().unwrap(), [0, 1, 2]);
        releases[2].add_permits(1);
        assert!(futures::poll!(&mut warm).is_pending());
        assert_eq!(*starts.lock().unwrap(), [0, 1, 2, 3]);
        releases[3].add_permits(1);
        warm.await.unwrap();
    }

    #[tokio::test]
    async fn prewarm_limits_empty_partitions_and_cancels_loaders() {
        let live = Arc::new(());
        let mut warm = Box::pin(run((0..10).map(|p| (p, 0)), 2, 8, |_| {
            let live = live.clone();
            async move {
                let _live = live;
                futures::future::pending().await
            }
        }));
        assert!(futures::poll!(&mut warm).is_pending());
        assert_eq!(Arc::strong_count(&live), 3);
        drop(warm);
        assert_eq!(Arc::strong_count(&live), 1);
    }

    #[tokio::test]
    async fn prewarm_propagates_error_and_drops_other_loaders() {
        let live = Arc::new(());
        let failed = run([(0, 1), (1, 1), (2, 1)], 2, 8, |part| {
            let live = live.clone();
            async move {
                let _live = live;
                if part == 1 {
                    return Err(Error::io("injected plane read failure"));
                }
                futures::future::pending().await
            }
        })
        .now_or_never()
        .unwrap()
        .unwrap_err();
        assert!(matches!(failed, Error::IO { .. }));
        assert!(failed.to_string().contains("injected plane read failure"));
        assert_eq!(Arc::strong_count(&live), 1);
    }
}
