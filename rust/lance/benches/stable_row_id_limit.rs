// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Measures V2 LIMIT/OFFSET scans over stable-row-ID fragments with deletions.
//!
//! ```text
//! cargo bench -p lance --bench stable_row_id_limit
//! ```

use std::sync::Arc;
use std::time::Duration;

use arrow_array::types::UInt32Type;
use criterion::{Criterion, criterion_group, criterion_main};
use lance::dataset::{Dataset, WriteParams};
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_table::format::Fragment;

const NUM_FRAGMENTS: usize = 128;
const ROWS_PER_FRAGMENT: usize = 1024;
const LIVE_ROWS_PER_FRAGMENT: usize = ROWS_PER_FRAGMENT - 1;
const OFFSET: usize = (NUM_FRAGMENTS - 8) * LIVE_ROWS_PER_FRAGMENT;
const LIMIT: usize = 16;

struct Fixture {
    _data_dir: TempStrDir,
    dataset: Arc<Dataset>,
    fragments: Vec<Fragment>,
}

impl Fixture {
    async fn open() -> Self {
        let data_dir = TempStrDir::default();
        let reader = gen_batch()
            .col("value", array::step::<UInt32Type>())
            .into_reader_rows(
                RowCount::from(ROWS_PER_FRAGMENT as u64),
                BatchCount::from(NUM_FRAGMENTS as u32),
            );
        let mut dataset = Dataset::write(
            reader,
            data_dir.as_str(),
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // One deletion per fragment forces the fallback path to load every deletion vector.
        dataset
            .delete(&format!("value % {ROWS_PER_FRAGMENT} = 0"))
            .await
            .unwrap();

        let fragments = dataset.fragments().as_ref().clone();
        assert_eq!(fragments.len(), NUM_FRAGMENTS);
        assert!(
            fragments
                .iter()
                .all(|fragment| fragment.deletion_file.is_some())
        );

        Self {
            _data_dir: data_dir,
            dataset: Arc::new(dataset),
            fragments,
        }
    }
}

async fn scan_slice(dataset: &Dataset, fragments: &[Fragment]) -> usize {
    let mut scanner = dataset.scan();
    scanner.with_fragments(fragments.to_vec());
    scanner
        .limit(Some(LIMIT as i64), Some(OFFSET as i64))
        .unwrap();
    scanner.try_into_batch().await.unwrap().num_rows()
}

fn bench_stable_row_id_limit(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let fixture = runtime.block_on(Fixture::open());

    c.bench_function("stable_row_id_limit", |b| {
        b.to_async(&runtime).iter(|| async {
            let rows = scan_slice(&fixture.dataset, &fixture.fragments).await;
            assert_eq!(rows, LIMIT);
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);
    targets = bench_stable_row_id_limit
);
criterion_main!(benches);
