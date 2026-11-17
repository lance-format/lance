// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise local optimization concurrency through the public dataset API.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::types::Int32Type;
use lance::dataset::WriteParams;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::{Dataset, Error};
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::tokio::{limit_compute_concurrency, spawn_cpu};
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::IndexBuildProgress;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::InvertedIndexParams;

#[derive(Debug, Default)]
struct CheckOptimizeConcurrency {
    stages: AtomicUsize,
}

#[async_trait::async_trait]
impl IndexBuildProgress for CheckOptimizeConcurrency {
    async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> lance_core::Result<()> {
        assert_eq!(limit_compute_concurrency(usize::MAX), 1);
        // Check the worker path as well as the top-level optimization future.
        assert_eq!(
            spawn_cpu(|| Ok::<_, Error>(limit_compute_concurrency(usize::MAX))).await?,
            1
        );
        self.stages.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn stage_progress(&self, _: &str, _: u64) -> lance_core::Result<()> {
        assert_eq!(limit_compute_concurrency(usize::MAX), 1);
        Ok(())
    }

    async fn stage_complete(&self, _: &str) -> lance_core::Result<()> {
        assert_eq!(limit_compute_concurrency(usize::MAX), 1);
        Ok(())
    }
}

async fn dataset(path: &str, num_fragments: u32) -> Dataset {
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("text", array::fill_utf8("hello world".into()))
        .into_reader_rows(RowCount::from(4), BatchCount::from(num_fragments));
    Dataset::write(
        data,
        path,
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn test_optimize_num_threads_reaches_index_build_stages() {
    let test_dir = TempStrDir::default();
    let mut dataset = dataset(&test_dir, 2).await;
    assert_eq!(dataset.get_fragments().len(), 2);
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let appended = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("text", array::fill_utf8("hello world".into()))
        .into_reader_rows(RowCount::from(4), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();
    let progress = Arc::new(CheckOptimizeConcurrency::default());
    dataset
        .optimize_indices(
            &OptimizeOptions::merge(1)
                .with_num_threads(1)
                .progress(progress.clone()),
        )
        .await
        .unwrap();
    assert!(progress.stages.load(Ordering::Relaxed) > 0);
    assert!(
        dataset
            .unindexed_fragments("text_idx")
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(dataset.count_rows(None).await.unwrap(), 12);
    let matches = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("hello".into()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(matches.num_rows(), 12);
    dataset.validate().await.unwrap();
    assert_eq!(limit_compute_concurrency(usize::MAX), usize::MAX);
}

#[tokio::test]
async fn test_optimize_num_threads_rejects_zero_before_work() {
    let test_dir = TempStrDir::default();
    let mut dataset = dataset(&test_dir, 1).await;
    let version = dataset.version().version;
    let error = dataset
        .optimize_indices(&OptimizeOptions::default().with_num_threads(0))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("num_threads must be greater than 0")
    );
    assert_eq!(dataset.version().version, version);
}
