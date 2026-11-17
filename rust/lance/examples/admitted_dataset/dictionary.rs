// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Validate the actual measured input instead of relying on encoder hints.

use super::*;
use lance_core::cache::LanceCache;
use lance_encoding::{
    decoder::PageEncoding,
    format::metadata::{CompressionScheme, compressive_encoding::Compression, page_layout::Layout},
};
use lance_file::reader::{FileReader, FileReaderOptions};

pub(super) async fn verify_layout(dataset: &Dataset) -> Result<()> {
    let leaf_id = dataset
        .schema()
        .field("frames")
        .and_then(|field| field.children.first())
        .ok_or("dictionary fixture requires a frames leaf field")?
        .id;
    let scheduler = ScanScheduler::new(
        dataset.object_store(None).await?,
        SchedulerConfig::new(1024 * 1024),
    );
    let mut has_pages = false;
    for fragment in dataset.iter_fragments() {
        for file in &fragment.files {
            let column = file
                .fields
                .iter()
                .zip(file.column_indices.iter())
                .find_map(|(field, column)| (*field == leaf_id).then_some(*column))
                .ok_or_else(|| {
                    format!("file {} has no physical frames leaf {leaf_id}", file.path)
                })?;
            let source = scheduler
                .open_file(
                    &dataset.data_dir().join(file.path.as_str()),
                    &file.file_size_bytes,
                )
                .await?;
            let reader = FileReader::try_open(
                source,
                None,
                Default::default(),
                &Arc::new(LanceCache::no_cache()),
                FileReaderOptions::default(),
            )
            .await?;
            let column = reader
                .metadata()
                .column_infos
                .get(usize::try_from(column)?)
                .ok_or("frames physical column is absent from file metadata")?;
            if column.page_infos.is_empty() {
                return Err(format!("file {} has no frames pages", file.path).into());
            }
            for page in column.page_infos.iter() {
                let layout = match &page.encoding {
                    PageEncoding::Structural(layout) => layout.clone(),
                    PageEncoding::Planar(layout) => layout.dense_execution_layout()?,
                    PageEncoding::Legacy(_) => {
                        return Err(
                            "dictionary fixture requires a current structural writer".into()
                        );
                    }
                };
                let is_valid = if let Some(Layout::MiniBlockLayout(layout)) = &layout.layout {
                    let scheme = match layout
                        .dictionary
                        .as_ref()
                        .and_then(|d| d.compression.as_ref())
                    {
                        Some(Compression::General(general)) => {
                            general.compression.as_ref().map(|c| c.scheme())
                        }
                        _ => None,
                    };
                    scheme == Some(CompressionScheme::CompressionAlgorithmLz4)
                        && layout.num_dictionary_items > 1
                        && layout.num_dictionary_items <= 31
                        && layout.repetition_index_depth == 1
                } else {
                    false
                };
                if !is_valid {
                    return Err(format!(
                        "file {} requires MiniBlock/LZ4 with 2..=31 dictionary frames: {layout:?}; encoder hints alone do not qualify this fixture",
                        file.path
                    ).into());
                }
                has_pages = true;
            }
        }
    }
    if !has_pages {
        return Err("dictionary fixture has no stored frames pages".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::one(1, 1)]
    #[case::contended(2, 1)]
    #[case::overlapping(3, 2)]
    #[tokio::test]
    async fn dictionary_workload_drains_default_codecs_with_shared_credit(
        #[case] requests: usize,
        #[case] slots: usize,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("dictionary.lance")
            .to_str()
            .unwrap()
            .to_string();
        let expected = prepare(Fixture::Dictionary, &uri, 256, 1024, 128)
            .await
            .unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let pool_bytes = slots * 128 * 1024;
        assert!(expected.payload_bytes > pool_bytes as u64);
        for mode in [Mode::Ordinary, Mode::Admitted] {
            let config = DrainConfig {
                mode,
                batch_rows: 17,
                workspace: 128 * 1024,
                slow: Duration::ZERO,
                requests,
                pool_bytes,
            };
            let budget = AllocationBudget::new(pool_bytes);
            let outputs = tokio::time::timeout(
                Duration::from_secs(10),
                drain_all(
                    &dataset,
                    &config,
                    &budget,
                    expected.rows,
                    &expected.hash,
                    None,
                ),
            )
            .await
            .expect("finite dictionary consumers must progress with one CPU worker")
            .unwrap();
            assert_eq!(outputs.len(), requests);
            for output in outputs {
                assert_eq!(output.rows, expected.rows);
                assert_eq!(output.payload_bytes, expected.payload_bytes);
                assert_eq!(output.hash, expected.hash);
                assert!(output.batches > 1);
                assert!(output.shared_reserved_bytes_on_batch_max <= pool_bytes);
                assert_eq!(output.first_batch_seconds, None);
                assert_eq!(output.completed_seconds, None);
            }
            assert_eq!(budget.reserved(), 0);
        }
    }

    #[tokio::test]
    async fn dictionary_preparation_rejects_a_non_dictionary_tail() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("too-small.lance")
            .to_str()
            .unwrap()
            .to_string();
        // The first fragment qualifies; the final 17 rows are below the
        // writer's dictionary minimum. Every fragment must be checked.
        let error = prepare(Fixture::Dictionary, &uri, 145, 1024, 128)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("requires MiniBlock/LZ4"),
            "{error}"
        );
        let dataset = Dataset::open(&uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 145);
        // A failed layout qualification never licenses overwriting the fixture.
        let error = prepare(Fixture::Dictionary, &uri, 256, 1024, 128)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("already exists"), "{error}");
    }

    #[tokio::test]
    async fn dictionary_layout_check_rejects_fullzip() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("fullzip.lance")
            .to_str()
            .unwrap()
            .to_string();
        prepare(Fixture::FullZip, &uri, 128, 1024, 128)
            .await
            .unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        let error = verify_layout(&dataset).await.unwrap_err();
        assert!(
            error.to_string().contains("requires MiniBlock/LZ4"),
            "{error}"
        );
    }
}
