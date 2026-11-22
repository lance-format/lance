// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, fs, path::Path};

use arrow_array::{RecordBatchIterator, record_batch};
use lance::{Dataset, index::DatasetIndexExt};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::{
    IndexType,
    optimize::OptimizeOptions,
    scalar::{FullTextSearchQuery, InvertedIndexParams, inverted::METADATA_FILE},
};

#[tokio::test]
async fn append_partitioned_fts_without_reading_old_partitions() {
    let dir = TempStrDir::default();
    let row = |name: &str| {
        let batch = record_batch!(("name", Utf8, [name])).unwrap();
        let schema = batch.schema();
        RecordBatchIterator::new(vec![Ok(batch)], schema)
    };
    let mut dataset = Dataset::write(row("FirstName"), dir.as_str(), None)
        .await
        .unwrap();
    dataset
        .create_index(
            &["name"],
            IndexType::Inverted,
            Some("names".into()),
            &InvertedIndexParams::default().lower_case(false).stem(false),
            false,
        )
        .await
        .unwrap();
    dataset.append(row("SecondName"), None).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let old_indices = dataset.load_indices().await.unwrap();
    assert_eq!(old_indices.len(), 2);
    dataset.append(row("ThirdName"), None).await.unwrap();
    drop(dataset);

    // A fresh session cannot satisfy a partition read from its index cache.
    let mut hidden = Vec::new();
    for index in old_indices.iter() {
        let root = Path::new(dir.as_str())
            .join("_indices")
            .join(index.uuid.to_string());
        for file in index.files.as_ref().unwrap() {
            if file.path != METADATA_FILE {
                let path = root.join(&file.path);
                let away = path.with_extension("hidden");
                fs::rename(&path, &away).unwrap();
                hidden.push((path, away));
            }
        }
    }
    assert!(!hidden.is_empty());
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    for (path, away) in hidden {
        fs::rename(away, path).unwrap();
    }

    let indices = dataset.load_indices().await.unwrap();
    let ids: HashSet<_> = indices.iter().map(|index| index.uuid).collect();
    assert_eq!(ids.len(), 3);
    assert!(old_indices.iter().all(|index| ids.contains(&index.uuid)));
    let new = indices
        .iter()
        .find(|index| !old_indices.iter().any(|old| old.uuid == index.uuid))
        .unwrap();
    assert_eq!(
        new.fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![2]
    );
    for (query, expected) in [
        ("FirstName", 1),
        ("SecondName", 1),
        ("ThirdName", 1),
        ("thirdname", 0),
    ] {
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new(query.to_owned()))
            .unwrap();
        assert_eq!(scan.count_rows().await.unwrap(), expected, "{query}");
    }
}
