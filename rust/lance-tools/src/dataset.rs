// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::Arc;

use futures::{StreamExt, stream};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::write::CommitBuilder;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_file::version::ConcreteFileVersion;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

use crate::cli::{
    LanceDatasetLocateArgs, LanceDatasetRepairArgs, LanceDatasetRestoreArgs,
    LanceDatasetSourceArgs, LanceDatasetVerifyArgs,
};

async fn open_dataset(args: &LanceDatasetSourceArgs, version: Option<u64>) -> Result<Dataset> {
    let mut builder = DatasetBuilder::from_uri(&args.source);
    if let Some(options) = &args.storage_options {
        let options =
            serde_json::from_str::<HashMap<String, String>>(options).map_err(|error| {
                Error::invalid_input(format!(
                    "storage-options must be a JSON object with string values: {error}"
                ))
            })?;
        builder = builder.with_storage_options(options);
    }
    if let Some(version) = version {
        builder = builder.with_version(version);
    }
    builder.load().await
}

// Exact identities avoid accidentally removing files with similar suffixes.
fn file_matches(manifest_path: &str, object_path: &Path, query: &str) -> bool {
    let query = query.trim_start_matches('/');
    query == manifest_path
        || query.strip_prefix("data/") == Some(manifest_path)
        || query == object_path.as_ref()
}

pub(crate) async fn verify_data_files(
    mut writer: impl Write,
    args: &LanceDatasetVerifyArgs,
) -> Result<()> {
    let dataset = open_dataset(&args.dataset, args.version).await?;
    writeln!(writer, "version: {}", dataset.version_id())?;
    let files = dataset
        .iter_fragments()
        .flat_map(|fragment| fragment.files.iter().map(move |file| (fragment, file)));
    // Limit concurrent metadata reads as well as requests: each scheduler has
    // its own memory budget, independent of the object's store concurrency.
    let concurrency = dataset.object_store(None).await?.io_parallelism().min(16);
    let mut checks = stream::iter(files)
        .map(|(fragment, file)| {
            let dataset = &dataset;
            async move {
                let result = async {
                    let (store, path) = dataset.resolve_data_file_location(file).await?;
                    let size = store.size(&path).await?;
                    if let Some(expected) = file.file_size_bytes.get()
                        && size != expected.get()
                    {
                        return Err(Error::corrupt_file(
                            path,
                            format!("size is {size} bytes; manifest records {expected} bytes"),
                        ));
                    }
                    if args.deep {
                        if file.file_version()? == ConcreteFileVersion::V1 {
                            return Err(Error::not_supported(
                                "deep verification requires a current-format data file",
                            ));
                        }
                        let scheduler =
                            ScanScheduler::new(store, SchedulerConfig::new(4 * 1024 * 1024));
                        let reader = scheduler
                            .open_file(&path, &CachedFileSize::new(size))
                            .await?;
                        let metadata = FileReader::read_all_metadata(&reader).await?;
                        if let Some(expected) = fragment.physical_rows
                            && metadata.num_rows != expected as u64
                        {
                            return Err(Error::corrupt_file(
                                path,
                                format!(
                                    "row count is {}; fragment records {expected} rows",
                                    metadata.num_rows
                                ),
                            ));
                        }
                    }
                    Ok(())
                }
                .await;
                (fragment.id, file, result)
            }
        })
        .buffered(concurrency);

    let mut checked = 0_usize;
    let mut failed = 0_usize;
    while let Some((fragment_id, file, result)) = checks.next().await {
        checked += 1;
        if let Err(error) = result {
            failed += 1;
            let status = if error.is_not_found() {
                "missing"
            } else {
                "error"
            };
            writeln!(
                writer,
                "{status} fragment={fragment_id} path={} base_id={} detail={error}",
                file.path,
                file.base_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "default".to_string())
            )?;
        }
    }
    writeln!(writer, "checked_data_files: {checked}")?;
    writeln!(writer, "bad_data_files: {failed}")?;
    if args.fail && failed > 0 {
        return Err(Error::invalid_input(format!(
            "{failed} of {checked} data files failed verification in version {}",
            dataset.version_id()
        )));
    }
    Ok(())
}

pub(crate) async fn repair_manifest(
    mut writer: impl Write,
    args: &LanceDatasetRepairArgs,
) -> Result<()> {
    let dataset = open_dataset(&args.dataset, None).await?;
    let mut matched = vec![false; args.remove_data_files.len()];
    let mut fragment_ids = BTreeSet::new();
    for fragment in dataset.iter_fragments() {
        for file in &fragment.files {
            let (_, path) = dataset.resolve_data_file_location(file).await?;
            for (query, matched) in args.remove_data_files.iter().zip(&mut matched) {
                if file_matches(&file.path, &path, query) {
                    *matched = true;
                    fragment_ids.insert(fragment.id);
                }
            }
        }
    }
    for (query, matched) in args.remove_data_files.iter().zip(matched) {
        if !matched {
            return Err(Error::invalid_input(format!(
                "data file {query:?} was not found in dataset version {}",
                dataset.version_id()
            )));
        }
    }

    writeln!(writer, "read_version: {}", dataset.version_id())?;
    for fragment in dataset
        .iter_fragments()
        .filter(|fragment| fragment_ids.contains(&fragment.id))
    {
        writeln!(
            writer,
            "remove fragment={} rows={}",
            fragment.id,
            fragment
                .num_rows()
                .map(|rows| rows.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )?;
        for file in &fragment.files {
            writeln!(
                writer,
                "  path={} base_id={}",
                file.path,
                file.base_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "default".to_string())
            )?;
        }
    }
    writeln!(writer, "affected_fragments: {}", fragment_ids.len())?;
    if args.dry_run {
        writeln!(writer, "dry_run: true")?;
        return Ok(());
    }

    let transaction = Transaction::new(
        dataset.version_id(),
        Operation::Delete {
            updated_fragments: vec![],
            deleted_fragment_ids: fragment_ids.into_iter().collect(),
            // The selection is explicit, not a predicate evaluated on damaged data.
            predicate: String::new(),
        },
        None,
    );
    let committed = CommitBuilder::new(Arc::new(dataset))
        // Preserve recovery options even when this version would trigger cleanup.
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    writeln!(writer, "new_version: {}", committed.version_id())?;
    Ok(())
}

pub(crate) async fn locate_data_file(
    mut writer: impl Write,
    args: &LanceDatasetLocateArgs,
) -> Result<()> {
    let dataset = open_dataset(&args.dataset, None).await?;
    let mut versions = dataset.versions().await?;
    versions.sort_by_key(|version| version.version);
    let mut previous_version = None;
    for version in versions {
        let historical = dataset.checkout_version(version.version).await?;
        for fragment in historical.iter_fragments() {
            for file in &fragment.files {
                let (_, path) = historical.resolve_data_file_location(file).await?;
                if file_matches(&file.path, &path, &args.data_file) {
                    writeln!(writer, "first_manifest_version: {}", version.version)?;
                    match previous_version {
                        Some(previous) => writeln!(writer, "previous_version: {previous}")?,
                        None => writeln!(writer, "previous_version: none")?,
                    }
                    writeln!(writer, "fragment={} path={path}", fragment.id)?;
                    return Ok(());
                }
            }
        }
        previous_version = Some(version.version);
    }
    Err(Error::invalid_input(format!(
        "data file {:?} was not found in retained dataset versions",
        args.data_file
    )))
}

pub(crate) async fn restore(mut writer: impl Write, args: &LanceDatasetRestoreArgs) -> Result<()> {
    let mut dataset = open_dataset(&args.dataset, Some(args.version)).await?;
    dataset.restore().await?;
    writeln!(writer, "restored_version: {}", args.version)?;
    writeln!(writer, "new_version: {}", dataset.version_id())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use clap::Parser;
    use lance::dataset::{NewColumnTransform, WriteMode, WriteParams};
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use rstest::rstest;
    use tempfile::TempDir;

    use crate::cli::LanceToolsArgs;

    async fn two_version_dataset() -> (TempDir, Dataset) {
        let dir = tempfile::tempdir().unwrap();
        let reader = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(2), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        let reader = gen_batch()
            .col("id", array::step_custom::<Int32Type>(2, 1))
            .into_reader_rows(RowCount::from(2), BatchCount::from(1));
        dataset.append(reader, None).await.unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        (dir, dataset)
    }

    // Exercise parsing and dispatch as well as the operation itself.
    async fn run_command(command: &str, source: &str, args: &[&str]) -> Result<String> {
        let args = LanceToolsArgs::try_parse_from(
            ["lance-tools", "dataset", command, "-s", source]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .unwrap();
        let mut output = Vec::new();
        args.run(&mut output).await?;
        Ok(String::from_utf8(output).unwrap())
    }

    #[derive(Clone, Copy)]
    enum Damage {
        Missing,
        Truncated,
        Footer,
    }

    #[rstest]
    #[case::missing(Damage::Missing)]
    #[case::truncated(Damage::Truncated)]
    #[case::footer(Damage::Footer)]
    #[tokio::test]
    async fn test_verify_and_repair_multifile_fragment(#[case] damage: Damage) {
        let (_dir, mut dataset) = two_version_dataset().await;
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("copy".into(), "id + 10".into())]),
                None,
                None,
            )
            .await
            .unwrap();
        let source = dataset.uri();
        assert_eq!(dataset.fragments()[1].files.len(), 2);
        let file = &dataset.fragments()[1].files[1];
        let (store, path) = dataset.resolve_data_file_location(file).await.unwrap();
        let original = store.read_one_all(&path).await.unwrap();
        match damage {
            Damage::Missing => store.delete(&path).await.unwrap(),
            Damage::Truncated => {
                store.put(&path, &original[..8]).await.unwrap();
            }
            Damage::Footer => {
                let mut bytes = original.to_vec();
                let len = bytes.len();
                bytes[len - 4..].copy_from_slice(b"oops");
                store.put(&path, &bytes).await.unwrap();
            }
        }
        let damaged = Dataset::open(source).await.unwrap();
        assert!(damaged.scan().try_into_batch().await.is_err());
        let shallow = run_command("verify-data-files", source, &[]).await.unwrap();
        let expected_bad = if matches!(damage, Damage::Footer) {
            0
        } else {
            1
        };
        assert!(shallow.contains(&format!("bad_data_files: {expected_bad}")));
        let verification = run_command("verify-data-files", source, &["--deep"])
            .await
            .unwrap();
        assert!(verification.contains("checked_data_files: 4"));
        assert!(verification.contains("bad_data_files: 1"));
        assert!(verification.contains(&format!("fragment=1 path={}", file.path)));
        if matches!(damage, Damage::Missing) {
            assert!(verification.contains("missing fragment=1"));
        }
        let error = run_command("verify-data-files", source, &["--deep", "--fail"])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("1 of 4 data files failed"));

        // Match an object key for dry-run, then a data/ path for the commit.
        let preview = run_command(
            "repair-manifest",
            source,
            &["--remove-data-files", path.as_ref(), "--dry-run"],
        )
        .await
        .unwrap();
        assert!(preview.contains("affected_fragments: 1"));
        for file in &dataset.fragments()[1].files {
            assert!(preview.contains(&file.path));
        }
        assert_eq!(Dataset::open(source).await.unwrap().version_id(), 3);
        let output = run_command(
            "repair-manifest",
            source,
            &["--remove-data-files", &format!("data/{}", file.path)],
        )
        .await
        .unwrap();
        assert!(output.contains("new_version: 4"));
        let repaired = Dataset::open(source).await.unwrap();
        assert_eq!(repaired.fragments().len(), 1);
        let batch = repaired.scan().try_into_batch().await.unwrap();
        assert_eq!(batch["id"].as_primitive::<Int32Type>().values(), &[0, 1]);
        assert_eq!(
            batch["copy"].as_primitive::<Int32Type>().values(),
            &[10, 11]
        );
        repaired.validate().await.unwrap();
        assert_eq!(
            repaired
                .checkout_version(3)
                .await
                .unwrap()
                .fragments()
                .len(),
            2
        );
        assert_eq!(
            store.exists(&path).await.unwrap(),
            !matches!(damage, Damage::Missing)
        );
    }

    #[tokio::test]
    async fn test_repair_preserves_history_with_auto_cleanup() {
        let (_dir, dataset) = two_version_dataset().await;
        let (store, historical_path) = dataset
            .resolve_data_file_location(&dataset.fragments()[0].files[0])
            .await
            .unwrap();
        let reader = gen_batch()
            .col("id", array::step_custom::<Int32Type>(100, 1))
            .into_reader_rows(RowCount::from(4), BatchCount::from(1));
        let mut dataset = Dataset::write(
            reader,
            dataset.uri(),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .update_config([
                ("lance.auto_cleanup.interval", "5"),
                ("lance.auto_cleanup.retain_versions", "1"),
            ])
            .await
            .unwrap();
        assert_eq!(dataset.version_id(), 4);
        assert_eq!(dataset.versions().await.unwrap().len(), 4);
        assert!(store.exists(&historical_path).await.unwrap());

        let damaged = &dataset.fragments()[1].files[0];
        let (damaged_store, damaged_path) =
            dataset.resolve_data_file_location(damaged).await.unwrap();
        damaged_store
            .put(&damaged_path, b"truncated")
            .await
            .unwrap();
        run_command(
            "repair-manifest",
            dataset.uri(),
            &["--remove-data-files", &damaged.path],
        )
        .await
        .unwrap();

        let repaired = Dataset::open(dataset.uri()).await.unwrap();
        assert_eq!(repaired.version_id(), 5);
        let versions: Vec<_> = repaired
            .versions()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect();
        assert_eq!(versions, vec![1, 2, 3, 4, 5]);
        assert!(store.exists(&historical_path).await.unwrap());
        let historical = repaired.checkout_version(2).await.unwrap();
        assert_eq!(
            historical.scan().try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values(),
            &[0, 1, 2, 3]
        );
        assert_eq!(
            repaired.scan().try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values(),
            &[100, 101]
        );
    }

    #[rstest]
    #[case::unknown("unknown.lance")]
    #[case::suffix("lance")]
    #[case::longer("wrong/path/file.lance")]
    #[tokio::test]
    async fn test_repair_rejects_unmatched_selection_atomically(#[case] unmatched: &str) {
        let (_dir, dataset) = two_version_dataset().await;
        let valid = &dataset.fragments()[1].files[0].path;
        let error = run_command(
            "repair-manifest",
            dataset.uri(),
            &["--remove-data-files", &format!("{valid},{unmatched}")],
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("was not found in dataset version 2")
        );
        assert_eq!(Dataset::open(dataset.uri()).await.unwrap().version_id(), 2);
    }

    #[tokio::test]
    async fn test_repair_can_remove_all_fragments_with_repeated_selections() {
        let (_dir, dataset) = two_version_dataset().await;
        let first = &dataset.fragments()[0].files[0].path;
        let second = &dataset.fragments()[1].files[0].path;
        let output = run_command(
            "repair-manifest",
            dataset.uri(),
            &[
                "--remove-data-files",
                &format!("{first},{second}"),
                "--remove-data-files",
                first,
            ],
        )
        .await
        .unwrap();
        assert!(output.contains("affected_fragments: 2"));
        let repaired = Dataset::open(dataset.uri()).await.unwrap();
        assert_eq!(repaired.version_id(), 3);
        assert!(repaired.fragments().is_empty());
        assert_eq!(repaired.count_rows(None).await.unwrap(), 0);
        assert_eq!(repaired.schema(), dataset.schema());
        repaired.validate().await.unwrap();
    }

    #[tokio::test]
    async fn test_locate_and_restore_before_missing_file() {
        let (_dir, dataset) = two_version_dataset().await;
        let file = &dataset.fragments()[1].files[0];
        let (store, path) = dataset.resolve_data_file_location(file).await.unwrap();
        store.delete(&path).await.unwrap();
        let located = run_command(
            "locate-data-file",
            dataset.uri(),
            &["--data-file", &file.path],
        )
        .await
        .unwrap();
        assert!(located.contains("first_manifest_version: 2"));
        assert!(located.contains("previous_version: 1"));
        let output = run_command("restore", dataset.uri(), &["--version", "1"])
            .await
            .unwrap();
        assert!(output.contains("restored_version: 1\nnew_version: 3"));
        let restored = Dataset::open(dataset.uri()).await.unwrap();
        let batch = restored.scan().try_into_batch().await.unwrap();
        assert_eq!(batch["id"].as_primitive::<Int32Type>().values(), &[0, 1]);
        // File lookup includes retained history even after the bad version is restored away.
        let located_again = run_command(
            "locate-data-file",
            dataset.uri(),
            &["--data-file", &file.path],
        )
        .await
        .unwrap();
        assert_eq!(located_again, located);
    }

    #[tokio::test]
    async fn test_verify_shallow_clone_resolves_source_base() {
        let (_dir, mut dataset) = two_version_dataset().await;
        let clone_dir = tempfile::tempdir().unwrap();
        let cloned = dataset
            .shallow_clone(clone_dir.path().to_str().unwrap(), 2, None)
            .await
            .unwrap();
        assert!(cloned.fragments()[0].files[0].base_id.is_some());
        let output = run_command("verify-data-files", cloned.uri(), &["--deep", "--fail"])
            .await
            .unwrap();
        assert!(output.contains("checked_data_files: 2\nbad_data_files: 0"));
        let file = &cloned.fragments()[1].files[0];
        let (store, path) = cloned.resolve_data_file_location(file).await.unwrap();
        store.delete(&path).await.unwrap();
        let output = run_command("verify-data-files", cloned.uri(), &[])
            .await
            .unwrap();
        assert!(output.contains("missing fragment=1"));
        run_command(
            "repair-manifest",
            cloned.uri(),
            &["--remove-data-files", &file.path],
        )
        .await
        .unwrap();
        let repaired = Dataset::open(cloned.uri()).await.unwrap();
        assert_eq!(
            repaired.scan().try_into_batch().await.unwrap().num_rows(),
            2
        );
        assert_eq!(Dataset::open(dataset.uri()).await.unwrap().version_id(), 2);
    }

    #[tokio::test]
    async fn test_verify_continues_after_errors_and_checks_historical_versions() {
        let (_dir, dataset) = two_version_dataset().await;
        let output = run_command(
            "verify-data-files",
            dataset.uri(),
            &["--version", "1", "--deep", "--fail"],
        )
        .await
        .unwrap();
        assert!(output.contains("version: 1\nchecked_data_files: 1\nbad_data_files: 0"));
        for fragment in dataset.iter_fragments() {
            let (store, path) = dataset
                .resolve_data_file_location(&fragment.files[0])
                .await
                .unwrap();
            store.delete(&path).await.unwrap();
        }
        let output = run_command("verify-data-files", dataset.uri(), &[])
            .await
            .unwrap();
        assert!(output.contains("missing fragment=0"));
        assert!(output.contains("missing fragment=1"));
        assert!(output.contains("checked_data_files: 2\nbad_data_files: 2"));
    }

    #[test]
    fn test_file_matching_is_exact() {
        let path = Path::from("root/data/a.lance");
        assert!(file_matches("a.lance", &path, "a.lance"));
        assert!(file_matches("a.lance", &path, "data/a.lance"));
        assert!(file_matches("a.lance", &path, "/root/data/a.lance"));
        assert!(!file_matches("a.lance", &path, "lance"));
        assert!(!file_matches("a.lance", &path, "wrong/data/a.lance"));
        assert!(!file_matches("a.lance", &path, ""));
    }

    #[rstest]
    #[case::empty_file(vec!["repair-manifest", "--remove-data-files", ""])]
    #[case::empty_csv_entry(vec!["repair-manifest", "--remove-data-files", "a.lance,"])]
    #[case::no_action(vec!["repair-manifest"])]
    #[case::no_restore_version(vec!["restore"])]
    fn test_repair_cli_rejects_incomplete_actions(#[case] args: Vec<&str>) {
        let error = LanceToolsArgs::try_parse_from(
            ["lance-tools", "dataset"]
                .into_iter()
                .chain(args)
                .chain(["-s", "test.lance"]),
        )
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            clap::error::ErrorKind::InvalidValue | clap::error::ErrorKind::MissingRequiredArgument
        ));
    }

    #[tokio::test]
    async fn test_storage_options_reject_non_string_values() {
        let error = run_command(
            "verify-data-files",
            "memory://",
            &["--storage-options", r#"{"region":1}"#],
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("storage-options must be a JSON object with string values")
        );
    }
}
