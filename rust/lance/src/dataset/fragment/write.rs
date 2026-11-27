// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Result, datatypes::Schema};
use lance_datafusion::utils::StreamingWriteSource;
use lance_table::format::Fragment;
use lance_table_runtime::fragment::FragmentCreateBuilder as RuntimeFragmentCreateBuilder;

use crate::dataset::WriteParams;
use crate::dataset::write::{DataFusionWriteSource, into_runtime_write_params};

pub(crate) use lance_table_runtime::fragment::generate_random_filename;

/// Builder for writing a new fragment.
///
/// This builder can be re-used to write multiple fragments.
pub struct FragmentCreateBuilder<'a> {
    dataset_uri: &'a str,
    schema: Option<&'a Schema>,
    write_params: Option<&'a WriteParams>,
}

impl<'a> FragmentCreateBuilder<'a> {
    pub fn new(dataset_uri: &'a str) -> Self {
        Self {
            dataset_uri,
            schema: None,
            write_params: None,
        }
    }

    /// Set the schema of the fragment. If it is not known, it will be inferred.
    ///
    /// If the schema isn't provided, but the `write_mode` is `WriteMode::Append`,
    /// the schema will be inferred from the existing dataset.
    ///
    /// If that fails, the schema will be inferred from the first batch.
    pub fn schema(mut self, schema: &'a Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the write parameters.
    pub fn write_params(mut self, params: &'a WriteParams) -> Self {
        self.write_params = Some(params);
        self
    }

    /// Write a fragment.
    pub async fn write(
        &self,
        source: impl StreamingWriteSource,
        id: Option<u64>,
    ) -> Result<Fragment> {
        let params = self.write_params.cloned().map(into_runtime_write_params);
        Box::pin(
            self.runtime_builder(params.as_ref())
                .write(DataFusionWriteSource(Box::new(source)), id),
        )
        .await
    }

    /// Write fragments split by the configured row count.
    pub async fn write_fragments(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Vec<Fragment>> {
        let params = self.write_params.cloned().map(into_runtime_write_params);
        Box::pin(
            self.runtime_builder(params.as_ref())
                .write_fragments(DataFusionWriteSource(Box::new(source))),
        )
        .await
    }

    fn runtime_builder<'b>(
        &'b self,
        params: Option<&'b lance_table_runtime::WriteParams>,
    ) -> RuntimeFragmentCreateBuilder<'b> {
        let mut builder = RuntimeFragmentCreateBuilder::new(self.dataset_uri);
        if let Some(schema) = self.schema {
            builder = builder.schema(schema);
        }
        if let Some(params) = params {
            builder = builder.write_params(params);
        }
        builder
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray, record_batch,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_arrow::SchemaExt;
    use lance_core::{
        Error,
        utils::tempfile::{TempDir, TempStrDir},
    };
    use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
    use lance_table::format::{BasePath, DataFile};
    use rstest::rstest;

    use super::*;
    use crate::Dataset;
    use crate::dataset::fragment::FileFragment;
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::{CommitBuilder, InsertBuilder, WriteMode};

    fn test_data() -> Box<dyn RecordBatchReader + Send> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int64, false),
            ArrowField::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        );
        Box::new(RecordBatchIterator::new(vec![batch], schema))
    }

    /// A dataset whose only registered base has the sparse id 2.
    async fn dataset_with_sparse_existing_base(
        primary: &TempStrDir,
        existing: &TempStrDir,
    ) -> Dataset {
        InsertBuilder::new(primary.as_str())
            .with_params(&WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![BasePath::new(
                    2,
                    existing.to_string(),
                    Some("existing".to_string()),
                    false,
                )]),
                ..Default::default()
            })
            .execute_stream(test_data())
            .await
            .unwrap()
    }

    fn assert_data_file_in(base: &TempStrDir, data_file: &DataFile) {
        let path = std::path::Path::new(base.as_str()).join(data_file.path.as_str());
        assert!(
            path.exists(),
            "data file was not written to the expected base: {}",
            path.display()
        );
    }

    #[tokio::test]
    async fn test_fragment_write_validation() {
        // Writing with empty schema produces an error
        let empty_schema = Arc::new(ArrowSchema::empty());
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], empty_schema));
        let tmp_dir = TempDir::default();
        let result = FragmentCreateBuilder::new(&tmp_dir.path_str())
            .write(empty_reader, None)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::InvalidInput { source, .. }
            if source.to_string().contains("Cannot write with an empty schema.")),
            "{:?}",
            result
        );

        // Writing empty reader produces an error
        let arrow_schema = test_data().schema();
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], arrow_schema.clone()));
        let result = FragmentCreateBuilder::new(tmp_dir.std_path().to_str().unwrap())
            .write(empty_reader, None)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::InvalidInput { source, .. }
            if source.to_string().contains("Input data was empty.")),
            "{:?}",
            result
        );

        // Writing with incorrect schema produces an error.
        let wrong_schema = arrow_schema
            .as_ref()
            .try_with_column(ArrowField::new("c", DataType::Utf8, false))
            .unwrap();
        let wrong_schema = Schema::try_from(&wrong_schema).unwrap();
        let result = FragmentCreateBuilder::new(tmp_dir.std_path().to_str().unwrap())
            .schema(&wrong_schema)
            .write(test_data(), None)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::SchemaMismatch { difference, .. }
            if difference.contains("fields did not match")),
            "{:?}",
            result
        );
    }

    /// The legacy writer is opened without the dataset writer, so the fragment
    /// builder checks the version itself, for one fragment and for many.
    #[rstest]
    #[tokio::test]
    async fn test_fragment_write_rejects_decimal32_before_2_3(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_2)] version: LanceFileVersion,
        #[values(false, true)] multi: bool,
    ) {
        let decimals = arrow_array::Decimal32Array::from(vec![1, -2])
            .with_precision_and_scale(9, 2)
            .unwrap();
        let batch =
            RecordBatch::try_from_iter([("d", Arc::new(decimals) as arrow_array::ArrayRef)])
                .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let tmp_dir = TempStrDir::default();
        let params = WriteParams {
            data_storage_version: Some(version),
            ..Default::default()
        };
        let builder = FragmentCreateBuilder::new(&tmp_dir).write_params(&params);
        let error = if multi {
            builder.write_fragments(reader).await.unwrap_err()
        } else {
            builder.write(reader, None).await.unwrap_err()
        };
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "{version:?}: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("require data storage version 2.3 or later"),
            "{version:?}: {error}"
        );
    }

    #[tokio::test]
    async fn test_fragment_write_default_schema() {
        // Infers schema and uses 0 as default field id
        let data = test_data();
        let tmp_dir = TempStrDir::default();
        let fragment = FragmentCreateBuilder::new(&tmp_dir)
            .write(data, None)
            .await
            .unwrap();

        // If unspecified, the fragment id should be 0.
        assert_eq!(fragment.id, 0);
        assert_eq!(fragment.deletion_file, None);
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(fragment.files[0].fields.as_ref(), &[0, 1]);
    }

    /// Blob v1 input written at 2.2 is stored as Blob v2, as `Dataset::write`
    /// does, so the commit that promotes the column never meets legacy pages
    /// in a 2.2 file.
    #[rstest]
    #[case::single_fragment(false)]
    #[case::multiple_fragments(true)]
    #[tokio::test]
    async fn test_legacy_blob_input_is_promoted_at_v2_2(#[case] multi: bool) {
        use arrow_array::{LargeBinaryArray, UInt32Array};

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::UInt32, false),
            ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(
                std::collections::HashMap::from([(
                    lance_arrow::BLOB_META_KEY.to_string(),
                    "true".to_string(),
                )]),
            ),
        ]));
        let reader = |ids: Vec<u32>, blobs: Vec<Option<&[u8]>>| {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(UInt32Array::from(ids)),
                    Arc::new(LargeBinaryArray::from(blobs)),
                ],
            )
            .unwrap();
            Box::new(RecordBatchIterator::new([Ok(batch)], schema.clone()))
                as Box<dyn RecordBatchReader + Send>
        };
        let test_dir = TempStrDir::default();
        let dataset = Dataset::write(
            reader(vec![0, 1], vec![Some(b"legacy".as_slice()), None]),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let builder = FragmentCreateBuilder::new(&test_dir).write_params(&params);
        let source = reader(vec![2, 3], vec![Some(b"fragment".as_slice()), Some(b"")]);
        let fragments = if multi {
            builder.write_fragments(source).await.unwrap()
        } else {
            vec![builder.write(source, Some(1)).await.unwrap()]
        };
        let data_file = &fragments[0].files[0];
        assert_eq!(data_file.file_version().unwrap(), ConcreteFileVersion::V2_2);
        let file_schema = crate::utils::test::data_file_schema(&dataset, &data_file.path).await;
        // The footer keeps the writer's view: a Blob v2 descriptor struct, where
        // the legacy layout keeps the large_binary input type.
        assert!(
            matches!(
                file_schema.field("blob").unwrap().data_type(),
                DataType::Struct(_)
            ),
            "{file_schema:?}"
        );

        let read_version = dataset.manifest.version;
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Append { fragments },
                None,
            ))
            .await
            .unwrap();
        let dataset = Arc::new(dataset);
        assert!(dataset.schema().field("blob").unwrap().is_blob_v2());
        let blobs = dataset
            .take_blobs_by_indices(&[0, 1, 2, 3], "blob")
            .await
            .unwrap();
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"legacy"
        );
        assert!(blobs[1].is_none());
        assert_eq!(
            blobs[2].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"fragment"
        );
        assert_eq!(blobs[3].as_ref().unwrap().size(), 0);
    }

    #[tokio::test]
    async fn test_fragment_write_with_schema() {
        // Uses provided schema. Field ids are correct in fragment metadata.
        let data = test_data();

        let arrow_schema = data.schema();
        let mut custom_schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        custom_schema.mut_field_by_id(0).unwrap().id = 3;
        custom_schema.mut_field_by_id(1).unwrap().id = 1;

        let tmp_dir = TempStrDir::default();
        let fragment = FragmentCreateBuilder::new(&tmp_dir)
            .schema(&custom_schema)
            .write(data, Some(42))
            .await
            .unwrap();

        assert_eq!(fragment.id, 42);
        assert_eq!(fragment.deletion_file, None);
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(fragment.files[0].fields.as_ref(), &[3, 1]);
        assert_eq!(fragment.files[0].column_indices.as_ref(), &[0, 1]);
    }

    #[tokio::test]
    async fn test_fragment_commit_rejects_system_column() {
        let tmp_dir = TempStrDir::default();
        let batch = record_batch!(("_rowid", UInt64, [1])).unwrap();
        let arrow_schema = batch.schema();
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let fragment = FragmentCreateBuilder::new(&tmp_dir)
            .write(RecordBatchIterator::new([Ok(batch)], arrow_schema), None)
            .await
            .unwrap();

        let transaction = Transaction::new(
            0,
            Operation::Overwrite {
                schema,
                fragments: vec![fragment],
                config_upsert_values: None,
                initial_bases: None,
            },
            None,
        );
        let error = CommitBuilder::new(tmp_dir.as_str())
            .execute(transaction)
            .await
            .unwrap_err();
        assert!(matches!(&error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("reserved name"), "{error}");
    }

    #[tokio::test]
    async fn test_fragment_create_with_session() {
        let session = Arc::new(crate::session::Session::new(
            0,
            1024 * 1024,
            Default::default(),
        ));
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };
        // Keep the dataset alive: the registry only holds weak references, so
        // the in-memory store survives through the dataset's strong reference.
        let initial_batch =
            record_batch!(("a", Int64, [1, 2, 3]), ("b", Utf8, ["a", "b", "c"])).unwrap();
        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute(vec![initial_batch])
            .await
            .unwrap();
        // Drop a column so the surviving field id is non-trivial (!= 0).
        dataset.drop_columns(&["a"]).await.unwrap();
        let field_id = dataset.schema().field("b").unwrap().id;
        assert_ne!(field_id, 0);

        let append_batch = record_batch!(("b", Utf8, ["d", "e"])).unwrap();
        let append_data =
            RecordBatchIterator::new([Ok(append_batch.clone())], append_batch.schema());

        let append_params = WriteParams {
            session: Some(session.clone()),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let fragment = FragmentCreateBuilder::new(dataset.uri())
            .write_params(&append_params)
            .write(append_data, None)
            .await
            .unwrap();

        assert_eq!(fragment.files[0].fields.as_ref(), &[field_id]);
        // The manifest load for schema inference went through the shared
        // session's metadata cache.
        assert!(session.metadata_cache_stats().await.num_entries > 0);
    }

    #[tokio::test]
    async fn test_write_fragments_validation() {
        // Writing with empty schema produces an error
        let empty_schema = Arc::new(ArrowSchema::empty());
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], empty_schema));
        let tmp_dir = TempDir::default();
        let result = FragmentCreateBuilder::new(&tmp_dir.path_str())
            .write_fragments(empty_reader)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::InvalidInput { source, .. }
            if source.to_string().contains("Cannot write with an empty schema.")),
            "{:?}",
            result
        );

        // Writing empty reader produces an error
        let arrow_schema = test_data().schema();
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], arrow_schema.clone()));
        let result = FragmentCreateBuilder::new(tmp_dir.std_path().to_str().unwrap())
            .write_fragments(empty_reader)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        // Writing with incorrect schema produces an error.
        let wrong_schema = arrow_schema
            .as_ref()
            .try_with_column(ArrowField::new("c", DataType::Utf8, false))
            .unwrap();
        let wrong_schema = Schema::try_from(&wrong_schema).unwrap();
        let result = FragmentCreateBuilder::new(tmp_dir.std_path().to_str().unwrap())
            .schema(&wrong_schema)
            .write_fragments(test_data())
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::SchemaMismatch { difference, .. }
            if difference.contains("fields did not match")),
            "{:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_write_fragments_default_schema() {
        // Infers schema and uses 0 as default field id
        let data = test_data();
        let tmp_dir = TempStrDir::default();
        let fragments = FragmentCreateBuilder::new(&tmp_dir)
            .write_fragments(data)
            .await
            .unwrap();

        // If unspecified, the fragment id should be 0.
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].deletion_file, None);
        assert_eq!(fragments[0].files.len(), 1);
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[0, 1]);
    }

    #[tokio::test]
    async fn test_write_fragments_with_options() {
        // Uses provided schema. Field ids are correct in fragment metadata.
        let data = test_data();
        let tmp_dir = TempStrDir::default();
        let writer_params = WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        };
        let fragments = FragmentCreateBuilder::new(&tmp_dir)
            .write_params(&writer_params)
            .write_fragments(data)
            .await
            .unwrap();

        assert_eq!(fragments.len(), 3);
        assert_eq!(fragments[0].deletion_file, None);
        assert_eq!(fragments[0].files.len(), 1);
        assert_eq!(fragments[0].files[0].column_indices.as_ref(), &[0, 1]);
        assert_eq!(fragments[1].deletion_file, None);
        assert_eq!(fragments[1].files.len(), 1);
        assert_eq!(fragments[1].files[0].column_indices.as_ref(), &[0, 1]);
        assert_eq!(fragments[2].deletion_file, None);
        assert_eq!(fragments[2].files.len(), 1);
        assert_eq!(fragments[2].files[0].column_indices.as_ref(), &[0, 1]);
    }

    #[tokio::test]
    async fn test_write_fragments_with_target_base() {
        let primary = TempStrDir::default();
        let base1 = TempStrDir::default();
        let base2 = TempStrDir::default();
        let create_params = WriteParams::default()
            .with_initial_bases(vec![
                BasePath::new(0, base1.to_string(), Some("base1".to_string()), false),
                BasePath::new(0, base2.to_string(), Some("base2".to_string()), false),
            ])
            .with_target_base_names_or_paths(vec!["base1".to_string()]);

        let dataset = InsertBuilder::new(primary.as_str())
            .with_params(&create_params)
            .execute_stream(test_data())
            .await
            .unwrap();

        let append_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }
        .with_target_base_names_or_paths(vec!["base2".to_string()]);
        let fragments = FragmentCreateBuilder::new(dataset.table_state.uri.as_str())
            .write_params(&append_params)
            .write_fragments(test_data())
            .await
            .unwrap();

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].files[0].base_id, Some(2));
    }

    #[tokio::test]
    async fn test_write_fragments_with_target_all_bases() {
        let primary = TempStrDir::default();
        let base1 = TempStrDir::default();
        let base2 = TempStrDir::default();
        let create_params = WriteParams::default().with_initial_bases(vec![
            BasePath::new(0, base1.to_string(), Some("base1".to_string()), false),
            BasePath::new(0, base2.to_string(), Some("base2".to_string()), false),
        ]);

        let dataset = InsertBuilder::new(primary.as_str())
            .with_params(&create_params)
            .execute_stream(test_data())
            .await
            .unwrap();

        // Without primary, the first slot is the lowest registered base id.
        let append_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }
        .with_target_all_bases(false);
        let fragments = FragmentCreateBuilder::new(dataset.table_state.uri.as_str())
            .write_params(&append_params)
            .write_fragments(test_data())
            .await
            .unwrap();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].files[0].base_id, Some(1));

        // With primary included, the first slot is primary storage.
        let append_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }
        .with_target_all_bases(true);
        let fragments = FragmentCreateBuilder::new(dataset.table_state.uri.as_str())
            .write_params(&append_params)
            .write_fragments(test_data())
            .await
            .unwrap();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].files[0].base_id, None);
    }

    #[rstest]
    #[case::v1(LanceFileVersion::Legacy)]
    #[case::v2(LanceFileVersion::V2_2)]
    #[tokio::test]
    async fn test_create_fragments_preserves_sparse_base_ids_for_numeric_target(
        #[case] data_storage_version: LanceFileVersion,
    ) {
        let primary = TempStrDir::default();
        let existing = TempStrDir::default();
        let base_a = TempStrDir::default();
        let base_b = TempStrDir::default();
        let dataset = dataset_with_sparse_existing_base(&primary, &existing).await;

        let params = WriteParams {
            mode: WriteMode::Create,
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1,
            max_rows_per_group: 1,
            initial_bases: Some(vec![
                BasePath::new(0, base_a.to_string(), Some("base-a".to_string()), false),
                BasePath::new(0, base_b.to_string(), Some("base-b".to_string()), false),
            ]),
            target_bases: Some(vec![3]),
            ..Default::default()
        };
        let fragments = FileFragment::create_fragments(
            dataset.table_state.uri.as_str(),
            test_data(),
            Some(params),
        )
        .await
        .unwrap();

        assert_eq!(fragments.len(), 3);
        for fragment in &fragments {
            let data_file = &fragment.files[0];
            // Existing base 2 makes the first automatic base 3, not a lower
            // free id. Numeric target 3 must therefore stay bound to base A.
            assert_eq!(data_file.base_id, Some(3));
            assert_data_file_in(&base_a, data_file);
        }
    }

    #[rstest]
    #[case::v1(LanceFileVersion::Legacy)]
    #[case::v2(LanceFileVersion::V2_2)]
    #[tokio::test]
    async fn test_create_fragments_all_bases_starts_with_sparse_existing_base(
        #[case] data_storage_version: LanceFileVersion,
    ) {
        let primary = TempStrDir::default();
        let existing = TempStrDir::default();
        let base_a = TempStrDir::default();
        let base_b = TempStrDir::default();
        let dataset = dataset_with_sparse_existing_base(&primary, &existing).await;

        let params = WriteParams {
            mode: WriteMode::Create,
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1,
            max_rows_per_group: 1,
            initial_bases: Some(vec![
                BasePath::new(0, base_a.to_string(), Some("base-a".to_string()), false),
                BasePath::new(0, base_b.to_string(), Some("base-b".to_string()), false),
            ]),
            target_all_bases: Some(false),
            ..Default::default()
        };
        let fragments = FileFragment::create_fragments(
            dataset.table_state.uri.as_str(),
            test_data(),
            Some(params),
        )
        .await
        .unwrap();

        assert_eq!(fragments.len(), 3);
        let expected = [(Some(2), &existing), (Some(3), &base_a), (Some(4), &base_b)];
        for (fragment, (expected_base_id, base)) in fragments.iter().zip(expected) {
            let data_file = &fragment.files[0];
            // Rotation starts at the existing base and keeps the automatic
            // bases in allocation order after it.
            assert_eq!(data_file.base_id, expected_base_id);
            assert_data_file_in(base, data_file);
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_with_format_version(
        #[values(
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::Legacy,
            LanceFileVersion::Stable
        )]
        file_version: LanceFileVersion,
    ) {
        let data = test_data();
        let tmp_dir = TempStrDir::default();
        let writer_params = WriteParams {
            data_storage_version: Some(file_version),
            ..Default::default()
        };
        let fragment = FragmentCreateBuilder::new(&tmp_dir)
            .write_params(&writer_params)
            .write(data, None)
            .await
            .unwrap();

        assert!(!fragment.files.is_empty());
        fragment.files.iter().for_each(|f| {
            let (major_version, minor_version) = file_version.resolve().to_data_file_numbers();
            assert_eq!(f.file_major_version, major_version);
            assert_eq!(f.file_minor_version, minor_version);
        })
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_fragments_with_format_version(
        #[values(
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::Legacy,
            LanceFileVersion::Stable
        )]
        file_version: LanceFileVersion,
    ) {
        let data = test_data();
        let tmp_dir = TempStrDir::default();
        let writer_params = WriteParams {
            data_storage_version: Some(file_version),
            ..Default::default()
        };
        let fragment = FragmentCreateBuilder::new(&tmp_dir)
            .write_params(&writer_params)
            .write_fragments(data)
            .await
            .unwrap();

        assert!(!fragment.is_empty());
        fragment[0].files.iter().for_each(|f| {
            let (major_version, minor_version) = file_version.resolve().to_data_file_numbers();
            assert_eq!(f.file_major_version, major_version);
            assert_eq!(f.file_minor_version, minor_version);
        })
    }

    /// A zero row limit used to panic on V2. On V1 a zero group size wrote
    /// nothing, and a zero file limit was ignored by `write`.
    #[rstest]
    #[case::v2_rows_per_file(LanceFileVersion::Stable, 0, 1024, "max_rows_per_file")]
    #[case::v1_rows_per_file(LanceFileVersion::Legacy, 0, 1024, "max_rows_per_file")]
    #[case::v1_rows_per_group(LanceFileVersion::Legacy, 1024, 0, "max_rows_per_group")]
    #[tokio::test]
    async fn test_fragment_write_rejects_zero_row_limits(
        #[case] file_version: LanceFileVersion,
        #[case] max_rows_per_file: usize,
        #[case] max_rows_per_group: usize,
        #[case] param: &str,
    ) {
        let tmp_dir = TempStrDir::default();
        let params = WriteParams {
            data_storage_version: Some(file_version),
            max_rows_per_file,
            max_rows_per_group,
            ..Default::default()
        };
        let builder = FragmentCreateBuilder::new(&tmp_dir).write_params(&params);
        let errors = [
            builder.write(test_data(), None).await.unwrap_err(),
            builder.write_fragments(test_data()).await.unwrap_err(),
        ];
        for err in errors {
            assert!(
                matches!(&err, Error::InvalidInput { source, .. }
                if source.to_string().contains(&format!("{param} must be greater than zero"))),
                "{err:?}"
            );
        }
    }

    #[test]
    fn test_binary_filename_generation() {
        use std::collections::HashSet;

        // Test format and uniqueness
        let mut filenames = HashSet::new();
        for _ in 0..100 {
            let filename = generate_random_filename();

            // Should be 50 characters: 24 binary + 26 hex
            assert_eq!(filename.len(), 50, "Filename should be 50 characters");

            // First 24 should be binary
            let binary_part = &filename[0..24];
            assert!(
                binary_part.chars().all(|c| c == '0' || c == '1'),
                "First 24 chars should be binary: {}",
                binary_part
            );

            // Last 26 should be hex
            let hex_part = &filename[24..];
            assert_eq!(hex_part.len(), 26, "Hex part should be 26 characters");
            assert!(
                hex_part.chars().all(|c| c.is_ascii_hexdigit()),
                "Last 26 chars should be hex: {}",
                hex_part
            );

            // Should be unique
            assert!(filenames.insert(filename.clone()));
        }
    }
}
