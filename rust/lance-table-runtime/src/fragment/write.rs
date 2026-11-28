// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::Schema as ArrowSchema;
use futures::StreamExt;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_file::version::{ConcreteFileVersion, stable_file_version};
use lance_file::versions::v1::writer::FileWriter as V1FileWriter;
use lance_io::object_store::ObjectStore;
use lance_io::stream::chunker::{break_stream, chunk_stream};
use lance_io::stream::{SendableRecordBatchStream, StreamingWriteSource};
use lance_table::format::{DataFile, Fragment};
use lance_table::io::manifest::ManifestDescribing;
use std::borrow::Cow;
use uuid::Uuid;

use crate::paths::DATA_DIR;
use crate::write::schema::{promote_legacy_blob_schema, validate_decimal32_and_decimal64};
use crate::write::{
    SeedCollector, WriteMode, WriteParams, WriterOptions, blob_v2_external_base_resolver,
    open_writer, validate_and_resolve_target_bases_with_primary, validate_row_limits,
    write_fragments_direct, writes_blob_v2,
};
use crate::{Table, TableBuilder};

/// Generates a filename optimized for S3 throughput using a UUID-based approach.
///
/// This approach follows Apache Iceberg's ObjectStoreLocationProvider pattern:
/// - Takes a UUID (16 bytes total)
/// - Uses first 3 bytes (24 bits) as binary string prefix for S3 distribution
/// - Uses remaining 13 bytes as hex string for uniqueness
///
/// Format: `<24-bit-binary><remaining-hex>`
/// Example: "101100101101010011010110a1b2c3d4e5f6g7h8i9j0"
///
/// We use binary instead of hex for the prefix because it helps S3 scale up on the
/// prefix faster with fewer throttling and retries. Binary provides maximum entropy per character
/// (1 bit) compared to hex (4 bits), allowing S3's internal partitioning to more
/// quickly recognize the access pattern and scale appropriately.
///
/// The binary prefix ensures files are distributed evenly across S3 prefixes,
/// minimizing throttling and maximizing throughput, while maintaining uniqueness.
#[doc(hidden)]
pub fn generate_random_filename() -> String {
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();

    let mut out = String::with_capacity(50);

    // Convert first 3 bytes to binary string (24 bits)
    for &b in &bytes[..3] {
        for i in (0..8).rev() {
            out.push(if (b >> i) & 1 == 1 { '1' } else { '0' });
        }
    }

    // Convert remaining 13 bytes to hex string (26 chars)
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in &bytes[3..] {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }

    out
}

/// Store blob v1 input as Blob v2 from file version 2.2 on, as `Dataset::write`
/// does, so the commit that promotes the column never meets legacy pages in a
/// 2.2 file. Below 2.2 the legacy layout is the only one.
fn promote_legacy_blobs(version: ConcreteFileVersion, schema: Schema) -> Result<Schema> {
    if writes_blob_v2(version) {
        promote_legacy_blob_schema(&schema)
    } else {
        Ok(schema)
    }
}

/// Builder for writing a new fragment.
///
/// This builder can be re-used to write multiple fragments.
pub struct FragmentCreateBuilder<'a> {
    dataset_uri: &'a str,
    schema: Option<&'a Schema>,
    write_params: Option<&'a WriteParams>,
}

impl<'a> FragmentCreateBuilder<'a> {
    /// Create a fragment writer for the given table URI.
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
        let (stream, schema) = self.get_stream_and_schema(source).await?;
        let version = self
            .write_params
            .map(|params| params.storage_version_or_default())
            .unwrap_or_else(stable_file_version);
        if let Some(params) = self.write_params {
            validate_row_limits(version, params)?;
        }
        let id = id.unwrap_or_default();
        // The legacy writer below is opened directly, not through `open_writer`.
        validate_decimal32_and_decimal64(&schema, version)?;
        if version == ConcreteFileVersion::V1 {
            self.write_v1_impl(stream, schema, id).await
        } else {
            Box::pin(self.write_current_impl(version, stream, schema, id)).await
        }
    }

    /// Write multi fragment which separated by max_rows_per_file.
    pub async fn write_fragments(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Vec<Fragment>> {
        let (stream, schema) = self.get_stream_and_schema(source).await?;
        self.write_fragments_v2_impl(stream, schema).await
    }

    async fn write_current_impl(
        &self,
        version: ConcreteFileVersion,
        stream: SendableRecordBatchStream,
        schema: Schema,
        id: u64,
    ) -> Result<Fragment> {
        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();
        let progress = params.progress.as_ref();

        Self::validate_schema(&schema, stream.schema().as_ref())?;
        let schema = promote_legacy_blobs(version, schema)?;

        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            params.store_registry(),
            self.dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
        // The dataset's writer, not a raw file writer: it carries the version's
        // write policies, such as preparing Blob v2 columns for the encoders.
        let external_base_resolver = if writes_blob_v2(version) {
            blob_v2_external_base_resolver(None, &params, &schema).await?
        } else {
            None
        };
        let mut writer = open_writer(
            version,
            &object_store,
            &schema,
            &base_path,
            WriterOptions::from_params(&params, external_base_resolver),
        )
        .await?;
        let mut fragment = Fragment::new(id);
        let (filename, _) = writer.data_file_path();
        fragment
            .files
            .push(DataFile::new_unstarted(filename.to_string(), version));

        progress.begin(&fragment).await?;

        let break_limit = (128 * 1024).min(params.max_rows_per_file);

        let mut broken_stream = break_stream(stream, break_limit);
        while let Some(batch) = broken_stream.next().await {
            writer.write(&[batch?]).await?;
        }

        let (num_rows, data_file) = writer.finish().await?;
        fragment.physical_rows = Some(num_rows as usize);

        if matches!(fragment.physical_rows, Some(0)) {
            return Err(Error::invalid_input("Input data was empty."));
        }

        fragment.files[0] = data_file;

        progress.complete(&fragment).await?;

        Ok(fragment)
    }
    async fn write_fragments_v2_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<Vec<Fragment>> {
        let mut params = self.write_params.cloned().unwrap_or_default();

        Self::validate_schema(&schema, stream.schema().as_ref())?;

        let version = params.storage_version_or_default();
        let schema = promote_legacy_blobs(version, schema)?;
        let needs_existing_dataset = params.target_base_names_or_paths.is_some()
            || params.target_bases.is_some()
            || params.target_all_bases.is_some()
            || params.initial_bases.is_some();
        let existing_dataset = if needs_existing_dataset {
            self.existing_dataset(&params).await?
        } else {
            None
        };
        let existing_base_paths = existing_dataset
            .as_ref()
            .map(|dataset| &dataset.manifest.base_paths);
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            params.store_registry(),
            self.dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
        let target_bases_info = if needs_existing_dataset {
            validate_and_resolve_target_bases_with_primary(
                &mut params,
                existing_base_paths,
                &object_store,
                &base_path,
                self.dataset_uri,
            )
            .await?
        } else {
            None
        };
        write_fragments_direct(
            version,
            existing_dataset.as_ref(),
            object_store,
            &base_path,
            &schema,
            stream,
            params,
            target_bases_info,
            SeedCollector::disabled(),
            None,
            None,
            None,
            false,
            None,
        )
        .await
    }

    async fn write_v1_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
        id: u64,
    ) -> Result<Fragment> {
        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();
        let progress = params.progress.as_ref();

        Self::validate_schema(&schema, stream.schema().as_ref())?;

        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            params.store_registry(),
            self.dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
        let filename = format!("{}.lance", generate_random_filename());
        let mut fragment = Fragment::with_file_legacy(id, &filename, &schema, None);
        let full_path = base_path.clone().join(DATA_DIR).join(filename.clone());
        let mut writer = V1FileWriter::<ManifestDescribing>::try_new(
            &object_store,
            &full_path,
            schema,
            &Default::default(),
        )
        .await?;

        progress.begin(&fragment).await?;

        let mut buffered_reader = chunk_stream(stream, params.max_rows_per_group);
        while let Some(batched_chunk) = buffered_reader.next().await {
            let batch = batched_chunk?;
            writer.write(&batch).await?;
        }

        if writer.is_empty() {
            return Err(Error::invalid_input("Input data was empty."));
        }

        fragment.physical_rows = Some(writer.finish().await?.num_rows as usize);

        progress.complete(&fragment).await?;

        Ok(fragment)
    }

    async fn get_stream_and_schema(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<(SendableRecordBatchStream, Schema)> {
        let schema = if let Some(schema) = self.schema {
            Some(schema.clone())
        } else if matches!(self.write_params.map(|p| p.mode), Some(WriteMode::Append)) {
            self.existing_dataset_schema().await?
        } else {
            None
        };
        let (stream, schema) = match schema {
            Some(schema) => (source.into_stream(), schema),
            None => source.into_stream_and_schema().await?,
        };
        Ok((stream, schema))
    }

    async fn existing_dataset_schema(&self) -> Result<Option<Schema>> {
        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();
        match self.load_existing_dataset(&params).await {
            // Use the schema from the dataset, because it has the correct
            // field ids.
            Ok(dataset) => Ok(Some(dataset.schema().clone())),
            // If the dataset does not exist, we can use the schema from
            // the reader.
            Err(Error::DatasetNotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn existing_dataset(&self, params: &WriteParams) -> Result<Option<Table>> {
        match self.load_existing_dataset(params).await {
            Ok(dataset) => Ok(Some(dataset)),
            Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn load_existing_dataset(&self, params: &WriteParams) -> Result<Table> {
        let mut builder = TableBuilder::from_uri(self.dataset_uri);
        if let Some(store_params) = &params.store_params {
            builder = builder.with_store_params(store_params.clone());
        }
        if let Some(commit_handler) = &params.commit_handler {
            builder = builder.with_commit_handler(commit_handler.clone());
        }
        if let Some(session) = &params.session {
            builder = builder.with_session(session.clone());
        }
        if let Some(base_store_params) = &params.base_store_params {
            for (base_path, store_params) in base_store_params {
                builder = builder.with_base_store_params(base_path, store_params.clone());
            }
        }
        builder.load().await
    }

    fn validate_schema(expected: &Schema, actual: &ArrowSchema) -> Result<()> {
        if actual.fields().is_empty() {
            return Err(Error::invalid_input("Cannot write with an empty schema."));
        }
        let actual_lance = Schema::try_from(actual)?;
        actual_lance.check_compatible(expected, &Default::default())?;

        Ok(())
    }
}
