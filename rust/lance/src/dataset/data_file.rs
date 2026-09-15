// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    sync::Arc,
};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use lance_core::{
    Error, Result,
    datatypes::{Field, Schema},
};
use lance_file::{concat::BlobTargetId, format::pb, version::ConcreteFileVersion};
use object_store::path::{Path, PathPart};
use prost::Message;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Dataset, fragment::write::generate_random_filename};
use crate::blob::prepared_to_logical_blob_schema;

/// Serializable identity and logical schema of a final concatenated data file.
///
/// Reuse the same identity for every part write and final concatenation. Callers
/// serialize and deserialize this value with serde. The representation preserves
/// field IDs, metadata, and loaded dictionary values. The caller owns checkpoint
/// state and must keep every use associated with the same dataset and resolved base;
/// Lance does not validate that association across [`Dataset`] instances.
/// Checkpoints must come from trusted application state. Deserialization does not
/// prove artifact ownership or establish whether a target has been committed.
/// Callers must fence stale workers and never resume writes or assembly for a
/// committed target; only staging cleanup via [`Self::finish`] remains valid.
///
/// ```
/// # use lance::dataset::DataFileTarget;
/// # fn checkpoint(target: &DataFileTarget) -> Result<(), serde_json::Error> {
/// let bytes = serde_json::to_vec(target)?;
/// let restored: DataFileTarget = serde_json::from_slice(&bytes)?;
/// # let _ = restored;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DataFileTarget {
    pub(super) file_name: String,
    pub(super) base_id: Option<u32>,
    pub(super) schema: Arc<Schema>,
    pub(super) version: ConcreteFileVersion,
}

impl DataFileTarget {
    /// Create a final data-file target with Lance's ordinary random file naming.
    ///
    /// This only creates a runtime identity; it does not create, reserve, or
    /// register an object. The caller owns the target lifetime, part storage,
    /// cleanup, and commit state. Prepared Blob v2 schemas are normalized to
    /// their caller-visible logical form; the persisted descriptor schema remains
    /// an internal writer detail.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance::dataset::DataFileTarget;
    /// use lance_core::datatypes::Schema;
    /// use lance_file::version::ConcreteFileVersion;
    ///
    /// # fn target(schema: Arc<Schema>) -> lance_core::Result<DataFileTarget> {
    /// DataFileTarget::new(
    ///     None,
    ///     schema,
    ///     ConcreteFileVersion::V2_2,
    /// )
    /// # }
    /// ```
    pub fn new(
        base_id: Option<u32>,
        schema: Arc<Schema>,
        version: ConcreteFileVersion,
    ) -> Result<Self> {
        if version == ConcreteFileVersion::V1 {
            return Err(Error::not_supported(
                "data-file part concatenation does not support Lance v1".to_string(),
            ));
        }
        if base_id == Some(0) {
            return Err(Error::invalid_input(
                "DataFileTarget.base_id must not use reserved ID 0",
            ));
        }
        if schema.fields.is_empty() {
            return Err(Error::invalid_input(
                "DataFileTarget.schema must contain at least one top-level field",
            ));
        }
        let mut field_ids = HashSet::with_capacity(schema.fields.len());
        for field in &schema.fields {
            if !field_ids.insert(field.id) {
                return Err(Error::invalid_input(format!(
                    "DataFileTarget.schema contains duplicate top-level field ID {}",
                    field.id
                )));
            }
        }
        let schema = Arc::new(prepared_to_logical_blob_schema(schema.as_ref())?);
        if schema
            .fields_pre_order()
            .any(|field| field.is_blob() && !field.is_blob_v2())
        {
            return Err(Error::not_supported(
                "DataFileTarget does not support legacy Blob v1 fields",
            ));
        }
        Ok(Self {
            file_name: format!("{}.lance", generate_random_filename()),
            base_id,
            schema,
            version,
        })
    }

    pub(super) fn blob_target_id(&self) -> Option<BlobTargetId> {
        self.schema
            .fields_pre_order()
            .any(|field| field.is_blob_v2())
            .then(|| {
                let base = self
                    .base_id
                    .map(|id| format!("base:{id}"))
                    .unwrap_or_else(|| "primary".to_string());
                BlobTargetId::new(format!("{base}/{}", self.file_name))
            })
    }

    pub(super) fn data_file_key(&self) -> &str {
        self.file_name
            .strip_suffix(".lance")
            .unwrap_or(&self.file_name)
    }

    pub(super) fn object_path(&self, data_dir: &Path) -> Path {
        data_dir.clone().join(self.file_name.as_str())
    }

    pub(super) fn parts_dir(&self, data_dir: &Path) -> Path {
        data_dir
            .clone()
            .join("_parts")
            .join(self.file_name.as_str())
    }

    /// Release staging parts after a successful commit, keeping the final file
    /// and its managed Blob payloads. Stop all writers and assemblers first.
    ///
    /// This also removes incomplete parts. Missing files are OK; storage failures
    /// may leave partial cleanup and can be retried. Use the same dataset and
    /// resolved base as the original write. This does not commit the target.
    /// All Blob payloads in the target's namespace are retained, including those
    /// from failed or unused retry leases. Ordinary dataset GC also retains these
    /// payloads while the parent data file is referenced; this operation does not
    /// perform per-Blob reachability collection.
    ///
    /// ```
    /// # use lance::{Dataset, dataset::DataFileTarget};
    /// # async fn release(dataset: &Dataset, target: &DataFileTarget) -> lance_core::Result<()> {
    /// // After committing the assembled file and releasing all part users:
    /// target.finish(dataset).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn finish(&self, dataset: &Dataset) -> Result<()> {
        let data_dir = dataset.data_file_dir_for_base(self.base_id)?;
        let store = dataset.object_store(self.base_id).await?;
        if let Err(error) = store.remove_dir_all(self.parts_dir(&data_dir)).await
            && !error.is_not_found()
        {
            return Err(error);
        }
        Ok(())
    }

    /// Delete an abandoned target's staging parts, final file, and managed Blob
    /// payloads, including objects left by failed writes or assembly.
    ///
    /// The caller must stop all users and ensure no current or retained dataset
    /// version, checkpoint, or future commit needs this target. Lance does not
    /// track task liveness or check historical references. Use the original
    /// dataset/base mapping. Missing objects are OK; retry on storage failures.
    ///
    /// ```
    /// # use lance::{Dataset, dataset::DataFileTarget};
    /// # async fn abandon(dataset: &Dataset, target: &DataFileTarget) -> lance_core::Result<()> {
    /// // After abandoning the target and stopping all its users:
    /// target.cleanup(dataset).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn cleanup(&self, dataset: &Dataset) -> Result<()> {
        self.finish(dataset).await?;
        let data_dir = dataset.data_file_dir_for_base(self.base_id)?;
        let store = dataset.object_store(self.base_id).await?;
        if let Err(error) = store.delete(&self.object_path(&data_dir)).await
            && !error.is_not_found()
        {
            return Err(error);
        }
        if let Err(error) = store
            .remove_dir_all(data_dir.join(self.data_file_key()))
            .await
            && !error.is_not_found()
        {
            return Err(error);
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct TargetData {
    format_version: u32,
    file_name: String,
    base_id: Option<u32>,
    file_version: String,
    fields: Vec<FieldData>,
    metadata: HashMap<String, String>,
}

// Keep the tree explicit: logical Blob children can carry unassigned field IDs,
// so reconstructing parent/child relationships from IDs alone is ambiguous.
#[derive(Serialize, Deserialize)]
struct FieldData {
    field: Vec<u8>,
    children: Vec<Self>,
    dictionary_values: Option<Vec<u8>>,
}

impl TryFrom<&Field> for FieldData {
    type Error = Error;

    fn try_from(field: &Field) -> Result<Self> {
        let dictionary_values = field
            .dictionary
            .as_ref()
            .and_then(|dictionary| dictionary.values.as_ref())
            .map(|values| -> Result<Vec<u8>> {
                let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "values",
                    values.data_type().clone(),
                    true,
                )]));
                let batch = RecordBatch::try_new(schema.clone(), vec![values.clone()])?;
                let mut bytes = Vec::new();
                let mut writer = StreamWriter::try_new(&mut bytes, schema.as_ref())?;
                writer.write(&batch)?;
                writer.finish()?;
                drop(writer);
                Ok(bytes)
            })
            .transpose()?;
        Ok(Self {
            field: pb::Field::from(field).encode_to_vec(),
            children: field
                .children
                .iter()
                .map(Self::try_from)
                .collect::<Result<_>>()?,
            dictionary_values,
        })
    }
}

impl TryFrom<FieldData> for Field {
    type Error = Error;

    fn try_from(data: FieldData) -> Result<Self> {
        let mut field = Self::from(&pb::Field::decode(data.field.as_slice())?);
        field.children = data
            .children
            .into_iter()
            .map(Self::try_from)
            .collect::<Result<_>>()?;
        if let Some(bytes) = data.dictionary_values {
            let dictionary = field.dictionary.as_mut().ok_or_else(|| {
                Error::invalid_input(format!(
                    "field '{}' has values without dictionary metadata",
                    field.name
                ))
            })?;
            let mut reader = StreamReader::try_new(Cursor::new(bytes), None)?;
            let batch = reader.next().transpose()?.ok_or_else(|| {
                Error::invalid_input(format!(
                    "field '{}' has an empty dictionary stream",
                    field.name
                ))
            })?;
            if batch.num_columns() != 1 || reader.next().transpose()?.is_some() {
                return Err(Error::invalid_input(format!(
                    "field '{}' dictionary stream must contain one column and one batch",
                    field.name
                )));
            }
            dictionary.values = Some(batch.column(0).clone());
        }
        Ok(field)
    }
}

impl Serialize for DataFileTarget {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let fields = self
            .schema
            .fields
            .iter()
            .map(FieldData::try_from)
            .collect::<Result<Vec<_>>>()
            .map_err(serde::ser::Error::custom)?;
        TargetData {
            format_version: 1,
            file_name: self.file_name.clone(),
            base_id: self.base_id,
            file_version: self.version.to_string(),
            fields,
            metadata: self.schema.metadata.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DataFileTarget {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let data = TargetData::deserialize(deserializer)?;
        if data.format_version != 1 {
            return Err(serde::de::Error::custom(format!(
                "unsupported data-file target checkpoint version {}",
                data.format_version
            )));
        }
        let fields = data
            .fields
            .into_iter()
            .map(Field::try_from)
            .collect::<Result<Vec<_>>>()
            .map_err(serde::de::Error::custom)?;
        let version = ConcreteFileVersion::from_manifest_string(&data.file_version)
            .map_err(serde::de::Error::custom)?;
        // The identity must select a child object, never a parent directory.
        PathPart::parse(&data.file_name).map_err(|error| {
            serde::de::Error::custom(format!("invalid target identity: {error}"))
        })?;
        if data
            .file_name
            .strip_suffix(".lance")
            .unwrap_or(&data.file_name)
            .is_empty()
        {
            return Err(serde::de::Error::custom(
                "target identity must not be empty",
            ));
        }
        let mut target = Self::new(
            data.base_id,
            Arc::new(Schema {
                fields,
                metadata: data.metadata,
            }),
            version,
        )
        .map_err(serde::de::Error::custom)?;
        target.file_name = data.file_name;
        Ok(target)
    }
}
