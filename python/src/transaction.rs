// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::DatasetBasePath;
use crate::schema::LanceSchema;
use crate::utils::{PyLance, class_name, export_vec, extract_vec};
use arrow::pyarrow::PyArrowType;
use arrow_schema::Schema as ArrowSchema;
use lance::dataset::transaction::{
    DataOverlayGroup, DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, Transaction,
    UpdateMap, UpdateMapEntry, UpdateMode, UpdatedFragmentOffsets,
};
use lance::datatypes::Schema;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_table::format::{BasePath, DataFile, Fragment, IndexFile, IndexMetadata};
use pyo3::exceptions::PyValueError;
use pyo3::types::PySet;
use pyo3::{Bound, FromPyObject, PyAny, PyResult, Python};
use pyo3::{intern, prelude::*};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

// IndexFile bindings
impl FromPyObject<'_, '_> for PyLance<IndexFile> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let path = ob.getattr("path")?.extract()?;
        let size_bytes = ob.getattr("size_bytes")?.extract()?;
        Ok(Self(IndexFile { path, size_bytes }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&IndexFile> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .expect("Failed to import lance module");

        let cls = namespace
            .getattr("IndexFile")
            .expect("Failed to get IndexFile class");
        cls.call1((self.0.path.clone(), self.0.size_bytes))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<IndexFile> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
    }
}

// IndexMetadata bindings
impl FromPyObject<'_, '_> for PyLance<IndexMetadata> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let uuid = ob.getattr("uuid")?.to_string();
        let name = ob.getattr("name")?.extract()?;
        let fields = ob.getattr("fields")?.extract()?;
        let dataset_version = ob.getattr("dataset_version")?.extract()?;
        let index_version = ob.getattr("index_version")?.extract()?;
        let fragment_ids = ob.getattr("fragment_ids")?;
        let created_at = ob.getattr("created_at")?.extract()?;

        let fragment_ids_ref: &Bound<'_, PySet> = fragment_ids.cast()?;
        let fragment_bitmap = Some(
            fragment_ids_ref
                .into_iter()
                .map(|id| id.extract::<u32>())
                .collect::<PyResult<RoaringBitmap>>()?,
        );
        let base_id: Option<u32> = ob
            .getattr("base_id")?
            .extract::<Option<i64>>()?
            .map(|id| id as u32);

        let files: Option<Vec<IndexFile>> = ob
            .getattr("files")?
            .extract::<Option<Vec<PyLance<IndexFile>>>>()?
            .map(|v| v.into_iter().map(|f| f.0).collect());
        let index_details = match ob.getattr("index_details") {
            Ok(details) => details
                .extract::<Option<(String, Vec<u8>)>>()?
                .map(|(type_url, value)| Arc::new(prost_types::Any { type_url, value })),
            Err(_) => None,
        };
        // Tolerate an object predating this attribute, as with `index_details`
        // above: absent means the index carries no covered columns.
        let covering_fields: Vec<i32> = match ob.getattr("covering_fields") {
            Ok(value) => value.extract()?,
            Err(_) => Vec::new(),
        };

        Ok(Self(IndexMetadata {
            uuid: Uuid::parse_str(&uuid).map_err(|e| PyValueError::new_err(e.to_string()))?,
            name,
            fields,
            covering_fields,
            dataset_version,
            fragment_bitmap,
            index_details,
            index_version,
            created_at,
            base_id,
            files,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&IndexMetadata> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .expect("Failed to import lance module");

        let uuid = self.0.uuid.to_string();
        let name = &self.0.name;
        let fields = &self.0.fields;
        let covering_fields = &self.0.covering_fields;
        let dataset_version = self.0.dataset_version;
        let index_version = self.0.index_version;
        let fragment_ids = self.0.fragment_bitmap.as_ref().map_or_else(
            || PySet::empty(py).unwrap(),
            |bitmap| {
                let set = PySet::empty(py).unwrap();
                for id in bitmap.iter() {
                    set.add(id).unwrap();
                }
                set
            },
        );
        let created_at = self.0.created_at;
        let base_id = self.0.base_id.map(|id| id as i64);
        let files = self
            .0
            .files
            .as_ref()
            .map(|f| export_vec(py, f.as_slice()))
            .transpose()?;
        let index_details = self
            .0
            .index_details
            .as_ref()
            .map(|details| (details.type_url.clone(), details.value.clone()));

        let cls = namespace
            .getattr("Index")
            .expect("Failed to get Index class");
        cls.call1((
            uuid,
            name.clone(),
            fields.clone(),
            dataset_version,
            fragment_ids,
            index_version,
            created_at,
            base_id,
            files,
            index_details,
            covering_fields.clone(),
        ))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<IndexMetadata> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
    }
}

impl FromPyObject<'_, '_> for PyLance<DataReplacementGroup> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let fragment_id = ob.getattr("fragment_id")?.extract::<u64>()?;
        let new_file = &ob.getattr("new_file")?.extract::<PyLance<DataFile>>()?;

        Ok(Self(DataReplacementGroup(fragment_id, new_file.0.clone())))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&DataReplacementGroup> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        let fragment_id = self.0.0;
        let new_file = PyLance(&self.0.1).into_pyobject(py)?;

        let cls = namespace
            .getattr("DataReplacementGroup")
            .expect("Failed to get DataReplacementGroup class");
        cls.call1((fragment_id, new_file))
    }
}

// The Nth offset in an overlay list positionally maps to the Nth value row in
// `data_file`, but `RoaringBitmap` stores offsets in ascending order and drops
// duplicates. A caller-supplied list that isn't strictly ascending would be
// silently reordered, breaking that mapping, so reject it here instead. This can
// go away once we expose RoaringBitmap directly to Python (issue #7695).
fn bitmap_from_sorted_offsets(offsets: Vec<u32>) -> PyResult<RoaringBitmap> {
    if offsets.windows(2).any(|w| w[0] >= w[1]) {
        return Err(PyValueError::new_err(
            "DataOverlayFile.offsets must be strictly ascending with no duplicates; \
             each offset positionally maps to a value row in data_file",
        ));
    }
    Ok(RoaringBitmap::from_sorted_iter(offsets).expect("offsets verified strictly ascending"))
}

impl FromPyObject<'_, '_> for PyLance<DataOverlayFile> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let data_file = ob.getattr("data_file")?.extract::<PyLance<DataFile>>()?.0;
        let offsets = ob.getattr("offsets")?;

        // A flat list of offsets is a dense overlay (one coverage shared by every
        // field); a list of per-field lists is a sparse overlay. Differentiate by
        // shape, trying the dense form first.
        let coverage = if let Ok(shared) = offsets.extract::<Vec<u32>>() {
            OverlayCoverage::dense(bitmap_from_sorted_offsets(shared)?)
        } else if let Ok(per_field) = offsets.extract::<Vec<Vec<u32>>>() {
            OverlayCoverage::sparse(
                per_field
                    .into_iter()
                    .map(bitmap_from_sorted_offsets)
                    .collect::<PyResult<Vec<_>>>()?,
            )
        } else {
            return Err(PyValueError::new_err(
                "DataOverlayFile.offsets must be a list of ints (dense coverage shared by \
                 every field) or a list of per-field int lists (sparse coverage)",
            ));
        };

        // Present (and preserved) when round-tripping an existing fragment's
        // overlays; None/0 when creating an overlay to commit, since the
        // DataOverlay commit stamps the effective version.
        let committed_version = ob
            .getattr("committed_version")?
            .extract::<Option<u64>>()?
            .unwrap_or(0);

        Ok(Self(DataOverlayFile {
            data_file,
            coverage,
            committed_version,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&DataOverlayFile> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        let data_file = PyLance(&self.0.data_file).into_pyobject(py)?;
        let cls = namespace
            .getattr("DataOverlayFile")
            .expect("Failed to get DataOverlayFile class");

        let committed_version = self.0.committed_version;

        // Mirror the read side: a dense overlay becomes a flat list of offsets, a
        // sparse overlay a list of per-field lists.
        match &self.0.coverage {
            OverlayCoverage::Shared(bitmap) => {
                let offsets: Vec<u32> = bitmap.iter().collect();
                cls.call1((data_file, offsets, committed_version))
            }
            OverlayCoverage::PerField(bitmaps) => {
                let offsets: Vec<Vec<u32>> = bitmaps.iter().map(|b| b.iter().collect()).collect();
                cls.call1((data_file, offsets, committed_version))
            }
        }
    }
}

impl FromPyObject<'_, '_> for PyLance<DataOverlayGroup> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let fragment_id = ob.getattr("fragment_id")?.extract::<u64>()?;
        let overlays = extract_vec(&ob.getattr("overlays")?)?;
        Ok(Self(DataOverlayGroup {
            fragment_id,
            overlays,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&DataOverlayGroup> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        let fragment_id = self.0.fragment_id;
        let overlays = export_vec(py, self.0.overlays.as_slice())?;

        let cls = namespace
            .getattr("DataOverlayGroup")
            .expect("Failed to get DataOverlayGroup class");
        cls.call1((fragment_id, overlays))
    }
}

#[derive(Debug, Clone)]
pub struct PyUpdateMode(pub UpdateMode);

impl FromPyObject<'_, '_> for PyUpdateMode {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let mode_str: String = ob.extract()?;
        match mode_str.as_str() {
            "rewrite_rows" => Ok(Self(UpdateMode::RewriteRows)),
            "rewrite_columns" => Ok(Self(UpdateMode::RewriteColumns)),
            _ => Err(PyValueError::new_err(format!(
                "Invalid UpdateMode: {}. Valid options are: rewrite_rows, rewrite_columns",
                mode_str
            ))),
        }
    }
}

impl FromPyObject<'_, '_> for PyLance<Operation> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        match class_name(&ob)?.as_str() {
            "Overwrite" => {
                let schema = extract_schema(&ob.getattr("new_schema")?)?;

                let fragments = extract_vec(&ob.getattr("fragments")?)?;

                let initial_bases = ob
                    .getattr("initial_bases")
                    .ok()
                    .and_then(|attr| attr.extract::<Option<Vec<DatasetBasePath>>>().ok())
                    .flatten()
                    .map(|py_bases| {
                        py_bases
                            .into_iter()
                            .map(BasePath::from)
                            .collect::<Vec<BasePath>>()
                    });

                let op = Operation::Overwrite {
                    schema,
                    fragments,
                    config_upsert_values: None,
                    initial_bases,
                };
                Ok(Self(op))
            }
            "Append" => {
                let fragments = extract_vec(&ob.getattr("fragments")?)?;
                let op = Operation::Append { fragments };
                Ok(Self(op))
            }
            "Delete" => {
                let updated_fragments = extract_vec(&ob.getattr("updated_fragments")?)?;
                let deleted_fragment_ids = ob.getattr("deleted_fragment_ids")?.extract()?;
                let predicate = ob.getattr("predicate")?.extract()?;

                let op = Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids,
                    predicate,
                };
                Ok(Self(op))
            }
            "Update" => {
                let removed_fragment_ids = ob.getattr("removed_fragment_ids")?.extract()?;

                let updated_fragments = extract_vec(&ob.getattr("updated_fragments")?)?;

                let new_fragments = extract_vec(&ob.getattr("new_fragments")?)?;

                let fields_modified = ob.getattr("fields_modified")?.extract()?;

                let fields_for_preserving_frag_bitmap = ob
                    .getattr("fields_for_preserving_frag_bitmap")?
                    .extract()
                    .unwrap_or_default();

                let update_mode = ob
                    .getattr("update_mode")?
                    .extract::<PyUpdateMode>()
                    .ok()
                    .map(|py_mode| py_mode.0);

                // Absent on objects predating the field.
                let updated_fragment_offsets = ob
                    .getattr("updated_fragment_offsets")
                    .ok()
                    .map(|v| v.extract::<Option<HashMap<u64, Vec<u8>>>>())
                    .transpose()?
                    .flatten()
                    .map(|offsets| {
                        offsets
                            .into_iter()
                            .map(|(frag_id, bytes)| {
                                RoaringBitmap::deserialize_from(&bytes[..])
                                    .map(|bitmap| (frag_id, bitmap))
                                    .map_err(|e| {
                                        PyValueError::new_err(format!(
                                            "updated_fragment_offsets[{frag_id}]: invalid \
                                             portable RoaringBitmap bytes: {e}"
                                        ))
                                    })
                            })
                            .collect::<PyResult<HashMap<_, _>>>()
                    })
                    .transpose()?
                    .map(UpdatedFragmentOffsets);

                let op = Operation::Update {
                    removed_fragment_ids,
                    updated_fragments,
                    new_fragments,
                    fields_modified,
                    compacted_sstables: vec![],
                    fields_for_preserving_frag_bitmap,
                    update_mode,
                    inserted_rows_filter: None,
                    updated_fragment_offsets,
                };
                Ok(Self(op))
            }
            "Merge" => {
                let schema = extract_schema(&ob.getattr("schema")?)?;

                let fragments = ob
                    .getattr("fragments")?
                    .extract::<Vec<PyLance<Fragment>>>()?;
                let fragments = fragments.into_iter().map(|f| f.0).collect();

                // Absent on objects predating the field: no assertion, which
                // conservatively conflicts.
                let preserves_nullability = ob
                    .getattr("preserves_nullability")
                    .and_then(|v| v.extract())
                    .unwrap_or(false);

                let op = Operation::Merge {
                    schema,
                    fragments,
                    preserves_nullability,
                };
                Ok(Self(op))
            }
            "Restore" => {
                let version = ob.getattr("version")?.extract()?;
                let op = Operation::Restore { version };
                Ok(Self(op))
            }
            "Rewrite" => {
                let groups = extract_vec(&ob.getattr("groups")?)?;
                let rewritten_indices = extract_vec(&ob.getattr("rewritten_indices")?)?;
                let op = Operation::Rewrite {
                    groups,
                    rewritten_indices,
                    // TODO: pass frag_reuse_index when available
                    frag_reuse_index: None,
                };
                Ok(Self(op))
            }
            "CreateIndex" => {
                let new_indices_py = ob.getattr("new_indices")?;
                let removed_indices_py = ob.getattr("removed_indices")?;

                let new_indices = extract_vec(&new_indices_py)?;
                let removed_indices = extract_vec(&removed_indices_py)?;

                let op = Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                };
                Ok(Self(op))
            }
            "DataReplacement" => {
                let replacements = extract_vec(&ob.getattr("replacements")?)?;

                let op = Operation::DataReplacement { replacements };

                Ok(Self(op))
            }
            "DataOverlay" => {
                let groups = extract_vec(&ob.getattr("groups")?)?;

                let op = Operation::DataOverlay { groups };

                Ok(Self(op))
            }
            "Project" => {
                let schema = extract_schema(&ob.getattr("schema")?)?;
                // Absent on objects predating the field: no assertion, which
                // conservatively conflicts.
                let preserves_nullability = ob
                    .getattr("preserves_nullability")
                    .and_then(|v| v.extract())
                    .unwrap_or(false);

                let op = Operation::Project {
                    schema,
                    preserves_nullability,
                };
                Ok(Self(op))
            }
            "UpdateConfig" => {
                let config_updates = extract_update_map(&ob.getattr("config_updates")?)?;
                let table_metadata_updates =
                    extract_update_map(&ob.getattr("table_metadata_updates")?)?;
                let schema_metadata_updates =
                    extract_update_map(&ob.getattr("schema_metadata_updates")?)?;

                // Extract field_metadata_updates
                let field_metadata_updates_py = ob.getattr("field_metadata_updates")?;
                let mut field_metadata_updates = HashMap::new();

                if !field_metadata_updates_py.is_none()
                    && let Ok(items) = field_metadata_updates_py.call_method0("items")
                {
                    for item in items.try_iter()? {
                        let item = item?;
                        // Extract as a tuple and then get individual elements
                        let tuple = item.cast::<pyo3::types::PyTuple>()?;
                        let field_id = tuple.get_item(0)?.extract::<i32>()?;
                        let update_map = tuple.get_item(1)?;
                        if let Some(map) = extract_update_map(&update_map)? {
                            field_metadata_updates.insert(field_id, map);
                        }
                    }
                }

                let op = Operation::UpdateConfig {
                    config_updates,
                    table_metadata_updates,
                    schema_metadata_updates,
                    field_metadata_updates,
                };
                Ok(Self(op))
            }
            unsupported => Err(PyValueError::new_err(format!(
                "Unsupported operation: {unsupported}",
            ))),
        }
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&Operation> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        match self.0 {
            Operation::Append { fragments } => {
                let fragments = export_vec(py, fragments.as_slice())?;
                let cls = namespace
                    .getattr("Append")
                    .expect("Failed to get Append class");
                cls.call1((fragments,))
            }
            Operation::Overwrite {
                fragments,
                schema,
                initial_bases,
                ..
            } => {
                let fragments_py = export_vec(py, fragments.as_slice())?;

                let schema_py = LanceSchema(schema.clone());

                let cls = namespace
                    .getattr("Overwrite")
                    .expect("Failed to get Overwrite class");

                let initial_bases_py = if let Some(bases) = initial_bases {
                    use crate::dataset::DatasetBasePath;
                    // Convert each Rust BasePath to a Python DatasetBasePath object
                    let bases_py: Vec<DatasetBasePath> = bases
                        .iter()
                        .map(|bp| DatasetBasePath::from(bp.clone()))
                        .collect();
                    pyo3::types::PyList::new(py, bases_py)?.into_any()
                } else {
                    py.None().into_bound(py)
                };

                cls.call1((schema_py, fragments_py, initial_bases_py))
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                fields_for_preserving_frag_bitmap,
                update_mode,
                updated_fragment_offsets,
                ..
            } => {
                let removed_fragment_ids = removed_fragment_ids.into_pyobject(py)?;
                let updated_fragments = export_vec(py, updated_fragments.as_slice())?;
                let new_fragments = export_vec(py, new_fragments.as_slice())?;
                let fields_modified = fields_modified.into_pyobject(py)?;
                let fields_for_preserving_frag_bitmap =
                    fields_for_preserving_frag_bitmap.into_pyobject(py)?;
                let update_mode = match update_mode {
                    Some(mode) => match mode {
                        lance::dataset::transaction::UpdateMode::RewriteRows => "rewrite_rows",
                        lance::dataset::transaction::UpdateMode::RewriteColumns => {
                            "rewrite_columns"
                        }
                    },
                    None => "rewrite_rows",
                };
                let updated_fragment_offsets =
                    updated_fragment_offsets
                        .as_ref()
                        .map(|UpdatedFragmentOffsets(offsets)| {
                            offsets
                                .iter()
                                .map(|(frag_id, bitmap)| {
                                    let mut buf = Vec::with_capacity(bitmap.serialized_size());
                                    bitmap
                                        .serialize_into(&mut buf)
                                        .expect("RoaringBitmap serialization cannot fail");
                                    (*frag_id, buf)
                                })
                                .collect::<HashMap<u64, Vec<u8>>>()
                        });
                let cls = namespace
                    .getattr("Update")
                    .expect("Failed to get Update class");
                cls.call1((
                    removed_fragment_ids,
                    updated_fragments,
                    new_fragments,
                    fields_modified,
                    fields_for_preserving_frag_bitmap,
                    update_mode,
                    updated_fragment_offsets,
                ))
            }
            Operation::DataReplacement { replacements } => {
                let replacements = export_vec(py, replacements.as_slice())?;
                let cls = namespace
                    .getattr("DataReplacement")
                    .expect("Failed to get DataReplacement class");
                cls.call1((replacements,))
            }
            Operation::DataOverlay { groups } => {
                let groups = export_vec(py, groups.as_slice())?;
                let cls = namespace
                    .getattr("DataOverlay")
                    .expect("Failed to get DataOverlay class");
                cls.call1((groups,))
            }
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => {
                let updated_fragments = export_vec(py, updated_fragments.as_slice())?;
                let deleted_fragment_ids = deleted_fragment_ids.into_pyobject(py)?;
                let cls = namespace
                    .getattr("Delete")
                    .expect("Failed to get Delete class");
                cls.call1((updated_fragments, deleted_fragment_ids, predicate))
            }
            Operation::Merge {
                fragments,
                schema,
                preserves_nullability,
            } => {
                let fragments_py = export_vec(py, fragments.as_slice())?;
                let schema_py = LanceSchema(schema.clone());
                let cls = namespace
                    .getattr("Merge")
                    .expect("Failed to get Merge class");
                cls.call1((fragments_py, schema_py, *preserves_nullability))
            }
            Operation::Restore { version } => {
                let cls = namespace
                    .getattr("Restore")
                    .expect("Failed to get Restore class");
                cls.call1((version,))
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
                ..
            } => {
                let groups_py = export_vec(py, groups.as_slice())?;
                let rewritten_indices_py = export_vec(py, rewritten_indices.as_slice())?;
                let cls = namespace
                    .getattr("Rewrite")
                    .expect("Failed to get Rewrite class");
                cls.call1((groups_py, rewritten_indices_py))
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
                ..
            } => {
                let new_indices_py = export_vec(py, new_indices.as_slice())?;
                let removed_indices_py = export_vec(py, removed_indices.as_slice())?;

                let cls = namespace
                    .getattr("CreateIndex")
                    .expect("Failed to get CreateIndex class");
                cls.call1((new_indices_py, removed_indices_py))
            }
            Operation::Project {
                schema,
                preserves_nullability,
            } => {
                let schema_py = LanceSchema(schema.clone());
                let cls = namespace
                    .getattr("Project")
                    .expect("Failed to get Project class");
                cls.call1((schema_py, *preserves_nullability))
            }
            Operation::ReserveFragments { num_fragments } => {
                if let Ok(cls) = namespace.getattr("ReserveFragments") {
                    cls.call1((num_fragments,))
                } else {
                    let base_op = namespace.getattr("BaseOperation")?;
                    base_op.call0()
                }
            }
            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            } => {
                if let Ok(cls) = namespace.getattr("UpdateConfig") {
                    let config = export_update_map(py, config_updates)?;
                    let table_meta = export_update_map(py, table_metadata_updates)?;
                    let schema_meta = export_update_map(py, schema_metadata_updates)?;

                    // Export field_metadata_updates
                    let field_meta = if field_metadata_updates.is_empty() {
                        py.None()
                    } else {
                        let dict = pyo3::types::PyDict::new(py);
                        for (field_id, update_map) in field_metadata_updates {
                            let map_obj = export_update_map(py, &Some(update_map.clone()))?;
                            dict.set_item(field_id, map_obj)?;
                        }
                        dict.into()
                    };

                    cls.call1((config, table_meta, schema_meta, field_meta))
                } else {
                    let base_op = namespace.getattr("BaseOperation")?;
                    base_op.call0()
                }
            }
            Operation::UpdateBases { new_bases } => {
                if let Ok(cls) = namespace.getattr("UpdateBases") {
                    use crate::dataset::DatasetBasePath;
                    let new_bases_py: Vec<DatasetBasePath> = new_bases
                        .iter()
                        .map(|bp| DatasetBasePath::from(bp.clone()))
                        .collect();
                    let new_bases_list = pyo3::types::PyList::new(py, new_bases_py)?;
                    cls.call1((new_bases_list,))
                } else {
                    let base_op = namespace.getattr("BaseOperation")?;
                    base_op.call0()
                }
            }
            _ => todo!(),
        }
    }
}

impl FromPyObject<'_, '_> for PyLance<Transaction> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let read_version = ob.getattr("read_version")?.extract()?;
        let uuid = ob.getattr("uuid")?.extract()?;
        let operation = ob.getattr("operation")?.extract::<PyLance<Operation>>()?.0;
        let transaction_properties = ob
            .getattr("transaction_properties")?
            .extract::<Option<HashMap<String, String>>>()?
            .filter(|map| !map.is_empty())
            .map(Arc::new);
        Ok(Self(Transaction {
            read_version,
            uuid,
            operation,
            tag: None,
            transaction_properties,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&Transaction> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .expect("Failed to import lance module");

        let read_version = self.0.read_version;
        let uuid = &self.0.uuid;
        let operation = PyLance(&self.0.operation).into_pyobject(py)?;

        let cls = namespace
            .getattr("Transaction")
            .expect("Failed to get Transaction class");

        let py_transaction = cls.call1((read_version, operation, uuid))?;

        if let Some(transaction_properties_arc) = &self.0.transaction_properties {
            let py_dict = transaction_properties_arc.as_ref().into_pyobject(py)?;
            py_transaction.setattr("transaction_properties", py_dict)?;
        }
        // Unwrap due to infallible
        Ok(py_transaction.into_pyobject(py).unwrap())
    }
}

impl<'py> IntoPyObject<'py> for PyLance<Transaction> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
    }
}

impl FromPyObject<'_, '_> for PyLance<RewriteGroup> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self(RewriteGroup {
            old_fragments: extract_vec(&ob.getattr("old_fragments")?)?,
            new_fragments: extract_vec(&ob.getattr("new_fragments")?)?,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&RewriteGroup> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .and_then(|cls| cls.getattr(intern!(py, "RewriteGroup")))
            .expect("Failed to get RewriteGroup class");

        let old_fragments = export_vec(py, self.0.old_fragments.as_slice())?;
        let new_fragments = export_vec(py, self.0.new_fragments.as_slice())?;

        cls.call1((old_fragments, new_fragments))
    }
}

impl FromPyObject<'_, '_> for PyLance<RewrittenIndex> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let old_id: String = ob.getattr("old_id")?.extract()?;
        let new_id: String = ob.getattr("new_id")?.extract()?;
        let old_id = Uuid::parse_str(&old_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse UUID: {}", e)))?;
        let new_id = Uuid::parse_str(&new_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse UUID: {}", e)))?;
        let new_details_type_url: String = ob.getattr("new_details_type_url")?.extract()?;
        let new_details_value: Vec<u8> = ob.getattr("new_details_value")?.extract()?;
        let new_index_version: u32 = ob.getattr("new_index_version")?.extract()?;
        Ok(Self(RewrittenIndex {
            old_id,
            new_id,
            new_index_details: prost_types::Any {
                type_url: new_details_type_url,
                value: new_details_value,
            },
            new_index_version,
            new_index_files: None,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&RewrittenIndex> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .and_then(|cls| cls.getattr(intern!(py, "RewrittenIndex")))
            .expect("Failed to get RewrittenIndex class");

        let old_id = self.0.old_id.to_string();
        let new_id = self.0.new_id.to_string();
        cls.call1((old_id, new_id))
    }
}

fn extract_update_map(ob: &Bound<'_, PyAny>) -> PyResult<Option<UpdateMap>> {
    if ob.is_none() {
        return Ok(None);
    }

    let updates_dict = ob.getattr("updates")?;
    let replace = ob.getattr("replace")?.extract::<bool>()?;

    let mut entries = Vec::new();
    if let Ok(items) = updates_dict.call_method0("items") {
        for item in items.try_iter()? {
            let item = item?;
            let tuple = item.extract::<(String, Option<String>)>()?;
            entries.push(UpdateMapEntry {
                key: tuple.0,
                value: tuple.1,
            });
        }
    }

    Ok(Some(UpdateMap {
        update_entries: entries,
        replace,
    }))
}

fn export_update_map(py: Python<'_>, update_map: &Option<UpdateMap>) -> PyResult<Py<PyAny>> {
    match update_map {
        None => Ok(py.None()),
        Some(map) => {
            let lance_module = py.import(intern!(py, "lance"))?;
            let lance_op = lance_module.getattr(intern!(py, "LanceOperation"))?;
            let update_map_cls = lance_op.getattr(intern!(py, "UpdateMap"))?;

            let dict = pyo3::types::PyDict::new(py);
            for entry in &map.update_entries {
                dict.set_item(&entry.key, &entry.value)?;
            }

            update_map_cls
                .call1((dict, map.replace))
                .map(|obj| obj.into())
        }
    }
}

fn extract_schema(schema: &Bound<'_, PyAny>) -> PyResult<Schema> {
    match schema.cast::<LanceSchema>() {
        Ok(schema) => Ok(schema.borrow().0.clone()),
        Err(_) => {
            let arrow_schema = schema.extract::<PyArrowType<ArrowSchema>>()?.0;
            convert_schema(&arrow_schema)
        }
    }
}

fn convert_schema(arrow_schema: &ArrowSchema) -> PyResult<Schema> {
    // Note: the field ids here are wrong.
    Schema::try_from(arrow_schema).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to convert Arrow schema to Lance schema: {}",
            e
        ))
    })
}
