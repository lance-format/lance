// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::fmt::Write;
use std::sync::Arc;

use arrow::pyarrow::{PyArrowType, ToPyArrow};
use arrow_array::{Array, FixedSizeListArray};
use arrow_data::ArrayData;
use chrono::{DateTime, Utc};
use lance::dataset::Dataset as LanceDataset;
use lance::index::DatasetIndexExt;
use lance::index::IndexSegment;
use lance::index::vector::ivf::builder::write_vector_storage;
use lance::index::vector::pq::build_pq_model_in_fragments;
use lance::io::ObjectStore;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::vector::ivf::shuffler::{IvfShuffler, shuffle_vectors};
use lance_index::vector::{
    ivf::{IvfBuildParams, storage::IvfModel},
    pq::{PQBuildParams, ProductQuantizer},
};
use lance_linalg::distance::DistanceType;
use lance_table::format::{IndexMetadata, list_index_files_with_sizes};
use pyo3::Bound;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;
use pyo3::{
    PyResult, Python, pyfunction,
    types::{PyList, PyModule},
    wrap_pyfunction,
};

use lance::index::DatasetIndexInternalExt;

use crate::fragment::FileFragment;
use crate::utils::PyJson;
use crate::{
    dataset::Dataset, error::PythonErrorExt, file::object_store_from_uri_or_path_no_options, rt,
};
use lance::index::vector::ivf::write_ivf_pq_file_from_existing_index;
use lance_index::{IndexDescription, IndexType};
use uuid::Uuid;

#[pyclass(
    name = "IndexConfig",
    module = "lance.indices",
    get_all,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyIndexConfig {
    pub index_type: String,
    pub config: String,
}

#[pymethods]
impl PyIndexConfig {
    #[new]
    fn new(index_type: &str, config: &str) -> PyResult<Self> {
        Ok(Self {
            index_type: index_type.to_string(),
            config: config.to_string(),
        })
    }
}

#[pyclass(name = "IndexSegment", module = "lance.indices", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyIndexSegment {
    pub(crate) inner: IndexSegment,
}

#[pymethods]
impl PyIndexSegment {
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid().to_string()
    }

    #[getter]
    fn fragment_ids(&self) -> HashSet<u32> {
        self.inner.fragment_bitmap().iter().collect()
    }

    #[getter]
    fn index_version(&self) -> i32 {
        self.inner.index_version()
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexSegment(uuid={}, fragment_ids={:?}, index_version={})",
            self.uuid(),
            self.fragment_ids(),
            self.index_version()
        )
    }
}

#[pyclass(name = "IvfModel", module = "lance.indices", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyIvfModel {
    pub(crate) inner: IvfModel,
}

#[pymethods]
impl PyIvfModel {
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(centroids) = &self.inner.centroids {
            let data = centroids.clone().into_data();
            Ok(Some(data.to_pyarrow(py)?))
        } else {
            Ok(None)
        }
    }
}

/// Internal helper to fetch an IVF model for the given index name.
async fn do_get_ivf_model(dataset: &Dataset, index_name: &str) -> PyResult<IvfModel> {
    use lance_index::metrics::NoOpMetricsCollector;

    // Load index metadata list
    let idx_metas = dataset.ds.load_indices().await.infer_error()?; // Convert Lance error to PyErr

    // Find the index by name
    let idx_meta = idx_metas
        .iter()
        .find(|idx| idx.name == index_name)
        .ok_or_else(|| PyValueError::new_err(format!("Index \"{}\" not found", index_name)))?;

    if idx_meta.fields.is_empty() {
        return Err(PyValueError::new_err("Index has no fields"));
    }

    let schema = dataset.ds.schema();
    let field = schema
        .field_by_id(idx_meta.fields[0])
        .ok_or_else(|| PyValueError::new_err("Failed to resolve index field"))?;
    let column_name = &field.name;

    // Open the vector index
    let vindex = dataset
        .ds
        .open_vector_index(column_name, &idx_meta.uuid, &NoOpMetricsCollector)
        .await
        .infer_error()?;

    // Clone the IVF model
    Ok(vindex.ivf_model().clone())
}

#[pyfunction]
fn get_ivf_model(py: Python<'_>, dataset: &Dataset, index_name: &str) -> PyResult<Py<PyIvfModel>> {
    let ivf_model = rt().block_on(Some(py), do_get_ivf_model(dataset, index_name))??;
    Py::new(py, PyIvfModel { inner: ivf_model })
}

#[allow(clippy::too_many_arguments)]
async fn do_train_ivf_model(
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_partitions: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
    fragment_ids: Option<Vec<u32>>,
) -> PyResult<ArrayData> {
    // We verify distance_type earlier so can unwrap here
    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let params = IvfBuildParams {
        max_iters: max_iters as usize,
        sample_rate: sample_rate as usize,
        num_partitions: Some(num_partitions as usize),
        ..Default::default()
    };
    let ivf_model = lance::index::vector::ivf::build_ivf_model(
        dataset.ds.as_ref(),
        column,
        dimension,
        distance_type,
        &params,
        fragment_ids.as_deref(),
        Arc::new(NoopIndexBuildProgress),
    )
    .await
    .infer_error()?;
    let centroids = ivf_model.centroids.unwrap();
    Ok(centroids.into_data())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(dataset, column, dimension, num_partitions, distance_type, sample_rate, max_iters, fragment_ids=None))]
fn train_ivf_model<'py>(
    py: Python<'py>,
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_partitions: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
    fragment_ids: Option<Vec<u32>>,
) -> PyResult<Bound<'py, PyAny>> {
    let centroids = rt().block_on(
        Some(py),
        do_train_ivf_model(
            dataset,
            column,
            dimension,
            num_partitions,
            distance_type,
            sample_rate,
            max_iters,
            fragment_ids,
        ),
    )??;
    centroids.to_pyarrow(py)
}

#[allow(clippy::too_many_arguments)]
async fn do_train_pq_model(
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
    num_bits: u32,
    ivf_model: IvfModel,
    fragment_ids: Option<Vec<u32>>,
) -> PyResult<ArrayData> {
    // We verify distance_type earlier so can unwrap here
    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let params = PQBuildParams {
        num_sub_vectors: num_subvectors as usize,
        num_bits: num_bits as usize,
        max_iters: max_iters as usize,
        sample_rate: sample_rate as usize,
        ..Default::default()
    };
    let pq_model = build_pq_model_in_fragments(
        dataset.ds.as_ref(),
        column,
        dimension,
        distance_type,
        &params,
        Some(&ivf_model),
        fragment_ids.as_deref(),
    )
    .await
    .infer_error()?;
    Ok(pq_model.codebook.into_data())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(dataset, column, dimension, num_subvectors, distance_type, sample_rate, max_iters, ivf_centroids, fragment_ids=None, num_bits=8))]
fn train_pq_model<'py>(
    py: Python<'py>,
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
    ivf_centroids: PyArrowType<ArrayData>,
    fragment_ids: Option<Vec<u32>>,
    num_bits: u32,
) -> PyResult<Bound<'py, PyAny>> {
    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);
    let ivf_model = IvfModel {
        centroids: Some(ivf_centroids),
        offsets: vec![],
        lengths: vec![],
        loss: None,
    };
    let codebook = rt().block_on(
        Some(py),
        do_train_pq_model(
            dataset,
            column,
            dimension,
            num_subvectors,
            distance_type,
            sample_rate,
            max_iters,
            num_bits,
            ivf_model,
            fragment_ids,
        ),
    )??;
    codebook.to_pyarrow(py)
}

/// Mint one RaBitQ rotation and return it as a JSON string.
///
/// Distributed IVF_RQ builds must pin a single rotation across all workers so that
/// independently built per-fragment segments rotate vectors identically and their
/// binary codes remain comparable when merged. A driver calls this once and broadcasts
/// the resulting string to every `create_index_uncommitted(..., rabitq_model=...)` call.
///
/// The rotation is always the "fast" rotation since its sign vector is JSON-serializable,
/// whereas the "matrix" rotation stores a dense matrix in a binary buffer that is dropped by
/// the JSON wire format. `dtype` is accepted for API symmetry but does not affect the fast
/// rotation.
///
/// # Example (Python)
///
/// ```python
/// from lance.lance import indices
///
/// # Mint one model and broadcast `model` to every worker.
/// model = indices.build_rq_model(dimension=128, num_bits=5)
/// seg = ds.create_index_uncommitted(
///     column="vector",
///     index_type="IVF_RQ",
///     num_partitions=256,
///     ivf_centroids=centroids,
///     rabitq_model=model,
///     fragment_ids=my_fragments,
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (dimension, num_bits=5, dtype="float32"))]
pub fn build_rq_model(dimension: usize, num_bits: u8, dtype: &str) -> PyResult<String> {
    use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
    use lance_index::vector::bq::RQRotationType;
    use lance_index::vector::bq::builder::RabitQuantizer;
    use lance_index::vector::quantizer::Quantization;

    if !dimension.is_multiple_of(u8::BITS as usize) {
        return Err(PyValueError::new_err(
            "dimension must be divisible by 8 for IVF_RQ",
        ));
    }
    let dim = dimension as i32;
    let rotation = RQRotationType::Fast;
    let quantizer = match dtype.to_lowercase().as_str() {
        "float16" => RabitQuantizer::new_with_rotation::<Float16Type>(num_bits, dim, rotation),
        "float32" => RabitQuantizer::new_with_rotation::<Float32Type>(num_bits, dim, rotation),
        "float64" => RabitQuantizer::new_with_rotation::<Float64Type>(num_bits, dim, rotation),
        other => {
            return Err(PyValueError::new_err(format!("unsupported dtype: {other}")));
        }
    };
    serde_json::to_string(&quantizer.metadata(None))
        .map_err(|e| PyValueError::new_err(format!("failed to serialize RQ model: {e}")))
}

#[allow(clippy::too_many_arguments)]
async fn do_transform_vectors(
    dataset: &Dataset,
    column: &str,
    distance_type: DistanceType,
    ivf_centroids: FixedSizeListArray,
    pq_model: ProductQuantizer,
    dst_uri: &str,
    fragments: Vec<FileFragment>,
    partitions_ds_uri: Option<&str>,
) -> PyResult<()> {
    let num_rows = dataset.ds.count_rows(None).await.infer_error()?;
    let fragments = fragments.iter().map(|item| item.metadata().0).collect();
    let transform_input = dataset
        .ds
        .scan()
        .with_fragments(fragments)
        .project(&[column])
        .infer_error()?
        .with_row_id()
        .batch_size(8192)
        .try_into_stream()
        .await
        .infer_error()?;

    let (obj_store, path) = object_store_from_uri_or_path_no_options(dst_uri).await?;
    let writer = obj_store.create(&path).await.infer_error()?;
    write_vector_storage(
        &dataset.ds,
        transform_input,
        num_rows as u64,
        ivf_centroids,
        pq_model,
        distance_type,
        column,
        writer,
        partitions_ds_uri,
    )
    .await
    .infer_error()?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(dataset, column, dimension, num_subvectors, distance_type, ivf_centroids, pq_codebook, dst_uri, fragments, partitions_ds_uri=None, num_bits=8))]
pub fn transform_vectors(
    py: Python<'_>,
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    ivf_centroids: PyArrowType<ArrayData>,
    pq_codebook: PyArrowType<ArrayData>,
    dst_uri: &str,
    fragments: Vec<FileFragment>,
    partitions_ds_uri: Option<&str>,
    num_bits: u32,
) -> PyResult<()> {
    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);
    let codebook = pq_codebook.0;
    let codebook = FixedSizeListArray::from(codebook);
    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let pq = ProductQuantizer::new(
        num_subvectors as usize,
        num_bits,
        dimension,
        codebook,
        distance_type,
    );
    rt().block_on(
        Some(py),
        do_transform_vectors(
            dataset,
            column,
            distance_type,
            ivf_centroids,
            pq,
            dst_uri,
            fragments,
            partitions_ds_uri,
        ),
    )?
}

#[allow(deprecated)]
async fn do_shuffle_transformed_vectors(
    unsorted_filenames: Vec<String>,
    dir_path: &str,
    ivf_centroids: FixedSizeListArray,
    shuffle_output_root_filename: &str,
) -> PyResult<Vec<String>> {
    let (obj_store, path) = ObjectStore::from_path(dir_path).infer_error()?;
    if !obj_store.is_local() {
        return Err(PyValueError::new_err(
            "shuffle_vectors input and output path is currently required to be local",
        ));
    }
    let partition_files = shuffle_vectors(
        unsorted_filenames,
        path,
        ivf_centroids,
        shuffle_output_root_filename,
    )
    .await
    .infer_error()?;
    Ok(partition_files)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn shuffle_transformed_vectors(
    py: Python<'_>,
    unsorted_filenames: Vec<String>,
    dir_path: &str,
    ivf_centroids: PyArrowType<ArrayData>,
    shuffle_output_root_filename: &str,
) -> PyResult<Py<PyAny>> {
    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);

    let result = rt().block_on(
        None,
        do_shuffle_transformed_vectors(
            unsorted_filenames,
            dir_path,
            ivf_centroids,
            shuffle_output_root_filename,
        ),
    )?;

    match result {
        Ok(partition_files) => PyList::new(py, partition_files).map(|py_list| py_list.into()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

async fn do_load_shuffled_vectors(
    filenames: Vec<String>,
    dir_path: &str,
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    ivf_model: IvfModel,
    pq_model: ProductQuantizer,
) -> PyResult<()> {
    let (_, path) = object_store_from_uri_or_path_no_options(dir_path).await?;
    let streams = IvfShuffler::load_partitioned_shuffles(&path, filenames)
        .await
        .infer_error()?;

    let index_id = Uuid::new_v4();

    write_ivf_pq_file_from_existing_index(
        &dataset.ds,
        column,
        index_name,
        index_id,
        ivf_model,
        pq_model,
        streams,
    )
    .await
    .infer_error()?;

    let mut ds = dataset.ds.as_ref().clone();
    let index_dir = ds.indices_dir().clone().join(index_id.to_string());
    let object_store = ds.object_store(None).await.infer_error()?;
    let files = list_index_files_with_sizes(object_store.as_ref(), &index_dir)
        .await
        .infer_error()?;
    let metadata = IndexMetadata {
        uuid: index_id,
        name: index_name.to_string(),
        fields: vec![ds.schema().field(column).unwrap().id],
        covering_fields: vec![],
        dataset_version: ds.manifest.version,
        fragment_bitmap: Some(ds.fragments().iter().map(|f| f.id as u32).collect()),
        index_details: Some(Arc::new(
            prost_types::Any::from_msg(&lance_index::pb::VectorIndexDetails::default()).unwrap(),
        )),
        index_version: IndexType::IvfPq.version(),
        created_at: Some(Utc::now()),
        base_id: None,
        files: Some(files),
    };
    ds.commit_existing_index_segments(index_name, column, vec![metadata])
        .await
        .infer_error()?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature=(filenames, dir_path, dataset, column, ivf_centroids, pq_codebook, pq_dimension, num_subvectors, distance_type, index_name=None, num_bits=8))]
#[allow(clippy::too_many_arguments)]
pub fn load_shuffled_vectors(
    filenames: Vec<String>,
    dir_path: &str,
    dataset: &Dataset,
    column: &str,
    ivf_centroids: PyArrowType<ArrayData>,
    pq_codebook: PyArrowType<ArrayData>,
    pq_dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    index_name: Option<&str>,
    num_bits: u32,
) -> PyResult<()> {
    let mut default_idx_name = column.to_string();
    default_idx_name.push_str("_idx");
    let idx_name = index_name.unwrap_or(default_idx_name.as_str());

    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);

    let ivf_model = IvfModel {
        centroids: Some(ivf_centroids),
        offsets: vec![],
        lengths: vec![],
        loss: None,
    };

    let codebook = pq_codebook.0;
    let codebook = FixedSizeListArray::from(codebook);

    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let pq_model = ProductQuantizer::new(
        num_subvectors as usize,
        num_bits,
        pq_dimension,
        codebook,
        distance_type,
    );

    rt().block_on(
        None,
        do_load_shuffled_vectors(
            filenames, dir_path, dataset, column, idx_name, ivf_model, pq_model,
        ),
    )?
}

#[pyclass(
    name = "IndexSegmentDescription",
    module = "lance.indices",
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyIndexSegmentDescription {
    /// The UUID of the index segment
    pub uuid: String,
    /// The dataset version at which the index segment was last updated
    pub dataset_version_at_last_update: u64,
    /// The fragment ids that are covered by the index segment
    pub fragment_ids: HashSet<u32>,
    /// The version of the index
    pub index_version: i32,
    /// The timestamp when the index segment was created
    pub created_at: Option<DateTime<Utc>>,
    /// The total size in bytes of all files in this segment
    /// (None for backward compatibility with indices created before file tracking)
    pub size_bytes: Option<u64>,
    /// The id of the dataset base path that stores this segment
    /// (None when the segment is stored in the dataset's default base path)
    pub base_id: Option<i64>,
    /// The ids of the fields whose values this segment carries but is not keyed on.
    /// Always the trailing entries of the segment's fields.
    pub covering_fields: Vec<i32>,
}

impl PyIndexSegmentDescription {
    pub fn from_metadata(segment: &lance_table::format::IndexMetadata) -> Self {
        let fragment_ids = segment
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| bitmap.iter().collect::<HashSet<_>>())
            .unwrap_or_default();
        let size_bytes = segment.total_size_bytes();

        Self {
            uuid: segment.uuid.to_string(),
            dataset_version_at_last_update: segment.dataset_version,
            fragment_ids,
            index_version: segment.index_version,
            created_at: segment.created_at,
            size_bytes,
            base_id: segment.base_id.map(|id| id as i64),
            covering_fields: segment.covering_fields.clone(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "IndexSegmentDescription(uuid={}, dataset_version_at_last_update={}, fragment_ids={:?}, index_version={}, created_at={:?}, size_bytes={:?}, base_id={:?}, covering_fields={:?})",
            self.uuid,
            self.dataset_version_at_last_update,
            self.fragment_ids,
            self.index_version,
            self.created_at,
            self.size_bytes,
            self.base_id,
            self.covering_fields
        )
    }
}

#[pyclass(name = "IndexDescription", module = "lance.indices", get_all)]
pub struct PyIndexDescription {
    /// The name of the index
    pub name: String,
    /// The full type URL of the index
    pub type_url: String,
    /// The short type of the index (may not be unique)
    pub index_type: String,
    /// The ids of the fields the index is keyed on -- the columns it can answer queries about.
    /// Covering ("included") columns are excluded; those are reported per-segment as
    /// `covering_fields`.
    pub fields: Vec<u32>,
    /// The full paths of the fields the index is keyed on, matching `fields`
    /// (dotted, with backtick-quoted segments for non-identifier names)
    pub field_names: Vec<String>,
    /// The number of rows indexed by the index
    pub num_rows_indexed: u64,
    /// The details of the index
    pub details: PyJson,
    /// The segments of the index
    pub segments: Vec<PyIndexSegmentDescription>,
    /// The total size in bytes of all files across all segments
    /// (None for backward compatibility with indices created before file tracking)
    pub total_size_bytes: Option<u64>,
}

impl PyIndexDescription {
    pub fn new(index: &dyn IndexDescription, dataset: &LanceDataset) -> Self {
        let field_names = index
            .field_ids()
            .iter()
            .map(|field| {
                dataset
                    .schema()
                    .field_path_minimal(*field as i32)
                    .unwrap_or_else(|_| "<unknown>".to_string())
            })
            .collect();

        let segments = index
            .metadata()
            .iter()
            .map(PyIndexSegmentDescription::from_metadata)
            .collect();

        let details = index.details().unwrap_or_else(|_| "{}".to_string());

        Self {
            name: index.name().to_string(),
            fields: index.field_ids().to_vec(),
            field_names,
            index_type: index.index_type().to_string(),
            segments,
            type_url: index.type_url().to_string(),
            num_rows_indexed: index.rows_indexed(),
            details: PyJson(details),
            total_size_bytes: index.total_size_bytes(),
        }
    }
}

#[pymethods]
impl PyIndexDescription {
    pub fn __repr__(&self) -> String {
        let mut repr = format!(
            "IndexDescription(name='{}', type_url='{}', num_rows_indexed={}, fields={:?}, field_names={:?}, num_segments={}",
            self.name,
            self.type_url,
            self.num_rows_indexed,
            self.fields,
            self.field_names,
            self.segments.len()
        );
        if let Some(byte_size) = self.total_size_bytes {
            write!(repr, ", total_size_bytes={}", byte_size).unwrap();
        }
        repr.push(')');
        repr
    }
}

pub fn register_indices(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let indices = PyModule::new(py, "indices")?;
    indices.add_wrapped(wrap_pyfunction!(train_ivf_model))?;
    indices.add_wrapped(wrap_pyfunction!(train_pq_model))?;
    indices.add_wrapped(wrap_pyfunction!(build_rq_model))?;
    indices.add_wrapped(wrap_pyfunction!(transform_vectors))?;
    indices.add_wrapped(wrap_pyfunction!(shuffle_transformed_vectors))?;
    indices.add_wrapped(wrap_pyfunction!(load_shuffled_vectors))?;
    indices.add_class::<PyIvfModel>()?;
    indices.add_class::<PyIndexConfig>()?;
    indices.add_class::<PyIndexSegment>()?;
    indices.add_class::<PyIndexDescription>()?;
    indices.add_class::<PyIndexSegmentDescription>()?;
    indices.add_wrapped(wrap_pyfunction!(get_ivf_model))?;
    m.add_submodule(&indices)?;
    Ok(())
}
