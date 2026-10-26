// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyString};
use pyo3::{Bound, PyAny, PyResult, Python, pyclass, pymethods};

use lance::session::{CacheSpec, Session as LanceSession};
use lance_core::cache::{
    BackendConfig, CacheActivity, CacheDiagnostics, CacheSnapshotMode, build_from_config,
    build_from_uri,
};

use crate::object_store::PyObjectStoreRegistry;
use crate::rt;

/// The Session holds stateful information for a dataset.
///
/// The session contains caches for opened indices and file metadata.
///
/// Parameters
/// ----------
/// index_cache_size_bytes : int, optional
///     Capacity of the default index cache in bytes.
/// metadata_cache_size_bytes : int, optional
///     Capacity of the default metadata cache in bytes.
/// index_cache_backend : str or dict, optional
///     Custom index cache backend. Strings are backend URIs such as
///     ``"moka://?capacity=1048576"``. Dicts must contain ``"kind"`` and may
///     contain ``"options"``, for example
///     ``{"kind": "moka", "options": {"capacity": "1048576"}}``.
/// metadata_cache_backend : str or dict, optional
///     Custom metadata cache backend with the same format as
///     ``index_cache_backend``.
///
/// ``index_cache_backend`` is mutually exclusive with
/// ``index_cache_size_bytes``. ``metadata_cache_backend`` is mutually
/// exclusive with ``metadata_cache_size_bytes``.
#[pyclass(name = "_Session", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct Session {
    pub inner: Arc<LanceSession>,
}

impl Session {
    pub fn new(inner: Arc<LanceSession>) -> Self {
        Self { inner }
    }
}

/// Turn a Python-supplied backend descriptor into an `Arc<dyn CacheBackend>`,
/// or return `Ok(None)` when the caller did not pass one.
///
/// Accepts:
///   * `str` — treated as a URI (`moka://?capacity=...`) and passed to
///     [`build_from_uri`].
///   * `dict` — must have string keys `kind` (required) and `options`
///     (optional `dict[str, str]`) matching [`BackendConfig`]; passed to
///     [`build_from_config`].
///
/// Any other Python type is rejected with a clear `TypeError`-style
/// `PyValueError`.
///
/// If `size_field_set` is `true` and `backend` is `Some`, both a size and a
/// backend were provided for the same cache. Rather than silently letting
/// one override the other (Proposal §7), this is rejected up-front so the
/// operator gets an actionable error.
fn resolve_cache_spec(
    backend_field: &str,
    backend: Option<&Bound<'_, PyAny>>,
    size_field: &str,
    size: Option<usize>,
) -> PyResult<CacheSpec> {
    if backend.is_some() && size.is_some() {
        return Err(PyValueError::new_err(format!(
            "{} and {} are mutually exclusive; set one or the other",
            size_field, backend_field,
        )));
    }

    let Some(value) = backend else {
        return Ok(size.map(CacheSpec::Size).unwrap_or(CacheSpec::Default));
    };

    if value.cast::<PyString>().is_ok() {
        let uri: String = value.extract()?;
        return build_from_uri(&uri)
            .map(CacheSpec::Backend)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", backend_field, e)));
    }

    if let Ok(dict) = value.cast::<PyDict>() {
        let cfg = backend_config_from_dict(backend_field, dict)?;
        return build_from_config(&cfg)
            .map(CacheSpec::Backend)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", backend_field, e)));
    }

    let type_name: String = value.get_type().getattr("__name__")?.extract()?;
    Err(PyValueError::new_err(format!(
        "{}: expected str (URI) or dict with 'kind'/'options' keys, got {}",
        backend_field, type_name,
    )))
}

fn backend_config_from_dict(field: &str, dict: &Bound<'_, PyDict>) -> PyResult<BackendConfig> {
    for (key, _) in dict.iter() {
        if key.cast::<PyString>().is_err() {
            return Err(PyValueError::new_err(format!(
                "{}: dict keys must be strings",
                field
            )));
        }
        let key: String = key.extract()?;
        if key != "kind" && key != "options" {
            return Err(PyValueError::new_err(format!(
                "{}: unknown dict key {:?}; expected 'kind' or 'options'",
                field, key
            )));
        }
    }

    let kind_obj = dict.get_item("kind")?.ok_or_else(|| {
        PyValueError::new_err(format!("{}: dict must contain a 'kind' key", field))
    })?;
    if kind_obj.cast::<PyString>().is_err() {
        return Err(PyValueError::new_err(format!(
            "{}: 'kind' must be a string",
            field
        )));
    }
    let kind: String = kind_obj.extract()?;

    let mut options: HashMap<String, String> = HashMap::new();
    if let Some(options_obj) = dict.get_item("options")? {
        let options_dict = options_obj.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("{}: 'options' must be a dict[str, str]", field))
        })?;
        for (k, v) in options_dict.iter() {
            if k.cast::<PyString>().is_err() {
                return Err(PyValueError::new_err(format!(
                    "{}: 'options' keys must be strings",
                    field
                )));
            }
            if v.cast::<PyString>().is_err() {
                return Err(PyValueError::new_err(format!(
                    "{}: 'options' values must be strings",
                    field
                )));
            }
            let key: String = k.extract()?;
            let value: String = v.extract()?;
            options.insert(key, value);
        }
    }

    let mut config = BackendConfig::new(&kind)
        .map_err(|e| PyValueError::new_err(format!("{}: {}", field, e)))?;
    config.options = options;
    Ok(config)
}

fn activity_to_dict<'py>(
    py: Python<'py>,
    activity: &CacheActivity,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("hits", activity.hits)?;
    result.set_item("misses", activity.misses)?;
    result.set_item("lookup_errors", activity.lookup_errors)?;
    result.set_item("type_mismatches", activity.type_mismatches)?;
    result.set_item("loads_started", activity.loads_started)?;
    result.set_item("loads_succeeded", activity.loads_succeeded)?;
    result.set_item("loads_failed", activity.loads_failed)?;
    result.set_item("loads_cancelled", activity.loads_cancelled)?;
    result.set_item("loads_in_flight", activity.loads_in_flight)?;
    result.set_item(
        "load_success_duration_ns",
        activity.load_success_duration_ns,
    )?;
    result.set_item("load_error_duration_ns", activity.load_error_duration_ns)?;
    result.set_item(
        "load_cancelled_duration_ns",
        activity.load_cancelled_duration_ns,
    )?;
    let warm = PyDict::new(py);
    warm.set_item("attempts", activity.warm.attempts)?;
    warm.set_item("hits", activity.warm.hits)?;
    warm.set_item("loads_started", activity.warm.loads_started)?;
    warm.set_item("loads_succeeded", activity.warm.loads_succeeded)?;
    warm.set_item("loads_failed", activity.warm.loads_failed)?;
    warm.set_item("loads_cancelled", activity.warm.loads_cancelled)?;
    warm.set_item("load_bytes", activity.warm.load_bytes)?;
    warm.set_item("errors", activity.warm.errors)?;
    result.set_item("warm", warm)?;
    Ok(result)
}

fn diagnostics_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &CacheDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let backend = PyDict::new(py);
    backend.set_item("kind", diagnostics.backend.kind.as_str())?;
    backend.set_item("pool_id", diagnostics.backend.pool_id)?;
    backend.set_item("capacity_bytes", diagnostics.backend.capacity_bytes)?;
    backend.set_item("enabled", diagnostics.backend.enabled)?;
    backend.set_item("size_bytes", diagnostics.backend.size_bytes)?;
    backend.set_item("num_entries", diagnostics.backend.num_entries)?;
    backend.set_item("write_attempts", diagnostics.backend.write_attempts)?;
    backend.set_item("write_bytes", diagnostics.backend.write_bytes)?;
    backend.set_item("size_removals", diagnostics.backend.size_removals)?;
    backend.set_item("size_removed_bytes", diagnostics.backend.size_removed_bytes)?;
    backend.set_item(
        "disabled_write_rejections",
        diagnostics.backend.disabled_write_rejections,
    )?;
    backend.set_item("disabled_bypasses", diagnostics.backend.disabled_bypasses)?;
    backend.set_item(
        "lost_placeholder_rejections",
        diagnostics.backend.lost_placeholder_rejections,
    )?;
    backend.set_item("weight_saturations", diagnostics.backend.weight_saturations)?;
    backend.set_item(
        "write_rejections_complete",
        diagnostics.backend.write_rejections_complete,
    )?;
    backend.set_item("resident_evictions", diagnostics.backend.resident_evictions)?;
    backend.set_item("admissions", diagnostics.backend.admissions)?;
    backend.set_item("coalesced_loads", diagnostics.backend.coalesced_loads)?;

    let result = PyDict::new(py);
    result.set_item("activity", activity_to_dict(py, &diagnostics.activity)?)?;
    result.set_item("backend", backend)?;
    result.set_item("utilization", diagnostics.utilization)?;
    if let Some(by_type) = &diagnostics.by_type {
        let activity = PyDict::new(py);
        for item in &by_type.activity {
            activity.set_item(&item.type_name, activity_to_dict(py, &item.activity)?)?;
        }

        let detail = PyDict::new(py);
        detail.set_item("activity", activity)?;
        detail.set_item(
            "type_label_overflow_events",
            by_type.type_label_overflow_events,
        )?;
        if let Some(occupancy) = &by_type.occupancy {
            let types = PyDict::new(py);
            for item in &occupancy.types {
                let values = PyDict::new(py);
                values.set_item("size_bytes", item.size_bytes)?;
                values.set_item("num_entries", item.num_entries)?;
                types.set_item(&item.type_name, values)?;
            }
            let values = PyDict::new(py);
            values.set_item("types", types)?;
            values.set_item("untagged_size_bytes", occupancy.untagged_size_bytes)?;
            values.set_item("untagged_num_entries", occupancy.untagged_num_entries)?;
            detail.set_item("occupancy", values)?;
        } else {
            detail.set_item("occupancy", py.None())?;
        }
        result.set_item("by_type", detail)?;
    }
    Ok(result)
}

#[pymethods]
impl Session {
    #[new]
    #[pyo3(signature=(
        index_cache_size_bytes=None,
        metadata_cache_size_bytes=None,
        index_cache_backend=None,
        metadata_cache_backend=None,
        store_registry=None,
    ))]
    fn create(
        index_cache_size_bytes: Option<usize>,
        metadata_cache_size_bytes: Option<usize>,
        index_cache_backend: Option<Bound<'_, PyAny>>,
        metadata_cache_backend: Option<Bound<'_, PyAny>>,
        store_registry: Option<PyObjectStoreRegistry>,
    ) -> PyResult<Self> {
        let index_cache = resolve_cache_spec(
            "index_cache_backend",
            index_cache_backend.as_ref(),
            "index_cache_size_bytes",
            index_cache_size_bytes,
        )?;
        let metadata_cache = resolve_cache_spec(
            "metadata_cache_backend",
            metadata_cache_backend.as_ref(),
            "metadata_cache_size_bytes",
            metadata_cache_size_bytes,
        )?;
        let store_registry = store_registry.map(|r| r.inner).unwrap_or_default();
        let session =
            LanceSession::with_cache_backends(index_cache, metadata_cache, store_registry);
        Ok(Self {
            inner: Arc::new(session),
        })
    }

    fn __repr__(&self) -> String {
        let (index_cache_size, meta_cache_size) = rt()
            .block_on(None, async move {
                (
                    self.inner.index_cache_stats().await.size_bytes,
                    self.inner.metadata_cache_stats().await.size_bytes,
                )
            })
            .unwrap_or((0, 0));
        format!(
            "Session(index_cache_size_bytes={}, metadata_cache_size_bytes={})",
            index_cache_size, meta_cache_size
        )
    }

    /// Return the current size of the session in bytes
    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes()
    }

    /// Return the current size of the index cache in bytes.
    pub fn index_cache_size_bytes(&self) -> PyResult<u64> {
        rt().block_on(None, async move {
            self.inner.index_cache_stats().await.size_bytes as u64
        })
    }

    /// Inspect lifetime activity and physical accounting for both session caches.
    ///
    /// The returned dictionary contains ``index`` and ``metadata`` entries.
    /// Each has an ``activity`` dictionary, a ``backend`` dictionary, and a
    /// derived ``utilization`` value. Unsupported measurements and utilization
    /// for zero or unknown capacity are ``None``. Counts do not reset when the
    /// cache is cleared. Fields are sampled independently and may change while
    /// cache operations are running.
    /// ``activity["warm"]`` reports explicit prewarm attempts, hits, submitted
    /// materializations, accounted bytes, and cache-call errors.
    ///
    /// Set ``refresh=True`` to request backend maintenance before sampling.
    /// Set ``by_type=True`` to scan resident entries and include bounded
    /// per-type activity and occupancy. Either option can be more expensive,
    /// and the GIL is released while waiting.
    #[pyo3(signature = (refresh=false, by_type=false))]
    pub fn cache_diagnostics<'py>(
        &self,
        py: Python<'py>,
        refresh: bool,
        by_type: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = if by_type {
            let session = self.inner.clone();
            let mode = if refresh {
                CacheSnapshotMode::Refreshed
            } else {
                CacheSnapshotMode::Approximate
            };
            rt().block_on(Some(py), async move {
                session.cache_diagnostics_by_type(mode).await
            })?
        } else if refresh {
            let session = self.inner.clone();
            rt().block_on(Some(py), async move {
                session
                    .cache_diagnostics_with_mode(CacheSnapshotMode::Refreshed)
                    .await
            })?
        } else {
            self.inner.cache_diagnostics()
        };

        let result = PyDict::new(py);
        result.set_item("index", diagnostics_to_dict(py, &diagnostics.index)?)?;
        result.set_item("metadata", diagnostics_to_dict(py, &diagnostics.metadata)?)?;
        Ok(result)
    }

    /// Return whether the other session is the same as this one.
    pub fn is_same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
