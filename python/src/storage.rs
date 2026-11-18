// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Runtime registration hook for external `StorageProvider` implementations.
//!
//! This module exposes two pyclasses:
//!
//! - `PyStorageRegistry` wraps [`lance_io::storage::StorageRegistry`]
//!   and lets Python code register additional `StorageProvider`s under new
//!   URL schemes. A registry constructed here can be passed to `Session` so
//!   `lance.dataset("myscheme://...")` dispatches through the new provider.
//! - `PyStorageProvider` is a bridge that adapts a built-in Rust provider
//!   (currently just `MemoryStoreProvider`), a Python object that implements
//!   the `new_backend` protocol, or an `Arc<dyn StorageProvider>` produced
//!   by a *separate* wheel and handed across a `PyCapsule`, to
//!   `Arc<dyn StorageProvider>`.
//!
//! The Python-callable path is intentionally stubbed for this first cut: the
//! full Python-to-Rust `Storage` bridge (i.e. wrapping a Python-returned
//! object as a Lance `StorageBackend`) is a follow-up. The smoke test
//! against the built-in memory provider proves the registration + dispatch
//! plumbing works end to end.
//!
//! # Out-of-tree providers via `PyCapsule`
//!
//! [`PyStorageProvider::from_capsule`] lets an external wheel register a
//! Rust `StorageProvider` it compiled itself. The external wheel builds an
//! `Arc<dyn StorageProvider>`, wraps it in a `PyCapsule` named
//! [`PROVIDER_CAPSULE_NAME`], and passes that capsule here. Because Rust has
//! no stable ABI, this is sound **only when both wheels are built in lockstep**:
//! identical `rustc`, identical `lance-io` / `object_store` source, and
//! identical resolved dependency versions, so the trait object's vtable and the
//! types in `new_backend`'s signature have the same layout on both sides. In
//! Phase I both wheels are built locally from the same branch and toolchain, so
//! the constraint holds; distributing pre-built wheels that must interoperate
//! is deferred (a packaging-phase concern).

use std::ffi::CStr;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use lance_io::storage::providers::memory::MemoryStoreProvider;
use lance_io::storage::{StorageParams, StorageProvider, StorageRegistry};

/// Name that every capsule passed to [`PyStorageProvider::from_capsule`]
/// must carry. External wheels create their capsule with this exact name so a
/// capsule holding some unrelated pointer cannot be mistaken for a provider.
pub const PROVIDER_CAPSULE_NAME: &CStr = c"lance_storage_provider";

/// Bridge between a Python object and the Rust `StorageProvider` trait.
///
/// For the memory variant we short-circuit to a real Rust provider so the
/// smoke test can prove the scheme-dispatch plumbing works. For the Python
/// callable variant we hold a `Py<PyAny>` and (in a follow-up) will call
/// `new_backend(base_path, storage_options)` on it via the GIL.
#[derive(Debug)]
enum PyProviderBridge {
    /// Built-in `MemoryStoreProvider`, wrapped directly.
    Memory(MemoryStoreProvider),
    /// A Python object implementing the `new_backend(base_path, storage_options)`
    /// protocol. Not yet dispatchable end-to-end (see module docstring).
    ///
    /// The wrapped `Py<PyAny>` is intentionally held even though we do not
    /// invoke it yet: keeping the Python object alive here means once the
    /// bridge lands, we can dispatch without changing the enum shape.
    #[allow(dead_code)]
    PyCallable(Py<PyAny>),
}

#[async_trait::async_trait]
impl StorageProvider for PyProviderBridge {
    async fn new_backend(
        &self,
        base_path: url::Url,
        params: &StorageParams,
    ) -> lance_core::Result<lance_io::storage::BackendConfig> {
        match self {
            Self::Memory(inner) => inner.new_backend(base_path, params).await,
            Self::PyCallable(_) => Err(lance_core::Error::not_supported(
                "PyStorageProvider: the Python-callable bridge is not yet \
                 implemented. Use PyStorageProvider.memory() for the current cut.",
            )),
        }
    }
}

/// Python-facing wrapper around `Arc<dyn StorageProvider>`.
///
/// There are three ways to construct one from Python, ordered here by how
/// complete they are today:
///
/// 1. `_StorageProvider.from_capsule(capsule)` — **fully dispatches.**
///    Adopt an `Arc<dyn StorageProvider>` built by a separate,
///    ABI-compatible wheel and handed over in a `PyCapsule`. `new_backend` then
///    calls that provider's own Rust implementation directly — the
///    `PyProviderBridge` below is not involved. This is how an out-of-tree Rust
///    provider (e.g. an on-node NVMe cache) plugs in without living in the
///    Lance source tree; see the module docs for the `PyCapsule` handoff and
///    its ABI-lockstep build requirement.
/// 2. `_StorageProvider.memory()` — **fully dispatches.** Wrap the built-in
///    `MemoryStoreProvider`; registrable under any scheme and functional for
///    read/write. Primarily a test/reference vehicle.
/// 3. `_StorageProvider(py_obj)` — **stub; does not dispatch yet.** Hold a
///    Python object implementing `new_backend(base_path, storage_options)`.
///    Registration succeeds, but dispatch raises `not_supported`: the full
///    Python-to-Rust `Storage` bridge (calling back into Python from
///    `new_backend` under the GIL) is a follow-up.
#[pyclass(name = "_StorageProvider", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct PyStorageProvider {
    pub(crate) inner: Arc<dyn StorageProvider>,
}

#[pymethods]
impl PyStorageProvider {
    /// Wrap a Python object implementing the `new_backend(base_path, storage_options)`
    /// protocol. Registration will succeed, but scheme-dispatch will raise until
    /// the full Python-to-Rust `Storage` bridge is implemented.
    #[new]
    fn new(py_object: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(PyProviderBridge::PyCallable(py_object)),
        }
    }

    /// Return a provider backed by the built-in `MemoryStoreProvider`. Every
    /// call to `new_backend` allocates a fresh in-memory `object_store::InMemory`;
    /// the enclosing `StorageRegistry` caches the resulting `Storage`
    /// so writers and readers using the same scheme share storage as long as
    /// something holds a strong reference.
    #[staticmethod]
    fn memory() -> Self {
        Self {
            inner: Arc::new(PyProviderBridge::Memory(MemoryStoreProvider)),
        }
    }

    /// Adopt an `Arc<dyn StorageProvider>` carried in a `PyCapsule` created
    /// by a separate wheel. The capsule must be named [`PROVIDER_CAPSULE_NAME`]
    /// and hold exactly an `Arc<dyn StorageProvider>`.
    ///
    /// See the module docstring for the ABI-lockstep requirement: the calling
    /// wheel must be built against the identical `lance-io` / `object_store`
    /// source and toolchain as this one.
    #[staticmethod]
    fn from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        // `pointer_checked(Some(name))` asks CPython for the pointer *and*
        // requires the capsule to carry exactly this name and a non-null
        // pointer, so a foreign or misnamed capsule is rejected here rather
        // than dereferenced. (Passing `None` asks for a *nameless* capsule and
        // would reject every correctly-named one.)
        let ptr = capsule
            .pointer_checked(Some(PROVIDER_CAPSULE_NAME))
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "expected a PyCapsule named {:?}: {e}",
                    PROVIDER_CAPSULE_NAME.to_string_lossy(),
                ))
            })?;

        // SAFETY: by the capsule-name contract above, the capsule carries an
        // `Arc<dyn StorageProvider>` built against the identical lance-io /
        // object_store types (same source, rustc, and resolved dependency
        // versions). We dereference only long enough to clone the `Arc`
        // (bumping the strong count); the capsule keeps its own reference and
        // its destructor drops that on GC.
        let provider = unsafe { ptr.cast::<Arc<dyn StorageProvider>>().as_ref() };
        Ok(Self {
            inner: provider.clone(),
        })
    }

    /// Test/reference producer: wrap the built-in memory provider in a
    /// `PyCapsule` named [`PROVIDER_CAPSULE_NAME`], mirroring what an external,
    /// ABI-compatible wheel emits. Lets `from_capsule` be exercised end to end
    /// from Python without a second wheel in the tree.
    #[staticmethod]
    fn _memory_capsule(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
        let provider: Arc<dyn StorageProvider> =
            Arc::new(PyProviderBridge::Memory(MemoryStoreProvider));
        PyCapsule::new_with_value(py, provider, PROVIDER_CAPSULE_NAME)
    }

    fn __repr__(&self) -> String {
        format!("_StorageProvider({:?})", self.inner)
    }
}

/// Python-facing wrapper around `Arc<StorageRegistry>`.
///
/// A new instance starts from `StorageRegistry::default()`, so all
/// built-in schemes (memory, file, and any of s3/az/gs/oss/... enabled at
/// build time) are already registered. Additional providers can be inserted
/// under new (or overridden) schemes via `register_provider`.
///
/// Pass an instance as the `store_registry` argument of `Session(...)` to
/// make its schemes visible to `lance.dataset(uri, session=...)` and
/// `lance.write_dataset(..., uri, session=...)`.
#[pyclass(name = "_StorageRegistry", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct PyStorageRegistry {
    pub(crate) inner: Arc<StorageRegistry>,
}

#[pymethods]
impl PyStorageRegistry {
    /// Create a new registry pre-populated with the built-in schemes.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(StorageRegistry::default()),
        }
    }

    /// Register a provider under a scheme. Idempotent: registering the same
    /// scheme again replaces the previous provider. Registering under a
    /// built-in scheme (e.g. `"memory"`) overrides that built-in.
    fn register_provider(&self, scheme: &str, provider: &PyStorageProvider) -> PyResult<()> {
        if scheme.is_empty() {
            return Err(PyValueError::new_err("scheme must be a non-empty string"));
        }
        self.inner.insert(scheme, provider.inner.clone());
        Ok(())
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "_StorageRegistry(active_stores={}, hits={}, misses={})",
            stats.active_stores, stats.hits, stats.misses,
        )
    }
}
