// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared runtime for Rust-native Lance storage extensions.
//!
//! The runtime and plugins are Rust dylibs built with the same toolchain.
//! Plugins receive the existing `ObjectStoreRegistry` and register an
//! `Arc<dyn ObjectStoreProvider>`. There is no C storage protocol here.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use lance_core::error::{Error, Result};
use lance_io::object_store::ObjectStoreRegistry;
use libloading::{Library, Symbol};

pub use lance_core;
pub use lance_io;
pub use object_store;

/// Rust ABI entry point exported by a storage plugin.
pub type PluginEntry = unsafe extern "Rust" fn(&ObjectStoreRegistry);

/// Process-local runtime that owns the registry and loaded plugin libraries.
pub struct ExtensionRuntime {
    registry: Arc<ObjectStoreRegistry>,
    libraries: Mutex<Vec<Library>>,
}

impl Default for ExtensionRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtensionRuntime {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(ObjectStoreRegistry::empty()),
            libraries: Mutex::new(Vec::new()),
        }
    }

    /// Return the registry used by Lance's normal object-store creation path.
    pub fn registry(&self) -> Arc<ObjectStoreRegistry> {
        self.registry.clone()
    }

    /// Load a Rust plugin and keep its dylib resident.
    ///
    /// The plugin entry point receives the existing `ObjectStoreRegistry`; it
    /// registers providers with `ObjectStoreRegistry::insert`.
    pub fn load_plugin(&self, plugin_path: impl AsRef<Path>) -> Result<()> {
        let plugin_path = plugin_path.as_ref();
        let library = unsafe { Library::new(plugin_path) }.map_err(|error| {
            Error::internal(format!(
                "failed to load plugin '{}': {error}",
                plugin_path.display()
            ))
        })?;
        let entry: Symbol<PluginEntry> =
            unsafe { library.get(b"lance_plugin_entry\0") }.map_err(|error| {
                Error::internal(format!(
                    "failed to resolve plugin entry point in '{}': {error}",
                    plugin_path.display()
                ))
            })?;

        unsafe { entry(&self.registry) };

        let mut libraries = self
            .libraries
            .lock()
            .map_err(|_| Error::internal("plugin library lock poisoned"))?;
        libraries.push(library);
        Ok(())
    }
}
