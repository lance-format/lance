// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Example Rust-native storage plugin.
//!
//! The plugin registers existing `ObjectStoreProvider` implementations in the
//! existing `ObjectStoreRegistry`. There is no plugin-specific storage
//! protocol.

mod goosefs;

use std::sync::Arc;

use goosefs::GooseFsStoreProvider;
use lance_extension_runtime::lance_io::object_store::ObjectStoreRegistry;
use lance_extension_runtime::lance_io::object_store::providers::memory::MemoryStoreProvider;

/// Register providers in the existing Lance object-store registry.
///
/// This is a Rust ABI entry point. Runtime and plugin must share the same
/// compiler, target, feature set, and runtime dylib.
#[unsafe(no_mangle)]
pub extern "Rust" fn lance_plugin_entry(registry: &ObjectStoreRegistry) {
    registry.insert("demo-memory", Arc::new(MemoryStoreProvider));
    registry.insert("goosefs", Arc::new(GooseFsStoreProvider));
}
