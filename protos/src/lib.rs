// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Protobuf definitions for the Lance columnar format.
//!
//! The crate packages the `.proto` sources without depending on the Lance
//! runtime crates. Build scripts can use [`proto_dir`] as a protoc include
//! path to compile schemas that import Lance protobufs without compiling
//! Lance or DataFusion for the host.

use std::path::Path;

/// Returns the directory containing the packaged `.proto` sources.
///
/// Use this directory as an include path when compiling protobuf definitions
/// that import Lance schemas. It belongs to this dependency's resolved source
/// and does not require a separate Lance checkout.
///
/// ```
/// let includes = lance_protos::proto_dir();
/// assert!(includes.join("table_identifier.proto").is_file());
/// ```
pub fn proto_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[cfg(test)]
mod tests {
    use super::proto_dir;

    #[test]
    fn proto_dir_contains_execution_plan_import_closure() {
        for file in [
            "ann.proto",
            "clustering.proto",
            "file.proto",
            "filtered_read.proto",
            "fragment_metadata.proto",
            "index.proto",
            "index_old.proto",
            "table.proto",
            "table_identifier.proto",
        ] {
            assert!(
                proto_dir().join(file).is_file(),
                "missing packaged proto {file}"
            );
        }
    }
}
