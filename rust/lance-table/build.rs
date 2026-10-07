// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Result;

fn main() -> Result<()> {
    // Watch files explicitly: Cargo may miss changes through the protos symlink.
    println!("cargo:rerun-if-changed=protos/table.proto");
    println!("cargo:rerun-if-changed=protos/fragment_metadata.proto");
    println!("cargo:rerun-if-changed=protos/file.proto");
    println!("cargo:rerun-if-changed=protos/transaction.proto");
    println!("cargo:rerun-if-changed=protos/rowids.proto");

    #[cfg(feature = "protoc")]
    // Use vendored protobuf compiler if requested.
    unsafe {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }

    let mut prost_build = prost_build::Config::new();
    prost_build.extern_path(".lance.file", "::lance_file::format::pb");
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.enable_type_names();
    // Inline row id sequences are ~98% of a large manifest. Decoding them as
    // `Bytes` slices the fetched buffer instead of copying into a `Vec<u8>`.
    prost_build.bytes([".lance.table.DataFragment.inline_row_ids"]);
    for name in [
        "FragmentTree",
        "FragmentTreeRoot",
        "FragmentTreeChild",
        "FragmentTreeMutation",
        "FragmentAction",
        "AddDataFile",
        "RemoveDataFile",
        "ReplaceDataFile",
        "AddDeletionFile",
        "ClearDeletionFile",
        "DataFragment",
        "DataFile",
        "DataOverlayFile",
        "FieldCoverage",
        "DeletionFile",
        "ExternalFile",
        "RowLineageColumn",
    ] {
        prost_build.type_attribute(
            format!(".lance.table.{name}"),
            "#[derive(lance_core::deepsize::DeepSizeOf)]",
        );
    }
    prost_build.compile_protos(
        &[
            "./protos/table.proto",
            "./protos/fragment_metadata.proto",
            "./protos/transaction.proto",
            "./protos/rowids.proto",
        ],
        &["./protos"],
    )?;

    Ok(())
}
