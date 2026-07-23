// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    #[cfg(feature = "protoc")]
    // Use vendored protobuf compiler if requested.
    unsafe {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }

    let new_config = || {
        let mut prost_build = prost_build::Config::new();
        prost_build.extern_path(".lance.file", "::lance_file::format::pb");
        prost_build.protoc_arg("--experimental_allow_proto3_optional");
        prost_build.enable_type_names();
        prost_build
    };

    // `lance.table` and `lance.transaction` reference each other. An extern
    // path resolves each package to the module that includes its generated
    // code, but it also stops prost from generating that package, hence one
    // pass per package.
    let mut prost_build = new_config();
    prost_build.extern_path(".lance.transaction", "crate::format::pb_transaction");
    // Inline row id sequences are ~98% of a large manifest. Decoding them as
    // `Bytes` slices the fetched buffer instead of copying into a `Vec<u8>`.
    prost_build.bytes([".lance.table.DataFragment.inline_row_ids"]);
    prost_build.compile_protos(
        &[
            "./protos/table.proto",
            "./protos/transaction/transaction.proto",
            "./protos/rowids.proto",
        ],
        &["./protos"],
    )?;

    let mut prost_build = new_config();
    prost_build.extern_path(".lance.table", "crate::format::pb");
    prost_build.compile_protos(
        &[
            "./protos/transaction/common.proto",
            "./protos/transaction/actions.proto",
        ],
        &["./protos"],
    )?;

    Ok(())
}
