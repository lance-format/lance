// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::process;
use std::process::{Command, Stdio};

const LICENSE_HEADER: &str = "// SPDX-License-Identifier: Apache-2.0\n\
// SPDX-FileCopyrightText: Copyright The Lance Authors\n\n";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let check = match args.as_slice() {
        [] => false,
        [argument] if argument == "--check" => true,
        _ => return Err("usage: lance-proto-codegen [--check]".into()),
    };
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("codegen crate should be under <repo>/protos/codegen");
    let temp_root = std::env::temp_dir().join(format!("lance-proto-codegen-{}", process::id()));

    if temp_root.exists() {
        fs::remove_dir_all(&temp_root)?;
    }
    fs::create_dir_all(&temp_root)?;

    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // SAFETY: this single-threaded tool sets PROTOC before invoking prost-build
    // and does not read or mutate the environment concurrently.
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }

    generate_encoding(repo_root, &temp_root, check)?;
    generate_file(repo_root, &temp_root, check)?;
    generate_table(repo_root, &temp_root, check)?;
    generate_index(repo_root, &temp_root, check)?;
    generate_datafusion(repo_root, &temp_root, check)?;
    generate_lance(repo_root, &temp_root, check)?;

    fs::remove_dir_all(temp_root)?;
    Ok(())
}

fn config(out_dir: &Path) -> io::Result<prost_build::Config> {
    fs::create_dir_all(out_dir)?;
    let mut config = prost_build::Config::new();
    config.out_dir(out_dir);
    config.protoc_arg("--experimental_allow_proto3_optional");
    Ok(config)
}

fn generate_encoding(
    repo_root: &Path,
    temp_root: &Path,
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = temp_root.join("lance-encoding");
    let proto_dir = repo_root.join("protos");
    let mut config = config(&out_dir)?;
    config.enable_type_names();
    config.bytes(["."]);
    config.compile_protos(&[proto_dir.join("encodings_v2_0.proto")], &[&proto_dir])?;
    config.compile_protos(&[proto_dir.join("encodings_v2_1.proto")], &[&proto_dir])?;
    config.compile_protos(&[proto_dir.join("encodings_v2_3.proto")], &[&proto_dir])?;
    install_generated(
        &out_dir,
        &repo_root.join("rust/lance-encoding/src/pb"),
        &[
            "lance.encodings.rs",
            "lance.encodings21.rs",
            "lance.encodings23.rs",
        ],
        check,
    )
}

fn generate_file(
    repo_root: &Path,
    temp_root: &Path,
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = temp_root.join("lance-file");
    let proto_dir = repo_root.join("protos");
    let mut config = config(&out_dir)?;
    config.extern_path(".lance.encodings", "::lance_encoding::format::pb");
    config.compile_protos(
        &[
            proto_dir.join("file.proto"),
            proto_dir.join("file2.proto"),
            proto_dir.join("encodings_v2_0.proto"),
            proto_dir.join("encodings_v2_1.proto"),
        ],
        &[&proto_dir],
    )?;
    install_generated(
        &out_dir,
        &repo_root.join("rust/lance-file/src/pb"),
        &["lance.file.rs", "lance.file.v2.rs"],
        check,
    )
}

fn generate_table(
    repo_root: &Path,
    temp_root: &Path,
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = temp_root.join("lance-table");
    let proto_dir = repo_root.join("protos");
    let new_config = || -> io::Result<prost_build::Config> {
        let mut config = config(&out_dir)?;
        config.extern_path(".lance.file", "::lance_file::format::pb");
        config.enable_type_names();
        Ok(config)
    };

    // `lance.table` and `lance.transaction` reference each other. An extern
    // path resolves each package to the module that includes its generated
    // code, but it also stops prost from generating that package, hence one
    // pass per package.
    let mut config = new_config()?;
    config.extern_path(".lance.transaction", "crate::format::pb_actions");
    config.bytes([
        ".lance.table.DataFragment.inline_row_ids",
        ".lance.table.DeletionFile.inline",
    ]);
    config.compile_protos(
        &[
            proto_dir.join("table.proto"),
            proto_dir.join("transaction/transaction.proto"),
            proto_dir.join("rowids.proto"),
        ],
        &[&proto_dir],
    )?;

    let mut config = new_config()?;
    config.extern_path(".lance.table", "crate::format::pb");
    config.compile_protos(
        &[
            proto_dir.join("transaction/common.proto"),
            proto_dir.join("transaction/actions.proto"),
        ],
        &[&proto_dir],
    )?;

    install_generated(
        &out_dir,
        &repo_root.join("rust/lance-table/src/pb"),
        &["lance.table.rs", "lance.transaction.rs"],
        check,
    )
}

fn generate_index(
    repo_root: &Path,
    temp_root: &Path,
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = temp_root.join("lance-index");
    let proto_dir = repo_root.join("protos");
    let cache_proto_dir = repo_root.join("rust/lance-index/protos-cache");
    let mut config = config(&out_dir)?;
    config.enable_type_names();
    config.compile_protos(&[proto_dir.join("index_old.proto")], &[&proto_dir])?;
    config.extern_path(".lance.table", "crate::pbold");
    config.compile_protos(
        &[
            proto_dir.join("index.proto"),
            cache_proto_dir.join("cache.proto"),
        ],
        &[&proto_dir, &cache_proto_dir],
    )?;
    install_generated(
        &out_dir,
        &repo_root.join("rust/lance-index/src/pb"),
        &[
            "lance.index.pb.rs",
            "lance.table.rs",
            "lance.index.cache.rs",
        ],
        check,
    )
}

fn generate_datafusion(
    repo_root: &Path,
    temp_root: &Path,
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = temp_root.join("lance-datafusion");
    let proto_dir = repo_root.join("protos");
    let mut config = config(&out_dir)?;
    config.enable_type_names();
    config.message_attribute(
        ".lance.datafusion.FilteredReadExecV2Proto",
        "#[non_exhaustive]",
    );
    config.compile_protos(
        &[
            proto_dir.join("table_identifier.proto"),
            proto_dir.join("filtered_read.proto"),
        ],
        &[&proto_dir],
    )?;
    install_generated(
        &out_dir,
        &repo_root.join("rust/lance-datafusion/src/pb"),
        &["lance.datafusion.rs"],
        check,
    )
}

fn generate_lance(
    repo_root: &Path,
    temp_root: &Path,
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = temp_root.join("lance");
    let proto_dir = repo_root.join("protos");
    let mut config = config(&out_dir)?;
    config.extern_path(".lance.table", "::lance_table::format::pb");
    config.extern_path(".lance.index.pb", "::lance_index::pb");
    config.extern_path(".lance.datafusion", "::lance_datafusion::pb");
    config.enable_type_names();
    config.compile_protos(
        &[
            proto_dir.join("ann.proto"),
            proto_dir.join("fts.proto"),
            proto_dir.join("cache_diagnostics.proto"),
        ],
        &[&proto_dir],
    )?;
    install_generated(
        &out_dir,
        &repo_root.join("rust/lance/src/pb"),
        &["lance.pb.rs", "lance.cache.rs"],
        check,
    )
}

fn install_generated(
    generated_dir: &Path,
    destination_dir: &Path,
    file_names: &[&str],
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !check {
        fs::create_dir_all(destination_dir)?;
    }
    for file_name in file_names {
        let generated = fs::read_to_string(generated_dir.join(file_name))?;
        let expected = normalize_generated(&format!("{LICENSE_HEADER}{generated}"))?;
        let destination = destination_dir.join(file_name);
        if check {
            let checked_in = fs::read_to_string(&destination)?;
            if checked_in != expected {
                return Err(io::Error::other(format!(
                    "{} is out of date; regenerate protobuf sources",
                    destination.display()
                ))
                .into());
            }
        } else {
            fs::write(destination, expected)?;
        }
    }
    Ok(())
}

/// Format generated sources with the repository toolchain's rustfmt.
///
/// Pre-commit runs `cargo fmt` over changed `.rs` files, including the
/// `include!`d files under `src/pb`. Normalizing here keeps regeneration and
/// repository formatting from rewriting each other, so `--check` still detects
/// stale sources after hooks run.
fn normalize_generated(source: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .expect("rustfmt stdin should be piped")
        .write_all(source.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(String::from_utf8(output.stdout)?)
}
