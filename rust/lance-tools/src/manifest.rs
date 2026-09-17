// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cli::LanceTableManifestArgs;
use chrono::{DateTime, Utc};
use lance_core::Result;
use lance_io::object_store::ObjectStore;
use lance_io::utils::read_message;
use lance_table::feature_flags::{
    FLAG_BASE_PATHS, FLAG_DELETION_FILES, FLAG_DISABLE_TRANSACTION_FILE, FLAG_STABLE_ROW_IDS,
    FLAG_TABLE_CONFIG, FLAG_UNKNOWN, FLAG_UNSTABLE_DATA_OVERLAY_FILES,
    FLAG_USE_V2_FORMAT_DEPRECATED,
};
use lance_table::format::overlay::OverlayCoverage;
use lance_table::format::pb;
use lance_table::format::{
    DataFile, DataStorageFormat, DeletionFile, ExternalFile, Fragment, IndexMetadata, Manifest,
    RowDatasetVersionMeta, RowIdMeta, WriterVersion,
};
use lance_table::io::manifest::read_manifest_proto;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fmt::Formatter;
use std::io::Write;
use std::sync::Arc;

const TRANSACTIONS_DIR: &str = "_transactions";
const VERSIONS_DIR: &str = "_versions";
const LATEST_MANIFEST: &str = "_latest.manifest";

pub(crate) struct LanceToolManifest {
    path: Path,
    dataset_base: Option<Path>,
    has_timestamp: bool,
    has_data_format: bool,
    manifest: Manifest,
    indices: OptionalSection<Vec<IndexView>>,
    transaction: OptionalSection<TransactionView>,
}

enum OptionalSection<T> {
    NotPresent,
    Loaded(T),
    Error(String),
}

struct IndexView {
    raw: pb::IndexMetadata,
    parsed: std::result::Result<IndexMetadata, String>,
}

struct TransactionView {
    source: TransactionSource,
    transaction: pb::Transaction,
}

enum TransactionSource {
    Inline { position: usize },
    External { path: Path },
}

struct ManifestSource {
    object_store: Arc<ObjectStore>,
    manifest_path: Path,
    dataset_base: Option<Path>,
}

impl LanceToolManifest {
    async fn open(args: &LanceTableManifestArgs) -> Result<Self> {
        let source = resolve_manifest_source(&args.source).await?;
        let raw_manifest =
            read_manifest_proto(&source.object_store, &source.manifest_path, None).await?;
        let has_timestamp = raw_manifest.timestamp.is_some();
        let has_data_format = raw_manifest.data_format.is_some();
        let manifest = Manifest::try_from(raw_manifest)?;
        let indices = load_indices(
            &source.object_store,
            &source.manifest_path,
            manifest.index_section,
        )
        .await;
        let transaction = load_transaction(
            &source.object_store,
            &source.manifest_path,
            source.dataset_base.as_ref(),
            &manifest,
        )
        .await;

        Ok(Self {
            path: source.manifest_path,
            dataset_base: source.dataset_base,
            has_timestamp,
            has_data_format,
            manifest,
            indices,
            transaction,
        })
    }
}

async fn resolve_manifest_source(source: &str) -> Result<ManifestSource> {
    let (object_store, path) = crate::util::get_object_store_and_path(&source.to_string()).await?;
    let dataset_base = infer_dataset_base_from_manifest_path(&path);
    Ok(ManifestSource {
        object_store,
        manifest_path: path,
        dataset_base,
    })
}

/// Infer the dataset base from `<base>/_latest.manifest` or
/// `<base>/_versions/*.manifest`.
///
/// Returns `None` when the dataset base cannot be inferred, which means
/// external transaction files cannot be resolved. Root-level manifests also
/// return `None` because [`path_from_parts`] rejects an empty prefix.
fn infer_dataset_base_from_manifest_path(path: &Path) -> Option<Path> {
    let parts = path
        .parts()
        .map(|part| part.as_ref().to_string())
        .collect::<Vec<_>>();
    let filename = parts.last()?;
    if filename == LATEST_MANIFEST {
        return path_from_parts(&parts[..parts.len().saturating_sub(1)]);
    }
    if filename.ends_with(".manifest") && parts.len() >= 2 && parts[parts.len() - 2] == VERSIONS_DIR
    {
        return path_from_parts(&parts[..parts.len() - 2]);
    }
    None
}

/// Build a non-empty object-store path from path parts.
///
/// Returns `None` for an empty prefix, including root-level `_latest.manifest`
/// paths where there is no dataset base before the manifest filename.
fn path_from_parts(parts: &[String]) -> Option<Path> {
    if parts.is_empty() {
        None
    } else {
        Some(Path::from_iter(parts.iter().map(|part| part.as_str())))
    }
}

async fn load_indices(
    object_store: &ObjectStore,
    manifest_path: &Path,
    index_section: Option<usize>,
) -> OptionalSection<Vec<IndexView>> {
    let Some(index_section) = index_section else {
        return OptionalSection::NotPresent;
    };
    match read_section::<pb::IndexSection>(object_store, manifest_path, index_section).await {
        Ok(section) => OptionalSection::Loaded(
            section
                .indices
                .into_iter()
                .map(|raw| {
                    let parsed = IndexMetadata::try_from(raw.clone()).map_err(|e| e.to_string());
                    IndexView { raw, parsed }
                })
                .collect(),
        ),
        Err(e) => OptionalSection::Error(e.to_string()),
    }
}

async fn load_transaction(
    object_store: &ObjectStore,
    manifest_path: &Path,
    dataset_base: Option<&Path>,
    manifest: &Manifest,
) -> OptionalSection<TransactionView> {
    if let Some(position) = manifest.transaction_section {
        match read_section::<pb::Transaction>(object_store, manifest_path, position).await {
            Ok(transaction) => OptionalSection::Loaded(TransactionView {
                source: TransactionSource::Inline { position },
                transaction,
            }),
            Err(e) => OptionalSection::Error(e.to_string()),
        }
    } else if let Some(transaction_file) = &manifest.transaction_file {
        let Some(dataset_base) = dataset_base else {
            return OptionalSection::Error(format!(
                "transaction_file is set to {}, but the dataset base path could not be inferred from manifest path {}",
                transaction_file, manifest_path
            ));
        };
        let path = dataset_base
            .clone()
            .join(TRANSACTIONS_DIR)
            .join(transaction_file.as_str());
        match read_external_transaction(object_store, &path).await {
            Ok(transaction) => OptionalSection::Loaded(TransactionView {
                source: TransactionSource::External { path },
                transaction,
            }),
            Err(error) => OptionalSection::Error(error),
        }
    } else {
        OptionalSection::NotPresent
    }
}

async fn read_external_transaction(
    object_store: &ObjectStore,
    path: &Path,
) -> std::result::Result<pb::Transaction, String> {
    let result = object_store
        .inner
        .get(path)
        .await
        .map_err(|error| format!("failed to read external transaction {path}: {error}"))?;
    let data = result.bytes().await.map_err(|error| {
        format!("failed to read bytes for external transaction {path}: {error}")
    })?;
    pb::Transaction::decode(data)
        .map_err(|error| format!("failed to decode external transaction {path}: {error}"))
}

async fn read_section<M: Message + Default>(
    object_store: &ObjectStore,
    manifest_path: &Path,
    position: usize,
) -> Result<M> {
    let reader = object_store.open(manifest_path).await?;
    read_message(reader.as_ref(), position).await
}

impl fmt::Display for LanceToolManifest {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "manifest:")?;
        write_kv(f, 2, "path", self.path.as_ref())?;
        write_kv(
            f,
            2,
            "dataset_base",
            &format_optional_path(self.dataset_base.as_ref()),
        )?;
        write_kv(f, 2, "version", &self.manifest.version.to_string())?;
        write_kv(
            f,
            2,
            "branch",
            &format_optional(self.manifest.branch.as_ref()),
        )?;
        write_kv(
            f,
            2,
            "writer_version",
            &format_writer_version(self.manifest.writer_version.as_ref()),
        )?;
        write_kv(
            f,
            2,
            "timestamp",
            &format_timestamp(self.manifest.timestamp_nanos, self.has_timestamp),
        )?;
        write_kv(f, 2, "tag", &format_optional(self.manifest.tag.as_ref()))?;
        write_kv(
            f,
            2,
            "data_storage_format",
            &format_data_storage_format(&self.manifest.data_storage_format, self.has_data_format),
        )?;
        write_kv(
            f,
            2,
            "reader_feature_flags",
            &format_feature_flags(self.manifest.reader_feature_flags),
        )?;
        write_kv(
            f,
            2,
            "writer_feature_flags",
            &format_feature_flags(self.manifest.writer_feature_flags),
        )?;
        write_kv(
            f,
            2,
            "max_fragment_id_stored",
            &format_option_u32(self.manifest.max_fragment_id),
        )?;
        write_kv(
            f,
            2,
            "max_fragment_id_effective",
            &format_option_u64(self.manifest.max_fragment_id()),
        )?;
        write_kv(
            f,
            2,
            "max_field_id",
            &self.manifest.max_field_id().to_string(),
        )?;
        write_kv(f, 2, "next_row_id", &self.manifest.next_row_id.to_string())?;
        write_kv(
            f,
            2,
            "version_aux_data_position",
            &format_position(self.manifest.version_aux_data),
        )?;
        write_kv(
            f,
            2,
            "index_section_position",
            &format_absent_usize(self.manifest.index_section),
        )?;
        write_kv(
            f,
            2,
            "transaction_file",
            &format_optional(self.manifest.transaction_file.as_ref()),
        )?;
        write_kv(
            f,
            2,
            "transaction_section_position",
            &format_absent_usize(self.manifest.transaction_section),
        )?;

        write_summary(f, &self.manifest)?;
        write_string_map(f, 2, "config", &self.manifest.config)?;
        write_string_map(f, 2, "table_metadata", &self.manifest.table_metadata)?;
        write_string_map(f, 2, "schema_metadata", &self.manifest.schema.metadata)?;
        write_base_paths(f, &self.manifest.base_paths)?;
        write_schema(f, &self.manifest)?;
        write_fragments(f, &self.manifest)?;
        write_indices(f, &self.indices)?;
        write_transaction(f, &self.transaction)?;
        Ok(())
    }
}

fn write_summary(f: &mut Formatter<'_>, manifest: &Manifest) -> fmt::Result {
    let summary = manifest.summary();
    writeln!(f, "  summary:")?;
    write_kv(
        f,
        4,
        "total_fragments",
        &summary.total_fragments.to_string(),
    )?;
    write_kv(
        f,
        4,
        "total_data_files",
        &summary.total_data_files.to_string(),
    )?;
    write_kv(
        f,
        4,
        "total_files_size",
        &summary.total_files_size.to_string(),
    )?;
    write_kv(
        f,
        4,
        "total_deletion_files",
        &summary.total_deletion_files.to_string(),
    )?;
    write_kv(
        f,
        4,
        "total_data_file_rows",
        &summary.total_data_file_rows.to_string(),
    )?;
    write_kv(
        f,
        4,
        "total_deletion_file_rows",
        &summary.total_deletion_file_rows.to_string(),
    )?;
    write_kv(f, 4, "total_rows", &summary.total_rows.to_string())
}

fn write_base_paths(
    f: &mut Formatter<'_>,
    base_paths: &HashMap<u32, lance_table::format::BasePath>,
) -> fmt::Result {
    writeln!(f, "  base_paths: {}", base_paths.len())?;
    let mut sorted = base_paths.values().collect::<Vec<_>>();
    sorted.sort_by_key(|base_path| base_path.id);
    for base_path in sorted {
        writeln!(f, "    - id: {}", base_path.id)?;
        write_kv(f, 6, "name", &format_optional(base_path.name.as_ref()))?;
        write_kv(
            f,
            6,
            "is_dataset_root",
            &base_path.is_dataset_root.to_string(),
        )?;
        write_kv(f, 6, "path", &base_path.path)?;
    }
    Ok(())
}

fn write_schema(f: &mut Formatter<'_>, manifest: &Manifest) -> fmt::Result {
    writeln!(f, "  schema:")?;
    write_indented_block(f, 4, &manifest.schema.to_string())
}

fn write_fragments(f: &mut Formatter<'_>, manifest: &Manifest) -> fmt::Result {
    writeln!(f, "  fragments: {}", manifest.fragments.len())?;
    for fragment in manifest.fragments.iter() {
        write_fragment(f, 4, fragment)?;
    }
    Ok(())
}

fn write_fragment(f: &mut Formatter<'_>, indent: usize, fragment: &Fragment) -> fmt::Result {
    writeln!(f, "{:indent$}- id: {}", "", fragment.id, indent = indent)?;
    write_kv(
        f,
        indent + 2,
        "physical_rows",
        &format_unknown_usize(fragment.physical_rows),
    )?;
    write_kv(
        f,
        indent + 2,
        "num_rows",
        &format_unknown_usize(fragment.num_rows()),
    )?;
    write_kv(
        f,
        indent + 2,
        "row_id_meta",
        &format_row_id_meta(fragment.row_id_meta.as_ref()),
    )?;
    write_kv(
        f,
        indent + 2,
        "last_updated_at_version_meta",
        &format_row_dataset_version_meta(fragment.last_updated_at_version_meta.as_ref()),
    )?;
    write_kv(
        f,
        indent + 2,
        "created_at_version_meta",
        &format_row_dataset_version_meta(fragment.created_at_version_meta.as_ref()),
    )?;
    write_deletion_file(f, indent + 2, fragment.deletion_file.as_ref())?;
    writeln!(
        f,
        "{:indent$}files: {}",
        "",
        fragment.files.len(),
        indent = indent + 2
    )?;
    for data_file in &fragment.files {
        write_data_file(f, indent + 4, data_file)?;
    }
    writeln!(
        f,
        "{:indent$}overlays: {}",
        "",
        fragment.overlays.len(),
        indent = indent + 2
    )?;
    for overlay in &fragment.overlays {
        writeln!(
            f,
            "{:indent$}- committed_version: {}",
            "",
            overlay.committed_version,
            indent = indent + 4
        )?;
        write_data_file(f, indent + 6, &overlay.data_file)?;
        write_kv(
            f,
            indent + 6,
            "coverage",
            &format_overlay_coverage(&overlay.coverage),
        )?;
    }
    Ok(())
}

fn write_data_file(f: &mut Formatter<'_>, indent: usize, data_file: &DataFile) -> fmt::Result {
    writeln!(
        f,
        "{:indent$}- path: {}",
        "",
        data_file.path,
        indent = indent
    )?;
    write_kv(
        f,
        indent + 2,
        "fields",
        &format!("{:?}", data_file.fields.as_ref()),
    )?;
    write_kv(
        f,
        indent + 2,
        "column_indices",
        &format!("{:?}", data_file.column_indices.as_ref()),
    )?;
    write_kv(
        f,
        indent + 2,
        "file_version",
        &format!(
            "{}.{}",
            data_file.file_major_version, data_file.file_minor_version
        ),
    )?;
    write_kv(
        f,
        indent + 2,
        "file_size_bytes",
        &format_cached_size(data_file.file_size_bytes.get()),
    )?;
    write_kv(
        f,
        indent + 2,
        "base_id",
        &format_option_u32(data_file.base_id),
    )
}

fn write_deletion_file(
    f: &mut Formatter<'_>,
    indent: usize,
    deletion_file: Option<&DeletionFile>,
) -> fmt::Result {
    let Some(deletion_file) = deletion_file else {
        return write_kv(f, indent, "deletion_file", "absent");
    };
    writeln!(f, "{:indent$}deletion_file:", "", indent = indent)?;
    write_kv(
        f,
        indent + 2,
        "read_version",
        &deletion_file.read_version.to_string(),
    )?;
    write_kv(f, indent + 2, "id", &deletion_file.id.to_string())?;
    write_kv(
        f,
        indent + 2,
        "file_type",
        &format!("{:?}", deletion_file.file_type),
    )?;
    write_kv(
        f,
        indent + 2,
        "num_deleted_rows",
        &format_unknown_usize(deletion_file.num_deleted_rows),
    )?;
    write_kv(
        f,
        indent + 2,
        "base_id",
        &format_option_u32(deletion_file.base_id),
    )
}

fn write_indices(f: &mut Formatter<'_>, indices: &OptionalSection<Vec<IndexView>>) -> fmt::Result {
    match indices {
        OptionalSection::NotPresent => writeln!(f, "  indices: absent"),
        OptionalSection::Error(error) => writeln!(f, "  indices_error: {}", error),
        OptionalSection::Loaded(indices) => {
            writeln!(f, "  indices: {}", indices.len())?;
            for index in indices {
                write_index(f, index)?;
            }
            Ok(())
        }
    }
}

fn write_index(f: &mut Formatter<'_>, index: &IndexView) -> fmt::Result {
    match &index.parsed {
        Ok(parsed) => {
            writeln!(f, "    - uuid: {}", parsed.uuid)?;
            write_kv(f, 6, "name", &parsed.name)?;
            write_kv(f, 6, "fields", &format!("{:?}", parsed.fields))?;
            write_kv(f, 6, "dataset_version", &parsed.dataset_version.to_string())?;
            let fragment_bitmap = parsed
                .fragment_bitmap
                .as_ref()
                .map(|bitmap| {
                    format!(
                        "{} fragments ({} serialized bytes)",
                        bitmap.len(),
                        bitmap.serialized_size()
                    )
                })
                .unwrap_or_else(|| "unknown".to_string());
            write_kv(f, 6, "fragment_bitmap", &fragment_bitmap)?;
            let details = parsed
                .index_details
                .as_ref()
                .map(|details| format!("{} ({} bytes)", details.type_url, details.value.len()))
                .unwrap_or_else(|| "absent".to_string());
            write_kv(f, 6, "index_details", &details)?;
            write_kv(f, 6, "index_version", &parsed.index_version.to_string())?;
            write_kv(
                f,
                6,
                "created_at",
                &parsed
                    .created_at
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_else(|| "absent".to_string()),
            )?;
            write_kv(f, 6, "base_id", &format_option_u32(parsed.base_id))?;
            write_index_files(f, parsed.files.as_ref())
        }
        Err(error) => {
            writeln!(f, "    - raw_index_metadata:")?;
            write_kv(f, 8, "parse_error", error)?;
            write_kv(f, 8, "name", &index.raw.name)?;
            write_kv(f, 8, "fields", &format!("{:?}", index.raw.fields))?;
            write_kv(
                f,
                8,
                "dataset_version",
                &index.raw.dataset_version.to_string(),
            )?;
            write_kv(
                f,
                8,
                "uuid_byte_count",
                &index
                    .raw
                    .uuid
                    .as_ref()
                    .map(|uuid| uuid.uuid.len().to_string())
                    .unwrap_or_else(|| "absent".to_string()),
            )?;
            write_kv(
                f,
                8,
                "fragment_bitmap_bytes",
                &index.raw.fragment_bitmap.len().to_string(),
            )?;
            write_kv(
                f,
                8,
                "index_details",
                &index
                    .raw
                    .index_details
                    .as_ref()
                    .map(|details| format!("{} ({} bytes)", details.type_url, details.value.len()))
                    .unwrap_or_else(|| "absent".to_string()),
            )?;
            write_kv(
                f,
                8,
                "index_version",
                &index
                    .raw
                    .index_version
                    .map(|version| version.to_string())
                    .unwrap_or_else(|| "absent".to_string()),
            )?;
            write_kv(
                f,
                8,
                "created_at_millis",
                &index
                    .raw
                    .created_at
                    .map(|created_at| created_at.to_string())
                    .unwrap_or_else(|| "absent".to_string()),
            )?;
            write_kv(f, 8, "base_id", &format_option_u32(index.raw.base_id))?;
            write_kv(f, 8, "files", &index.raw.files.len().to_string())
        }
    }
}

fn write_index_files(
    f: &mut Formatter<'_>,
    files: Option<&Vec<lance_table::format::IndexFile>>,
) -> fmt::Result {
    let Some(files) = files else {
        return write_kv(f, 6, "files", "unknown");
    };
    writeln!(f, "      files: {}", files.len())?;
    for file in files {
        writeln!(f, "        - path: {}", file.path)?;
        write_kv(f, 10, "size_bytes", &file.size_bytes.to_string())?;
    }
    Ok(())
}

fn write_transaction(
    f: &mut Formatter<'_>,
    transaction: &OptionalSection<TransactionView>,
) -> fmt::Result {
    match transaction {
        OptionalSection::NotPresent => writeln!(f, "  transaction: absent"),
        OptionalSection::Error(error) => writeln!(f, "  transaction_error: {}", error),
        OptionalSection::Loaded(transaction) => {
            writeln!(f, "  transaction:")?;
            match &transaction.source {
                TransactionSource::Inline { position } => {
                    write_kv(f, 4, "source", &format!("inline at position {position}"))?;
                }
                TransactionSource::External { path } => {
                    write_kv(f, 4, "source", &format!("external {}", path))?;
                }
            }
            write_transaction_body(f, 4, &transaction.transaction)
        }
    }
}

fn write_transaction_body(
    f: &mut Formatter<'_>,
    indent: usize,
    transaction: &pb::Transaction,
) -> fmt::Result {
    write_kv(
        f,
        indent,
        "read_version",
        &transaction.read_version.to_string(),
    )?;
    write_kv(f, indent, "uuid", &transaction.uuid)?;
    write_kv(
        f,
        indent,
        "tag",
        &if transaction.tag.is_empty() {
            "absent".to_string()
        } else {
            transaction.tag.clone()
        },
    )?;
    write_string_map(
        f,
        indent,
        "transaction_properties",
        &transaction.transaction_properties,
    )?;
    write_transaction_operation(f, indent, transaction.operation.as_ref())
}

fn write_transaction_operation(
    f: &mut Formatter<'_>,
    indent: usize,
    operation: Option<&pb::transaction::Operation>,
) -> fmt::Result {
    let Some(operation) = operation else {
        return write_kv(f, indent, "operation", "absent");
    };
    match operation {
        pb::transaction::Operation::Append(append) => {
            write_kv(f, indent, "operation", "append")?;
            write_pb_fragments(f, indent, "fragments", &append.fragments)
        }
        pb::transaction::Operation::Delete(delete) => {
            write_kv(f, indent, "operation", "delete")?;
            write_pb_fragments(f, indent, "updated_fragments", &delete.updated_fragments)?;
            write_kv(
                f,
                indent,
                "deleted_fragment_ids",
                &format!("{:?}", delete.deleted_fragment_ids),
            )?;
            write_kv(f, indent, "predicate", &delete.predicate)
        }
        pb::transaction::Operation::Overwrite(overwrite) => {
            write_kv(f, indent, "operation", "overwrite")?;
            write_pb_fragments(f, indent, "fragments", &overwrite.fragments)?;
            write_kv(
                f,
                indent,
                "schema_fields",
                &overwrite.schema.len().to_string(),
            )?;
            write_kv(
                f,
                indent,
                "schema_metadata",
                &format_bytes_map_summary(&overwrite.schema_metadata),
            )?;
            write_string_map(
                f,
                indent,
                "config_upsert_values",
                &overwrite.config_upsert_values,
            )?;
            write_kv(
                f,
                indent,
                "initial_bases",
                &overwrite.initial_bases.len().to_string(),
            )
        }
        pb::transaction::Operation::CreateIndex(create_index) => {
            write_kv(f, indent, "operation", "create_index")?;
            write_kv(
                f,
                indent,
                "new_indices",
                &create_index.new_indices.len().to_string(),
            )?;
            write_kv(
                f,
                indent,
                "removed_indices",
                &create_index.removed_indices.len().to_string(),
            )
        }
        pb::transaction::Operation::Rewrite(rewrite) => {
            write_kv(f, indent, "operation", "rewrite")?;
            write_pb_fragments(f, indent, "old_fragments", &rewrite.old_fragments)?;
            write_pb_fragments(f, indent, "new_fragments", &rewrite.new_fragments)?;
            write_kv(f, indent, "groups", &rewrite.groups.len().to_string())?;
            write_kv(
                f,
                indent,
                "rewritten_indices",
                &rewrite.rewritten_indices.len().to_string(),
            )
        }
        pb::transaction::Operation::Merge(merge) => {
            write_kv(f, indent, "operation", "merge")?;
            write_pb_fragments(f, indent, "fragments", &merge.fragments)?;
            write_kv(f, indent, "schema_fields", &merge.schema.len().to_string())?;
            write_kv(
                f,
                indent,
                "schema_metadata",
                &format_bytes_map_summary(&merge.schema_metadata),
            )
        }
        pb::transaction::Operation::Restore(restore) => {
            write_kv(f, indent, "operation", "restore")?;
            write_kv(f, indent, "version", &restore.version.to_string())
        }
        pb::transaction::Operation::ReserveFragments(reserve) => {
            write_kv(f, indent, "operation", "reserve_fragments")?;
            write_kv(
                f,
                indent,
                "num_fragments",
                &reserve.num_fragments.to_string(),
            )
        }
        pb::transaction::Operation::Update(update) => {
            write_kv(f, indent, "operation", "update")?;
            write_kv(
                f,
                indent,
                "removed_fragment_ids",
                &format!("{:?}", update.removed_fragment_ids),
            )?;
            write_pb_fragments(f, indent, "updated_fragments", &update.updated_fragments)?;
            write_pb_fragments(f, indent, "new_fragments", &update.new_fragments)?;
            write_kv(
                f,
                indent,
                "fields_modified",
                &format!("{:?}", update.fields_modified),
            )?;
            write_kv(
                f,
                indent,
                "compacted_sstables",
                &update.compacted_sstables.len().to_string(),
            )?;
            write_kv(
                f,
                indent,
                "update_mode",
                &format_update_mode(update.update_mode),
            )?;
            write_kv(
                f,
                indent,
                "fields_for_preserving_frag_bitmap",
                &format!("{:?}", update.fields_for_preserving_frag_bitmap),
            )?;
            write_kv(
                f,
                indent,
                "inserted_rows",
                &if update.inserted_rows.is_some() {
                    "present".to_string()
                } else {
                    "absent".to_string()
                },
            )?;
            write_kv(
                f,
                indent,
                "updated_fragment_offsets",
                &update.updated_fragment_offsets.len().to_string(),
            )
        }
        pb::transaction::Operation::Project(project) => {
            write_kv(f, indent, "operation", "project")?;
            write_kv(
                f,
                indent,
                "schema_fields",
                &project.schema.len().to_string(),
            )
        }
        pb::transaction::Operation::UpdateConfig(update_config) => {
            write_kv(f, indent, "operation", "update_config")?;
            write_kv(
                f,
                indent,
                "config_updates",
                &format_update_map(update_config.config_updates.as_ref()),
            )?;
            write_kv(
                f,
                indent,
                "table_metadata_updates",
                &format_update_map(update_config.table_metadata_updates.as_ref()),
            )?;
            write_kv(
                f,
                indent,
                "schema_metadata_updates",
                &format_update_map(update_config.schema_metadata_updates.as_ref()),
            )?;
            write_kv(
                f,
                indent,
                "field_metadata_updates",
                &update_config.field_metadata_updates.len().to_string(),
            )?;
            write_string_map(
                f,
                indent,
                "deprecated_upsert_values",
                &update_config.upsert_values,
            )?;
            write_kv(
                f,
                indent,
                "deprecated_delete_keys",
                &format!("{:?}", update_config.delete_keys),
            )?;
            write_string_map(
                f,
                indent,
                "deprecated_schema_metadata",
                &update_config.schema_metadata,
            )?;
            write_kv(
                f,
                indent,
                "deprecated_field_metadata",
                &update_config.field_metadata.len().to_string(),
            )
        }
        pb::transaction::Operation::DataReplacement(data_replacement) => {
            write_kv(f, indent, "operation", "data_replacement")?;
            write_kv(
                f,
                indent,
                "replacements",
                &data_replacement.replacements.len().to_string(),
            )
        }
        pb::transaction::Operation::UpdateMemWalState(update_mem_wal_state) => {
            write_kv(f, indent, "operation", "update_mem_wal_state")?;
            write_kv(
                f,
                indent,
                "compacted_sstables",
                &update_mem_wal_state.compacted_sstables.len().to_string(),
            )
        }
        pb::transaction::Operation::Clone(clone) => {
            write_kv(f, indent, "operation", "clone")?;
            write_kv(f, indent, "is_shallow", &clone.is_shallow.to_string())?;
            write_kv(
                f,
                indent,
                "ref_name",
                &format_optional(clone.ref_name.as_ref()),
            )?;
            write_kv(f, indent, "ref_version", &clone.ref_version.to_string())?;
            write_kv(f, indent, "ref_path", &clone.ref_path)?;
            write_kv(
                f,
                indent,
                "branch_name",
                &format_optional(clone.branch_name.as_ref()),
            )
        }
        pb::transaction::Operation::UpdateBases(update_bases) => {
            write_kv(f, indent, "operation", "update_bases")?;
            write_kv(
                f,
                indent,
                "new_bases",
                &update_bases.new_bases.len().to_string(),
            )
        }
        pb::transaction::Operation::DataOverlay(data_overlay) => {
            write_kv(f, indent, "operation", "data_overlay")?;
            write_kv(f, indent, "groups", &data_overlay.groups.len().to_string())?;
            let overlay_count = data_overlay
                .groups
                .iter()
                .map(|group| group.overlays.len())
                .sum::<usize>();
            write_kv(f, indent, "overlays", &overlay_count.to_string())
        }
    }
}

fn write_pb_fragments(
    f: &mut Formatter<'_>,
    indent: usize,
    title: &str,
    fragments: &[pb::DataFragment],
) -> fmt::Result {
    writeln!(
        f,
        "{:indent$}{}: {}",
        "",
        title,
        fragments.len(),
        indent = indent
    )?;
    for fragment in fragments {
        writeln!(
            f,
            "{:indent$}- id: {}",
            "",
            fragment.id,
            indent = indent + 2
        )?;
        write_kv(f, indent + 4, "files", &fragment.files.len().to_string())?;
        write_kv(
            f,
            indent + 4,
            "physical_rows",
            &if fragment.physical_rows == 0 {
                "unknown".to_string()
            } else {
                fragment.physical_rows.to_string()
            },
        )?;
        write_kv(
            f,
            indent + 4,
            "deletion_file",
            &if fragment.deletion_file.is_some() {
                "present".to_string()
            } else {
                "absent".to_string()
            },
        )?;
        write_kv(
            f,
            indent + 4,
            "overlays",
            &fragment.overlays.len().to_string(),
        )?;
    }
    Ok(())
}

fn write_string_map(
    f: &mut Formatter<'_>,
    indent: usize,
    title: &str,
    map: &HashMap<String, String>,
) -> fmt::Result {
    if map.is_empty() {
        return writeln!(f, "{:indent$}{}: {{}}", "", title, indent = indent);
    }
    writeln!(f, "{:indent$}{}:", "", title, indent = indent)?;
    for (key, value) in sorted_map(map) {
        write_kv(f, indent + 2, key, value)?;
    }
    Ok(())
}

fn sorted_map(map: &HashMap<String, String>) -> BTreeMap<&str, &str> {
    map.iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

fn write_indented_block(f: &mut Formatter<'_>, indent: usize, block: &str) -> fmt::Result {
    for line in block.lines() {
        writeln!(f, "{:indent$}{}", "", line, indent = indent)?;
    }
    Ok(())
}

fn write_kv(f: &mut Formatter<'_>, indent: usize, key: &str, value: &str) -> fmt::Result {
    writeln!(f, "{:indent$}{}: {}", "", key, value, indent = indent)
}

fn format_optional(value: Option<&String>) -> String {
    value.cloned().unwrap_or_else(|| "absent".to_string())
}

fn format_optional_path(value: Option<&Path>) -> String {
    value
        .map(|path| path.to_string())
        .unwrap_or_else(|| "absent".to_string())
}

fn format_option_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "absent".to_string())
}

fn format_option_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "absent".to_string())
}

fn format_unknown_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn format_absent_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "absent".to_string())
}

fn format_position(value: usize) -> String {
    if value == 0 {
        "absent".to_string()
    } else {
        value.to_string()
    }
}

fn format_cached_size(value: Option<std::num::NonZero<u64>>) -> String {
    value
        .map(|value| value.get().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn format_writer_version(value: Option<&WriterVersion>) -> String {
    let Some(value) = value else {
        return "absent".to_string();
    };
    let mut version = value.version.clone();
    if let Some(prerelease) = &value.prerelease {
        version.push('-');
        version.push_str(prerelease);
    }
    if let Some(build_metadata) = &value.build_metadata {
        version.push('+');
        version.push_str(build_metadata);
    }
    format!("{} {}", value.library, version)
}

fn format_timestamp(timestamp_nanos: u128, was_present: bool) -> String {
    if !was_present {
        return "absent".to_string();
    }
    let nanos = (timestamp_nanos % 1_000_000_000) as u32;
    let seconds = ((timestamp_nanos - u128::from(nanos)) / 1_000_000_000) as i64;
    DateTime::<Utc>::from_timestamp(seconds, nanos)
        .map(|timestamp| format!("{} ({} ns)", timestamp.to_rfc3339(), timestamp_nanos))
        .unwrap_or_else(|| format!("{timestamp_nanos} ns"))
}

fn format_data_storage_format(value: &DataStorageFormat, was_present: bool) -> String {
    let source = if was_present { "manifest" } else { "inferred" };
    format!("{} {} ({})", value.file_format, value.version, source)
}

fn format_feature_flags(flags: u64) -> String {
    if flags == 0 {
        return "0 (none)".to_string();
    }
    let known_flags = [
        (FLAG_DELETION_FILES, "deletion_files"),
        (FLAG_STABLE_ROW_IDS, "stable_row_ids"),
        (FLAG_USE_V2_FORMAT_DEPRECATED, "use_v2_format_deprecated"),
        (FLAG_TABLE_CONFIG, "table_config"),
        (FLAG_BASE_PATHS, "base_paths"),
        (FLAG_DISABLE_TRANSACTION_FILE, "disable_transaction_file"),
        (
            FLAG_UNSTABLE_DATA_OVERLAY_FILES,
            "unstable_data_overlay_files",
        ),
    ];
    let mut names = known_flags
        .iter()
        .filter_map(|(flag, name)| if flags & flag != 0 { Some(*name) } else { None })
        .collect::<Vec<_>>();
    // FLAG_UNKNOWN is the first bit after all known feature flags, so all lower
    // bits are the known-mask. Keep future feature flags below FLAG_UNKNOWN.
    let known_mask = FLAG_UNKNOWN - 1;
    let unknown = flags & !known_mask;
    if unknown != 0 {
        names.push("unknown");
    }
    format!("{} ({})", flags, names.join(", "))
}

fn format_update_mode(value: i32) -> String {
    match pb::transaction::UpdateMode::try_from(value) {
        Ok(mode) => format!("{:?}", mode),
        Err(_) => value.to_string(),
    }
}

fn format_row_id_meta(value: Option<&RowIdMeta>) -> String {
    match value {
        Some(RowIdMeta::Inline(data)) => format!("inline ({} bytes)", data.len()),
        Some(RowIdMeta::External(file)) => format_external_file(file),
        None => "absent".to_string(),
    }
}

fn format_row_dataset_version_meta(value: Option<&RowDatasetVersionMeta>) -> String {
    match value {
        Some(RowDatasetVersionMeta::Inline(data)) => format!("inline ({} bytes)", data.len()),
        Some(RowDatasetVersionMeta::External(file)) => format_external_file(file),
        None => "absent".to_string(),
    }
}

fn format_external_file(file: &ExternalFile) -> String {
    format!(
        "external path={} offset={} size={}",
        file.path, file.offset, file.size
    )
}

fn format_overlay_coverage(coverage: &OverlayCoverage) -> String {
    match coverage {
        OverlayCoverage::Shared(bitmap) => format!(
            "shared bitmap ({} offsets, {} serialized bytes)",
            bitmap.len(),
            bitmap.serialized_size()
        ),
        OverlayCoverage::PerField(bitmaps) => {
            let counts = bitmaps
                .iter()
                .map(|bitmap| bitmap.len().to_string())
                .collect::<Vec<_>>();
            format!("per-field bitmaps [{} offsets]", counts.join(", "))
        }
    }
}

fn format_bytes_map_summary(map: &HashMap<String, Vec<u8>>) -> String {
    if map.is_empty() {
        return "{}".to_string();
    }
    let mut parts = map
        .iter()
        .map(|(key, value)| format!("{}={} bytes", key, value.len()))
        .collect::<Vec<_>>();
    parts.sort();
    parts.join(", ")
}

fn format_update_map(update_map: Option<&pb::transaction::UpdateMap>) -> String {
    update_map
        .map(|update_map| {
            format!(
                "{} entries, replace={}",
                update_map.update_entries.len(),
                update_map.replace
            )
        })
        .unwrap_or_else(|| "absent".to_string())
}

pub(crate) async fn show_table_manifest(
    mut writer: impl Write,
    args: &LanceTableManifestArgs,
) -> Result<()> {
    let metadata = LanceToolManifest::open(args).await?;
    write!(writer, "{}", metadata)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_core::datatypes::Schema;
    use std::path::{Path as FsPath, PathBuf};

    #[test]
    fn test_infer_dataset_base_from_version_manifest_path() {
        let path = Path::from("dataset/_versions/1.manifest");
        assert_eq!(
            infer_dataset_base_from_manifest_path(&path).unwrap(),
            Path::from("dataset")
        );
    }

    #[test]
    fn test_infer_dataset_base_from_latest_manifest_path() {
        let path = Path::from("dataset/_latest.manifest");
        assert_eq!(
            infer_dataset_base_from_manifest_path(&path).unwrap(),
            Path::from("dataset")
        );
    }

    #[test]
    fn test_infer_dataset_base_from_unrecognized_manifest_paths() {
        let cases = [
            "1.manifest",
            "dataset/_versions/1.lance",
            "dataset/versions/1.manifest",
            "_latest.manifest",
        ];
        for path in cases {
            assert_eq!(
                infer_dataset_base_from_manifest_path(&Path::from(path)),
                None,
                "path {path} should not infer a dataset base"
            );
        }
    }

    #[test]
    fn test_render_optional_section_errors() {
        let manifest = LanceToolManifest {
            path: Path::from("dataset/_versions/1.manifest"),
            dataset_base: Some(Path::from("dataset")),
            has_timestamp: false,
            has_data_format: true,
            manifest: empty_manifest(),
            indices: OptionalSection::Error("index section is corrupt".to_string()),
            transaction: OptionalSection::Error("transaction section is corrupt".to_string()),
        };

        let rendered = manifest.to_string();

        assert!(rendered.contains("indices_error: index section is corrupt"));
        assert!(rendered.contains("transaction_error: transaction section is corrupt"));
    }

    #[tokio::test]
    async fn test_external_transaction_error_includes_path() {
        let object_store = ObjectStore::memory();
        let mut manifest = empty_manifest();
        manifest.transaction_file = Some("1.txn".to_string());
        let manifest_path = Path::from("dataset/_versions/1.manifest");

        let transaction = load_transaction(
            &object_store,
            &manifest_path,
            Some(&Path::from("dataset")),
            &manifest,
        )
        .await;

        let OptionalSection::Error(error) = transaction else {
            panic!("missing external transaction should render as an error");
        };
        assert!(error.contains("dataset/_transactions/1.txn"));
    }

    #[test]
    fn test_format_update_mode_names_known_values() {
        assert_eq!(
            format_update_mode(pb::transaction::UpdateMode::RewriteRows as i32),
            "RewriteRows"
        );
        assert_eq!(format_update_mode(99), "99");
    }

    #[tokio::test]
    async fn test_parse_all_test_data_manifests() {
        let test_data_dir = repo_root().join("test_data");
        let mut manifest_paths = Vec::new();
        collect_manifest_paths(&test_data_dir, &mut manifest_paths);
        manifest_paths.sort();

        assert!(
            !manifest_paths.is_empty(),
            "test_data should contain manifest compatibility samples"
        );

        for manifest_path in manifest_paths {
            let args = LanceTableManifestArgs {
                source: manifest_path.to_string_lossy().to_string(),
            };
            let manifest = LanceToolManifest::open(&args)
                .await
                .unwrap_or_else(|e| panic!("failed to parse {}: {}", manifest_path.display(), e));
            if let OptionalSection::Error(error) = &manifest.indices {
                panic!(
                    "failed to parse index section for {}: {}",
                    manifest_path.display(),
                    error
                );
            }
            if let OptionalSection::Error(error) = &manifest.transaction {
                panic!(
                    "failed to parse transaction for {}: {}",
                    manifest_path.display(),
                    error
                );
            }

            let rendered = manifest.to_string();
            let expected_version = format!("  version: {}", manifest.manifest.version);
            let expected_fragment_count =
                format!("  fragments: {}", manifest.manifest.fragments.len());
            assert!(
                rendered.contains("manifest:"),
                "rendered output missing manifest section for {}",
                manifest_path.display()
            );
            assert!(
                rendered.contains("schema:"),
                "rendered output missing schema section for {}",
                manifest_path.display()
            );
            assert!(
                rendered.contains(&expected_version),
                "rendered output missing version value for {}",
                manifest_path.display()
            );
            assert!(
                rendered.contains(&expected_fragment_count),
                "rendered output missing fragment count for {}",
                manifest_path.display()
            );
        }
    }

    fn empty_manifest() -> Manifest {
        Manifest::new(
            Schema::default(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        )
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("lance-tools should be two levels below the repository root")
            .to_path_buf()
    }

    fn collect_manifest_paths(dir: &FsPath, manifest_paths: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("failed to read directory {}: {}", dir.display(), e))
        {
            let entry = entry
                .unwrap_or_else(|e| panic!("failed to read entry in {}: {}", dir.display(), e));
            let path = entry.path();
            if path.is_dir() {
                collect_manifest_paths(&path, manifest_paths);
            } else if path
                .extension()
                .is_some_and(|extension| extension == "manifest")
            {
                manifest_paths.push(path);
            }
        }
    }
}
