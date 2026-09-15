// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Named ways to stage a real [`Transaction`] against a dataset version.
//!
//! Every scenario goes through a production write path wherever one exists
//! (`InsertBuilder`, `DeleteBuilder`, `MergeInsertBuilder`, `CreateIndexBuilder`,
//! `FileFragment::write_overlay`), so the transaction under test carries the
//! fragment contents, deletion files and affected-row sets a real writer would
//! produce. Operations with no uncommitted builder are assembled by hand from
//! state read out of the fixture, never from placeholder fragments.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::ROW_ADDR;
use lance_core::utils::address::RowAddress;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_select::RowAddrTreeMap;
use lance_table::format::{DataFile, Fragment};

use crate::Result;
use crate::dataset::transaction::{
    DataReplacementGroup, Operation, RewriteGroup, Transaction, UpdateMode,
};
use crate::dataset::{
    CommitBuilder, Dataset, DeleteBuilder, InsertBuilder, MergeInsertBuilder, NewColumnTransform,
    WriteMode, WriteParams,
};
use crate::index::CreateIndexBuilder;

/// A transaction staged against a dataset version but not yet committed.
pub(super) struct Staged {
    pub transaction: Transaction,
    /// Carried separately because `CommitBuilder` takes it separately; the
    /// second phase of delete/update conflict detection needs it.
    pub affected_rows: Option<RowAddrTreeMap>,
}

impl Staged {
    fn new(transaction: Transaction) -> Self {
        Self {
            transaction,
            affected_rows: None,
        }
    }

    fn with_affected_rows(transaction: Transaction, affected_rows: Option<RowAddrTreeMap>) -> Self {
        Self {
            transaction,
            affected_rows,
        }
    }

    /// Commit against `base`, which pins the read version. `base` being behind
    /// the latest version is what forces the rebase under test.
    pub async fn commit(self, base: &Arc<Dataset>) -> Result<Dataset> {
        let mut builder = CommitBuilder::new(base.clone()).with_max_retries(1);
        if let Some(rows) = self.affected_rows {
            builder = builder.with_affected_rows(rows);
        }
        builder.execute(self.transaction).await
    }
}

/// The operations this suite stages, one variant per column and row of the
/// matrix. `Update` appears twice because its two modes take different arms of
/// `check_update_txn` and build different manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Scenario {
    Append,
    Delete,
    UpdateRewriteRows,
    UpdateRewriteColumns,
    DataOverlay,
    DataReplacement,
    Project,
    Merge,
    CreateIndex,
    Rewrite,
    Overwrite,
    Restore,
}

impl Scenario {
    pub const ALL: [Self; 12] = [
        Self::Append,
        Self::Delete,
        Self::UpdateRewriteRows,
        Self::UpdateRewriteColumns,
        Self::DataOverlay,
        Self::DataReplacement,
        Self::Project,
        Self::Merge,
        Self::CreateIndex,
        Self::Rewrite,
        Self::Overwrite,
        Self::Restore,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Delete => "delete",
            Self::UpdateRewriteRows => "update_rewrite_rows",
            Self::UpdateRewriteColumns => "update_rewrite_columns",
            Self::DataOverlay => "data_overlay",
            Self::DataReplacement => "data_replacement",
            Self::Project => "project",
            Self::Merge => "merge",
            Self::CreateIndex => "create_index",
            Self::Rewrite => "rewrite",
            Self::Overwrite => "overwrite",
            Self::Restore => "restore",
        }
    }

    /// Whether this operation replaces whole-dataset state rather than editing
    /// it incrementally. The fragment-level invariants do not apply across such
    /// a commit: `Overwrite` mints a fresh fragment set and `Restore` reinstates
    /// an older manifest, so row counts and overlay counts legitimately drop.
    pub fn replaces_state(self) -> bool {
        matches!(self, Self::Overwrite | Self::Restore)
    }

    pub async fn stage(self, dataset: &Arc<Dataset>) -> Result<Staged> {
        match self {
            Self::Append => stage_append(dataset).await,
            Self::Delete => stage_delete(dataset).await,
            Self::UpdateRewriteRows => stage_update_rewrite_rows(dataset).await,
            Self::UpdateRewriteColumns => stage_update_rewrite_columns(dataset).await,
            Self::DataOverlay => stage_data_overlay(dataset).await,
            Self::DataReplacement => stage_data_replacement(dataset).await,
            Self::Project => stage_project(dataset).await,
            Self::Merge => stage_merge(dataset).await,
            Self::CreateIndex => stage_create_index(dataset).await,
            Self::Rewrite => stage_rewrite(dataset).await,
            Self::Overwrite => stage_overwrite(dataset).await,
            Self::Restore => stage_restore(dataset).await,
        }
    }
}

/// Two rows of new data, used by the scenarios that add rows. Carries the full
/// fixture schema, which an append requires.
fn new_rows(start: i32) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![start, start + 1])),
            Arc::new(Int32Array::from(vec![start * 10, start * 10 + 1])),
            Arc::new(Int32Array::from(vec![start + 100, start + 101])),
        ],
    )
    .unwrap()
}

async fn stage_append(dataset: &Arc<Dataset>) -> Result<Staged> {
    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&params)
        .execute_uncommitted(vec![new_rows(100)])
        .await?;
    Ok(Staged::new(transaction))
}

async fn stage_overwrite(dataset: &Arc<Dataset>) -> Result<Staged> {
    let params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&params)
        .execute_uncommitted(vec![new_rows(200)])
        .await?;
    Ok(Staged::new(transaction))
}

async fn stage_delete(dataset: &Arc<Dataset>) -> Result<Staged> {
    let staged = DeleteBuilder::new(dataset.clone(), "a = 0")
        .execute_uncommitted()
        .await?;
    Ok(Staged::with_affected_rows(
        staged.transaction,
        staged.affected_rows,
    ))
}

/// A merge insert that matches the whole schema, which is the production
/// producer of `UpdateMode::RewriteRows`: matched rows are deleted in place and
/// rewritten into a new fragment.
async fn stage_update_rewrite_rows(dataset: &Arc<Dataset>) -> Result<Staged> {
    let source = new_rows(1);
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(source.clone())], source.schema());
    let staged = MergeInsertBuilder::try_new(dataset.clone(), vec!["a".into()])?
        .when_matched(crate::dataset::WhenMatched::UpdateAll)
        .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
        .try_build()?
        .execute_uncommitted(reader)
        .await?;
    Ok(Staged::with_affected_rows(
        staged.transaction,
        staged.affected_rows,
    ))
}

/// `RewriteColumns` has no uncommitted builder, so it is assembled from the
/// fixture's own fragments: field `c`'s data is restated for fragment 0 while
/// every other field keeps its existing file.
///
/// `c` rather than `b` because `c` is the only field with a data file to
/// itself. Swapping the file that `a` and `b` share for one holding just `b`
/// would leave `a` with no file at all, and the fragment would read `a` back as
/// null.
async fn stage_update_rewrite_columns(dataset: &Arc<Dataset>) -> Result<Staged> {
    let mut fragment = dataset.fragments()[0].clone();
    let field_c = dataset.schema().field("c").unwrap().id;
    let replacement = write_value_file(dataset, "update_columns", &["c"], &[7, 7, 7]).await?;
    // Replace the file carrying `c` rather than appending one, so the fragment
    // keeps exactly one file per field.
    let target = fragment
        .files
        .iter_mut()
        .find(|file| file.fields.contains(&field_c))
        .expect("the fixture gives `c` a data file of its own");
    *target = replacement;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![fragment],
            new_fragments: vec![],
            fields_modified: vec![field_c as u32],
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
    )))
}

/// The overlay this suite installs everywhere: fragment 0, field `b`, the last
/// row of the fragment. That row survives the `delete` scenario, so an overlay
/// that disappears did so because it was dropped, not because its row went away.
pub(super) const OVERLAY_VALUE: i32 = 42;

async fn stage_data_overlay(dataset: &Arc<Dataset>) -> Result<Staged> {
    let fragment = dataset.get_fragment(0).unwrap();
    let last_offset = fragment.physical_rows().await? as u32 - 1;
    let overlay_schema = dataset.schema().project(&["b"])?;
    let mut writer = fragment.write_overlay(&overlay_schema).await?;

    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ADDR, DataType::UInt64, false),
            ArrowField::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(UInt64Array::from(vec![u64::from(
                RowAddress::new_from_parts(0, last_offset),
            )])),
            Arc::new(Int32Array::from(vec![OVERLAY_VALUE])),
        ],
    )
    .unwrap();
    writer.write_batch(&batch).await?;
    let group = writer
        .finish()
        .await?
        .expect("overlay covers one cell, so a group is produced");

    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::DataOverlay {
            groups: vec![group],
        },
    )))
}

async fn stage_data_replacement(dataset: &Arc<Dataset>) -> Result<Staged> {
    // `DataReplacement` requires the new file to carry exactly the fields of the
    // file it replaces, so mirror fragment 0's first file.
    let existing = &dataset.fragments()[0].files[0];
    let names = existing
        .fields
        .iter()
        .map(|id| {
            dataset
                .schema()
                .field_by_id(*id)
                .expect("fragment file references a field in the schema")
                .name
                .clone()
        })
        .collect::<Vec<_>>();
    let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
    let replacement = write_value_file(dataset, "replacement", &name_refs, &[5, 5, 5]).await?;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, replacement)],
        },
    )))
}

async fn stage_project(dataset: &Arc<Dataset>) -> Result<Staged> {
    // Drop the column added by the fixture, which is the only one whose data
    // file can be pruned whole.
    let schema = dataset.schema().project(&["a", "b"])?;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Project {
            schema,
            preserves_nullability: true,
        },
    )))
}

async fn stage_merge(dataset: &Arc<Dataset>) -> Result<Staged> {
    // A merge restates every fragment against a schema; here it drops `c`'s
    // data the way a merge that omits a column does.
    let schema = dataset.schema().project(&["a", "b"])?;
    let field_c = dataset.schema().field("c").unwrap().id;
    let fragments = dataset
        .fragments()
        .iter()
        .map(|fragment| {
            let mut fragment = fragment.clone();
            fragment
                .files
                .retain(|file| !file.fields.contains(&field_c));
            fragment
        })
        .collect::<Vec<_>>();
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Merge {
            fragments,
            schema,
            preserves_nullability: true,
        },
    )))
}

async fn stage_create_index(dataset: &Arc<Dataset>) -> Result<Staged> {
    let mut owned = dataset.as_ref().clone();
    let params = ScalarIndexParams::new("btree".to_string());
    let index = CreateIndexBuilder::new(&mut owned, &["a"], IndexType::Scalar, &params)
        .name("a_idx".to_string())
        .execute_uncommitted()
        .await?;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: vec![],
        },
    )))
}

/// Compaction of fragment 0 into a freshly written fragment holding the same
/// rows. Written through `InsertBuilder` so the new fragment has real files.
async fn stage_rewrite(dataset: &Arc<Dataset>) -> Result<Staged> {
    let old = dataset.fragments()[0].clone();
    let rows = dataset
        .scan()
        .with_fragments(vec![old.clone()])
        .try_into_batch()
        .await?;
    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let written = InsertBuilder::new(dataset.clone())
        .with_params(&params)
        .execute_uncommitted(vec![rows])
        .await?;
    let Operation::Append { fragments } = written.operation else {
        unreachable!("an append in uncommitted form always carries Append");
    };
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![old],
                new_fragments: fragments,
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
    )))
}

async fn stage_restore(dataset: &Arc<Dataset>) -> Result<Staged> {
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        // The fixture's first version, before the column `c` was added.
        Operation::Restore { version: 1 },
    )))
}

/// Write a standalone data file holding `values` for `fields`, laid out so it
/// can stand in for an existing file of the same fields.
async fn write_value_file(
    dataset: &Arc<Dataset>,
    name: &str,
    fields: &[&str],
    values: &[i32],
) -> Result<DataFile> {
    let schema = dataset.schema().project(fields)?;
    let filename = format!("{name}.lance");
    let path = dataset.base.clone().join("data").join(filename.as_str());
    let writer = dataset.object_store.create(&path).await?;
    let file_version = dataset.manifest.data_storage_format.lance_file_format();
    let mut writer = lance_file::versions::create_writer(
        file_version,
        writer,
        schema,
        lance_file::writer::FileWriterOptions::default(),
    )?;
    for column in 0..fields.len() {
        writer
            .write_column(column, Arc::new(Int32Array::from(values.to_vec())))
            .await?;
    }
    let summary = writer.finish().await?;

    let mut data_file = DataFile::new_unstarted(filename, file_version);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(field_id, _)| *field_id as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, column_index)| *column_index as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = lance_io::utils::CachedFileSize::new(summary.size_bytes);
    Ok(data_file)
}

/// The dataset every case starts from.
///
/// Deliberately tiny — three rows per fragment, two fragments, three fields —
/// so a case costs a few milliseconds. The shape is still rich enough for the
/// invariants to bite: two fragments so a fragment-scoped conflict can be
/// distinguished from a dataset-wide one, and a third column `c` living in its
/// own data file so `project` and `merge` have a file to prune whole.
///
/// Version 1 holds `a`/`b`; version 2 adds `c`. Cases stage against version 2,
/// which gives `restore` a version to fall back to.
pub(super) async fn fixture() -> Arc<Dataset> {
    let data = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
            Arc::new(Int32Array::from(vec![0, 0, 0, 0, 0, 0])),
        ],
    )
    .unwrap();
    let params = WriteParams {
        max_rows_per_file: 3,
        ..Default::default()
    };
    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&params)
        .execute(vec![data])
        .await
        .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("c".into(), "a + 100".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    Arc::new(dataset)
}

/// Fragment ids and their logical row counts, for the row-count invariant.
pub(super) fn row_counts(dataset: &Dataset) -> HashMap<u64, u64> {
    dataset
        .fragments()
        .iter()
        .map(|fragment: &Fragment| {
            let deleted = fragment
                .deletion_file
                .as_ref()
                .and_then(|f| f.num_deleted_rows)
                .unwrap_or(0) as u64;
            (
                fragment.id,
                fragment.physical_rows.unwrap_or(0) as u64 - deleted,
            )
        })
        .collect()
}
