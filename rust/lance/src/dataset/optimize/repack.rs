// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column repacking, the compaction task that rewrites some of one fragment's
//! columns into fewer data files without moving rows.
//!
//! Every `add_columns` backfill gives each fragment one more data file, and no
//! rewrite trigger counts files, so they pile up. A
//! [`CompactionTaskKind::RepackColumns`] task reads some of the fragment's
//! columns, writes them to new files, and commits the files as an
//! `Operation::DataReplacement` with `data_change: false`: the fragment id,
//! row addresses, overlays and index coverage are left as they are.
//!
//! [`CompactionTaskKind::RepackColumns`]: super::CompactionTaskKind::RepackColumns

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use lance_core::datatypes::{Field as LanceField, Schema};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{DataFile, Fragment};

use super::{CompactionMetrics, CompactionOptions, RepackedFiles, RewriteResult, TaskData};
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{DataReplacementGroup, Operation, TransactionBuilder};
use crate::dataset::{Dataset, cleanup_data_fragments, versions};
use crate::{Error, Result};

/// The ids of the leaf fields at and beneath `field`. The planner places a
/// column by where its leaves are: a V2.0 file can keep a struct's header
/// after the struct's last child in it was dropped.
fn leaf_ids(field: &LanceField, ids: &mut HashSet<i32>) {
    if field.children.is_empty() {
        ids.insert(field.id);
    }
    for child in &field.children {
        leaf_ids(child, ids);
    }
}

fn holds_blob(field: &LanceField) -> bool {
    field.is_blob() || field.children.iter().any(holds_blob)
}

/// What a commit of one new file holding `columns` (top-level field ids) leaves
/// of `fragment`, or `None` when the commit would refuse that file. The file
/// stands in for the one a repack writes: it has no path and takes the version
/// of the fragment's first file.
fn propose(schema: &Schema, fragment: &Fragment, columns: &BTreeSet<i32>) -> Option<Fragment> {
    let columns: Vec<i32> = columns.iter().copied().collect();
    let fields = schema.project_by_ids(&columns, true).field_ids();
    let column_indices = (0..fields.len() as i32).collect();
    let version = fragment.files.first()?.file_version().ok()?;
    let file = DataFile::new("", fields, column_indices, version, None, None);
    fragment.replace_files(schema, &[file]).ok()
}

/// The new files a repack of `fragment` should write, each the top-level
/// field ids it holds, or `None` when no repack is due. One is due when
/// `live_files`, the fragment's [`FragmentColumnLayoutStats::live_file_count`],
/// is above `max_files` and at least two files can be emptied together.
///
/// Each proposal merges whole files and is applied with
/// [`Fragment::replace_files`], as the commit will apply it; it takes only the
/// files the commit drops. Unless the limit is 1 or a file must stay anyway,
/// one file is kept when a proposal can leave one: a proposal leaving the
/// largest file in place wins, then the one leaving the fewest live files.
///
/// [`FragmentColumnLayoutStats::live_file_count`]: crate::dataset::compaction_stats::FragmentColumnLayoutStats::live_file_count
pub(super) fn plan_fragment_repack(
    schema: &Schema,
    fragment: &Fragment,
    live_files: usize,
    max_files: usize,
) -> Option<Vec<Vec<i32>>> {
    if live_files <= max_files {
        return None;
    }
    // A V1 file cannot tombstone single fields.
    if fragment.files.iter().any(|file| {
        !file
            .file_version()
            .is_ok_and(|version| version != ConcreteFileVersion::V1)
    }) {
        return None;
    }
    let schema_ids: HashSet<i32> = schema.fields_pre_order().map(|field| field.id).collect();
    let is_live = |file: &DataFile| file.fields.iter().any(|id| schema_ids.contains(id));
    let spilled = fragment.spilled_row_lineage_field_ids();
    let live: Vec<(&DataFile, HashSet<i32>)> = fragment
        .files
        .iter()
        .filter(|file| is_live(file))
        .map(|file| (file, file.covered_field_ids(schema)))
        .collect();
    let covered: HashSet<i32> = live.iter().flat_map(|(_, ids)| ids).copied().collect();
    let column_leaves: Vec<(i32, HashSet<i32>)> = schema
        .fields
        .iter()
        .map(|field| {
            let mut ids = HashSet::new();
            leaf_ids(field, &mut ids);
            (field.id, ids)
        })
        .collect();
    let movable: HashSet<i32> = schema
        .fields
        .iter()
        .zip(&column_leaves)
        .filter(|(field, (_, leaves))| {
            !leaves.is_disjoint(&covered) && !holds_blob(field) && leaves.is_subset(&covered)
        })
        .map(|(field, _)| field.id)
        .collect();
    // The top-level columns each live file holds data for.
    let file_columns: Vec<BTreeSet<i32>> = live
        .iter()
        .map(|(_, coverage)| {
            column_leaves
                .iter()
                .filter(|(_, leaves)| !leaves.is_disjoint(coverage))
                .map(|(column, _)| *column)
                .collect()
        })
        .collect();
    let holds_lineage = |file: &DataFile| file.fields.iter().any(|id| spilled.contains(id));
    let lineage_columns: BTreeSet<i32> = live
        .iter()
        .zip(&file_columns)
        .filter(|((file, _), _)| holds_lineage(file))
        .flat_map(|(_, columns)| columns.iter().copied())
        .collect();
    // The live files a merge may take: every column can move (no blob, all of
    // its leaves in the fragment's files), and moving them takes nothing out
    // of a file holding spilled lineage, which the new files do not carry.
    let candidates: Vec<usize> = (0..live.len())
        .filter(|index| {
            let columns = &file_columns[*index];
            !columns.is_empty()
                && !holds_lineage(live[*index].0)
                && columns.is_disjoint(&lineage_columns)
                && columns.iter().all(|column| movable.contains(column))
        })
        .collect();
    let columns_of = |files: &[usize]| -> BTreeSet<i32> {
        files
            .iter()
            .flat_map(|index| file_columns[*index].iter().copied())
            .collect()
    };
    // Shrinks `files` to those the commit of moving all their columns drops,
    // and returns them with the post-image; `None` once fewer than two are
    // left or the commit would refuse one new file holding all their columns.
    // A file can hold the struct header of a column it has no data for, so
    // taking one file out of the set can keep another alive.
    let merge = |mut files: Vec<usize>| loop {
        if files.len() < 2 {
            return None;
        }
        let after = propose(schema, fragment, &columns_of(&files))?;
        let dropped: Vec<usize> = files
            .iter()
            .copied()
            .filter(|index| {
                !after
                    .files
                    .iter()
                    .any(|file| file.path == live[*index].0.path)
            })
            .collect();
        if dropped.len() == files.len() {
            return Some((files, after));
        }
        files = dropped;
    };

    let (all, _) = merge(candidates)?;
    let holding = file_columns.iter().filter(|c| !c.is_empty()).count();
    let merged = if all.len() < holding || max_files == 1 {
        all
    } else {
        let mut by_size = all.clone();
        by_size.sort_by_key(|index| {
            std::cmp::Reverse((
                live[*index].0.file_size_bytes.get().map(|size| size.get()),
                file_columns[*index].len(),
            ))
        });
        let largest = by_size[0];
        // For each file kept: merge the files sharing no column with it,
        // which leaves its data in place, or every other file as long as the
        // kept file keeps a column.
        let proposals = by_size.iter().flat_map(|kept| {
            let untouched = all
                .iter()
                .copied()
                .filter(|index| file_columns[*index].is_disjoint(&file_columns[*kept]))
                .collect();
            let others = merge(all.iter().copied().filter(|i| i != kept).collect())
                .filter(|(files, _)| !file_columns[*kept].is_subset(&columns_of(files)));
            [merge(untouched), others].into_iter().flatten()
        });
        let mut best: Option<((bool, usize), Vec<usize>)> = None;
        for (files, after) in proposals {
            let live_after = after.files.iter().filter(|file| is_live(file)).count();
            let score = (files.contains(&largest), live_after);
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best = Some((score, files));
            }
            // Every proposal leaves at least the new file and the file kept.
            if score == (false, 2) {
                break;
            }
        }
        best.map_or(all, |(_, files)| files)
    };
    Some(vec![columns_of(&merged).into_iter().collect()])
}

/// Write one new file per entry of `files` (the top-level field ids it holds)
/// for repack task `task`, against `dataset`, which must be at the plan's
/// read version.
pub(super) async fn execute_repack(
    dataset: &Dataset,
    task: &TaskData,
    files: &[Vec<i32>],
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let [fragment] = task.fragments.as_slice() else {
        return Err(Error::invalid_input(format!(
            "a RepackColumns task names exactly one fragment, got {}",
            task.fragments.len()
        )));
    };
    if task.output_clustering_id.is_some() || !task.output_metadata.is_empty() {
        return Err(Error::invalid_input(
            "a RepackColumns task writes no new fragment, so it takes no output_clustering_id \
             or output_metadata",
        ));
    }
    let write_version = options.write_version(dataset);
    versions::validate_write_version(
        dataset.manifest.data_storage_format.lance_file_format(),
        write_version,
    )?;

    // Read the base values only: the overlays stay on the fragment and keep
    // shadowing them, so every live row reads as it did.
    let mut base = fragment.clone();
    base.overlays.clear();
    let source = FileFragment::new(Arc::new(dataset.clone()), base);
    let schema = dataset.schema();
    let batch_size = options.batch_size.map(|size| size as u32);

    let mut written: Vec<DataFile> = Vec::with_capacity(files.len());
    for column_ids in files {
        let names: Vec<&str> = schema
            .fields
            .iter()
            .filter(|field| column_ids.contains(&field.id))
            .map(|field| field.name.as_str())
            .collect();
        match write_one_file(&source, schema, &names, batch_size, write_version).await {
            Ok(file) => written.push(file),
            Err(err) => {
                discard(dataset, fragment.id, written).await;
                return Err(err);
            }
        }
    }

    let after = match fragment.replace_files(schema, &written) {
        Ok(after) => after,
        Err(err) => {
            discard(dataset, fragment.id, written).await;
            return Err(err);
        }
    };
    let metrics = CompactionMetrics {
        files_added: written.len(),
        files_removed: fragment.files.len() + written.len() - after.files.len(),
        ..Default::default()
    };
    Ok(RewriteResult {
        metrics,
        new_fragments: Vec::new(),
        read_version: dataset.manifest.version,
        original_fragments: Vec::new(),
        row_addrs: None,
        repacked_files: Some(RepackedFiles {
            fragment_id: fragment.id,
            files: written,
        }),
    })
}

/// Delete the files a repack task of fragment `fragment_id` wrote before it
/// failed.
async fn discard(dataset: &Dataset, fragment_id: u64, files: Vec<DataFile>) {
    let partial = Fragment {
        files,
        ..Fragment::new(fragment_id)
    };
    cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &[partial]).await;
}

/// Write `columns` of `source` to one new data file.
async fn write_one_file(
    source: &FileFragment,
    schema: &Schema,
    columns: &[&str],
    batch_size: Option<u32>,
    write_version: ConcreteFileVersion,
) -> Result<DataFile> {
    let write_schema = schema.project(columns)?;
    let mut updater = source
        .updater_with_version(
            Some(columns),
            Some((write_schema, schema.clone())),
            batch_size,
            None,
            write_version,
        )
        .await?;
    let finished: Result<Fragment> = async {
        while let Some(batch) = updater.next().await?.cloned() {
            updater.update(batch).await?;
        }
        updater.finish().await
    }
    .await;
    let mut fragment = match finished {
        Ok(fragment) => fragment,
        Err(err) => {
            updater.cleanup_unfinished_writer().await;
            return Err(err);
        }
    };
    // The updater appends the file it wrote, and writes none for no rows.
    fragment
        .files
        .split_off(source.metadata().files.len())
        .pop()
        .ok_or_else(|| Error::invalid_input(format!("fragment {} has no rows", fragment.id)))
}

/// Commit the files of repack results as one `DataReplacement` that moves
/// values without changing them. Each result applies to its fragment as it
/// stands at commit.
pub(super) async fn commit_repacked_files(
    dataset: &mut Dataset,
    results: Vec<RewriteResult>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    let mut metrics = CompactionMetrics::default();
    let mut read_version = u64::MAX;
    let mut replacements = Vec::new();
    for result in results {
        let Some(repacked) = result.repacked_files else {
            continue;
        };
        metrics += result.metrics;
        read_version = read_version.min(result.read_version);
        replacements.extend(
            repacked
                .files
                .into_iter()
                .map(|file| DataReplacementGroup(repacked.fragment_id, file)),
        );
    }
    if replacements.is_empty() {
        return Ok(CompactionMetrics::default());
    }
    let transaction = TransactionBuilder::new(
        // The earliest version a repack read, so the conflict check covers
        // every write since, as for rewrites.
        read_version,
        Operation::DataReplacement {
            replacements,
            data_change: false,
        },
    )
    .transaction_properties(options.transaction_properties.clone())
    .build();
    // Like a rewrite's, a repack result can be retried after an ambiguous
    // success, so its files are left for cleanup rather than deleted here.
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(metrics)
}
