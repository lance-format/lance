// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset metadata gathered before logical optimization.
//!
//! DataFusion optimizer rules are synchronous, but deciding whether an index
//! can serve part of a plan needs index metadata that is loaded asynchronously.
//! [`PlanningContext::collect`] walks a logical plan once, loads what the rules
//! will ask about, and the rules then hold an `Arc` to it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::new_empty_array;
use datafusion::catalog::default_table_source::source_as_provider;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_expr::{Expr, LogicalPlan, TableScan};
use lance_index::IndexCriteria;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::ScalarIndex;
use roaring::RoaringBitmap;

use crate::datafusion::LanceTableProvider;
use crate::dataset::Dataset;
use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::index::DatasetIndexExt;
use crate::index::scalar_logical::{load_named_scalar_segments, open_named_scalar_index};
use crate::{Error, Result};

/// Planning metadata for the Lance tables in one logical plan.
#[derive(Debug, Default)]
pub struct PlanningContext {
    tables: Vec<TableContext>,
}

/// Planning metadata for one Lance table scanned by the plan.
#[derive(Debug)]
pub struct TableContext {
    pub dataset: Arc<Dataset>,
    /// Join columns of this table that have an index answering
    /// [`ScalarIndex::lookup`], keyed by column name.
    lookup_indices: HashMap<String, Arc<LookupIndex>>,
}

/// An opened index that can answer exact-equality [`ScalarIndex::lookup`]s on
/// one column.
#[derive(Debug)]
pub struct LookupIndex {
    pub column: String,
    pub index_name: String,
    pub index: Arc<dyn ScalarIndex>,
    /// Fragments covered by at least one segment of the index.
    pub coverage: RoaringBitmap,
    /// Rows of covered fragments whose index entries a data overlay committed
    /// after the index was built may have made stale, as fragment id to row
    /// offsets.
    pub stale_rows: HashMap<u32, RoaringBitmap>,
}

impl PlanningContext {
    /// Load the metadata for every Lance table in `plan` that is equi-joined
    /// on plain columns.
    pub async fn collect(plan: &LogicalPlan) -> Result<Self> {
        let mut join_columns: Vec<(Arc<Dataset>, HashSet<String>)> = Vec::new();
        plan.apply(|node| {
            if let LogicalPlan::Join(join) = node {
                let sides = [
                    (
                        join.left.as_ref(),
                        join.on.iter().map(|(l, _)| l).collect::<Vec<_>>(),
                    ),
                    (
                        join.right.as_ref(),
                        join.on.iter().map(|(_, r)| r).collect(),
                    ),
                ];
                for (side, keys) in sides {
                    let Some(scan) = lance_table_scan(side) else {
                        continue;
                    };
                    let Some(dataset) = lance_dataset(scan) else {
                        continue;
                    };
                    let columns = match join_columns
                        .iter_mut()
                        .find(|(existing, _)| same_dataset(existing, &dataset))
                    {
                        Some((_, columns)) => columns,
                        None => {
                            join_columns.push((dataset, HashSet::new()));
                            &mut join_columns.last_mut().unwrap().1
                        }
                    };
                    columns.extend(keys.into_iter().filter_map(|key| match key {
                        Expr::Column(column) => Some(column.name.clone()),
                        _ => None,
                    }));
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        let mut tables = Vec::with_capacity(join_columns.len());
        for (dataset, columns) in join_columns {
            let mut lookup_indices = HashMap::new();
            for column in columns {
                if let Some(index) = load_lookup_index(&dataset, &column).await? {
                    lookup_indices.insert(column, Arc::new(index));
                }
            }
            tables.push(TableContext {
                dataset,
                lookup_indices,
            });
        }
        Ok(Self { tables })
    }

    /// The metadata for the table `dataset`, if the plan scans it.
    pub fn table(&self, dataset: &Dataset) -> Option<&TableContext> {
        self.tables
            .iter()
            .find(|table| same_dataset(&table.dataset, dataset))
    }
}

impl TableContext {
    pub fn lookup_index(&self, column: &str) -> Option<&Arc<LookupIndex>> {
        self.lookup_indices.get(column)
    }
}

/// The Lance table scan `plan` reads, looking through aliases.
pub fn lance_table_scan(plan: &LogicalPlan) -> Option<&TableScan> {
    match plan {
        LogicalPlan::SubqueryAlias(alias) => lance_table_scan(&alias.input),
        LogicalPlan::TableScan(scan) => lance_dataset(scan).is_some().then_some(scan),
        _ => None,
    }
}

pub fn lance_dataset(scan: &TableScan) -> Option<Arc<Dataset>> {
    let provider = source_as_provider(&scan.source).ok()?;
    provider
        .downcast_ref::<LanceTableProvider>()
        .map(LanceTableProvider::dataset)
}

fn same_dataset(a: &Dataset, b: &Dataset) -> bool {
    a.uri() == b.uri() && a.manifest.version == b.manifest.version
}

async fn load_lookup_index(dataset: &Arc<Dataset>, column: &str) -> Result<Option<LookupIndex>> {
    let Some(field) = dataset.schema().field(column) else {
        return Ok(None);
    };
    let Some(metadata) = dataset
        .load_scalar_index(
            IndexCriteria::default()
                .for_column(column)
                .supports_exact_equality(),
        )
        .await?
    else {
        return Ok(None);
    };
    let segments = load_named_scalar_segments(dataset, column, &metadata.name).await?;
    if segments.is_empty() {
        return Ok(None);
    }
    let index =
        open_named_scalar_index(dataset, column, &metadata.name, &NoOpMetricsCollector).await?;
    // Exact-equality indices do not all implement lookup, and the trait has no
    // capability flag, so ask with no keys: an unsupported index answers
    // `NotSupported` without doing any work.
    match index
        .lookup(
            new_empty_array(&field.data_type()).as_ref(),
            &NoOpMetricsCollector,
        )
        .await
    {
        Ok(_) => {}
        Err(Error::NotSupported { .. }) => return Ok(None),
        Err(err) => return Err(err),
    }

    let mut coverage = RoaringBitmap::new();
    for segment in &segments {
        match &segment.fragment_bitmap {
            Some(bitmap) => coverage |= bitmap,
            // Without coverage the indexed rows are unknown, so nothing can be
            // routed around the index.
            None => return Ok(None),
        }
    }

    let overlaid = overlaid_fragments(dataset.fragments());
    let mut stale_rows = HashMap::new();
    for segment in &segments {
        collect_overlay_stale_rows_for_segment(
            segment,
            &overlaid,
            &mut stale_rows,
            dataset.schema(),
        )?;
    }

    Ok(Some(LookupIndex {
        column: column.to_string(),
        index_name: metadata.name,
        index,
        coverage,
        stale_rows,
    }))
}
