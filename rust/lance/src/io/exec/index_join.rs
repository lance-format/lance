// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Equi-join a stream against a Lance table through scalar index lookups.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::compute::{filter_record_batch, take};
use arrow_array::{
    Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array, UInt64Array, cast::AsArray,
    types::UInt64Type,
};
use arrow_row::{RowConverter, SortField};
use arrow_schema::SchemaRef;
use datafusion::common::JoinType;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
    metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
    stream::RecordBatchStreamAdapter,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{StreamExt, TryStreamExt};
use lance_core::{ROW_ADDR, ROW_ID, Result, utils::address::RowAddress};
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::Fragment;
use lance_table::rowids::RowIdIndex;
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;

use super::utils::IndexMetrics;
use crate::Dataset;
use crate::datafusion::planning_context::LookupIndex;
use crate::dataset::rowids::{get_row_id_index, translate_addr_treemap_to_row_ids};
use crate::index::prefilter::DatasetPreFilter;

/// What [`IndexJoinExec`] joins against: a Lance table whose join columns all
/// have a lookup index.
#[derive(Debug)]
pub struct IndexJoinSpec {
    pub dataset: Arc<Dataset>,
    /// [`JoinType::Inner`] or [`JoinType::Right`], with the table on the left.
    pub join_type: JoinType,
    /// One index per join column, in join-key order.
    pub keys: Vec<Arc<LookupIndex>>,
    /// The table columns the join emits ahead of the input's columns.
    pub output_columns: Vec<TargetColumn>,
    /// Fragments every key's index covers. Index lookups are trusted only here.
    pub coverage: RoaringBitmap,
    /// Rows the lookups cannot see, which are matched through an in-memory map
    /// instead: whole fragments outside `coverage`...
    pub uncovered_fragments: Vec<Fragment>,
    /// ...and rows inside it whose index entries a data overlay may have made
    /// stale, as fragment id to row offsets.
    pub stale_rows: HashMap<u32, RoaringBitmap>,
}

/// A table column emitted by [`IndexJoinExec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetColumn {
    /// The join key at this position. A matched row's key equals the input's,
    /// so it is copied from the input rather than read.
    Key(usize),
    RowId,
    RowAddr,
}

impl IndexJoinSpec {
    fn has_uncovered_rows(&self) -> bool {
        !self.uncovered_fragments.is_empty() || !self.stale_rows.is_empty()
    }
}

/// Joins each input batch against a Lance table by looking its keys up in the
/// table's scalar indices, so no table data is read for rows the indices
/// cover.
///
/// Output is the table's [`TargetColumn`]s followed by the input's columns,
/// the same as a hash join with the table on the left. Unmatched input rows
/// are kept, with null table columns, for [`JoinType::Right`]. Null keys never
/// match.
///
/// Rows the indices do not cover (see [`IndexJoinSpec::uncovered_fragments`]
/// and [`IndexJoinSpec::stale_rows`]) have their keys read once, on first
/// execution, into an in-memory map. The planner only chooses this node when
/// that map is small.
#[derive(Debug)]
pub struct IndexJoinExec {
    input: Arc<dyn ExecutionPlan>,
    spec: Arc<IndexJoinSpec>,
    /// Column of each join key in `input`, in join-key order.
    input_keys: Vec<usize>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    /// Shared by every partition so the masks and map are built once.
    prepared: Arc<OnceCell<Arc<Prepared>>>,
    metrics: ExecutionPlanMetricsSet,
}

impl IndexJoinExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        spec: Arc<IndexJoinSpec>,
        input_keys: Vec<usize>,
        schema: SchemaRef,
    ) -> Result<Self> {
        if input_keys.len() != spec.keys.len() {
            return Err(lance_core::Error::internal(format!(
                "IndexJoinExec has {} key indices but {} input key columns",
                spec.keys.len(),
                input_keys.len()
            )));
        }
        let expected_columns = spec.output_columns.len() + input.schema().fields().len();
        if schema.fields().len() != expected_columns {
            return Err(lance_core::Error::internal(format!(
                "IndexJoinExec output schema has {} columns, expected {expected_columns}",
                schema.fields().len()
            )));
        }
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(
                input.properties().output_partitioning().partition_count(),
            ),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            spec,
            input_keys,
            schema,
            properties,
            prepared: Arc::new(OnceCell::new()),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for IndexJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                let indices = self
                    .spec
                    .keys
                    .iter()
                    .map(|key| format!("{}({})", key.index_name, key.column))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "IndexJoin: type={}, indices=[{indices}]",
                    self.spec.join_type
                )?;
                if !self.spec.uncovered_fragments.is_empty() {
                    write!(
                        f,
                        ", unindexed_fragments={}",
                        self.spec.uncovered_fragments.len()
                    )?;
                }
                if !self.spec.stale_rows.is_empty() {
                    let stale_rows: u64 = self.spec.stale_rows.values().map(|r| r.len()).sum();
                    write!(f, ", overlay_stale_rows={stale_rows}")?;
                }
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for IndexJoinExec {
    fn name(&self) -> &str {
        "IndexJoinExec"
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "IndexJoinExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::try_new(
            children.remove(0),
            self.spec.clone(),
            self.input_keys.clone(),
            self.schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let joiner = BatchJoiner {
            spec: self.spec.clone(),
            input_keys: self.input_keys.clone(),
            schema: self.schema.clone(),
            index_metrics: IndexMetrics::new(&self.metrics, partition),
        };
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let prepared = self.prepared.clone();
        let spec = self.spec.clone();
        let stream = futures::stream::once(async move {
            let prepared = prepared
                .get_or_try_init(|| Prepared::load(spec))
                .await?
                .clone();
            let joiner = Arc::new(joiner);
            Ok::<_, DataFusionError>(input.and_then(move |batch| {
                let joiner = joiner.clone();
                let prepared = prepared.clone();
                async move { Ok(joiner.join(batch, &prepared).await?) }
            }))
        })
        .try_flatten()
        .map(move |batch| {
            let _timer = baseline.elapsed_compute().timer();
            if let Ok(batch) = &batch {
                baseline.record_output(batch.num_rows());
            }
            batch
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

/// Per-execution state built once and shared by every partition.
#[derive(Debug)]
struct Prepared {
    /// Rows a lookup may report that must not match, in the row-id domain the
    /// indices report: deleted rows, rows outside [`IndexJoinSpec::coverage`]
    /// and overlay-stale rows. `None` when nothing is blocked.
    lookup_mask: Option<Arc<RowAddrMask>>,
    /// Translates row ids to addresses under stable row ids.
    row_id_index: Option<Arc<RowIdIndex>>,
    uncovered: Option<UncoveredRows>,
}

/// The keys of the rows the indices cannot see, for probing in memory.
#[derive(Debug)]
struct UncoveredRows {
    converter: RowConverter,
    /// Encoded key to the `(row id, row address)` of every row with it.
    rows: HashMap<Box<[u8]>, Vec<(u64, u64)>>,
}

impl Prepared {
    async fn load(spec: Arc<IndexJoinSpec>) -> DFResult<Arc<Self>> {
        let dataset = spec.dataset.clone();
        let deletion_mask = match DatasetPreFilter::create_restricted_deletion_mask(
            dataset.clone(),
            spec.coverage.clone(),
        ) {
            Some(mask) => Some(mask.await?),
            None => None,
        };
        let stale_block = if spec.stale_rows.is_empty() {
            None
        } else {
            let mut stale = RowAddrTreeMap::new();
            for (&fragment_id, offsets) in &spec.stale_rows {
                stale.insert_bitmap(fragment_id, offsets.clone());
            }
            if dataset.manifest.uses_stable_row_ids() {
                stale = translate_addr_treemap_to_row_ids(&dataset, &stale).await?;
            }
            Some(RowAddrMask::from_block(stale))
        };
        let lookup_mask = match (deletion_mask, stale_block) {
            (None, None) => None,
            (Some(mask), None) => Some(mask),
            (None, Some(block)) => Some(Arc::new(block)),
            (Some(mask), Some(block)) => Some(Arc::new(mask.as_ref().clone() & block)),
        };

        let row_id_index = get_row_id_index(&dataset).await?;
        let uncovered = if spec.has_uncovered_rows() {
            Some(UncoveredRows::load(&spec).await?)
        } else {
            None
        };
        Ok(Arc::new(Self {
            lookup_mask,
            row_id_index,
            uncovered,
        }))
    }
}

impl UncoveredRows {
    async fn load(spec: &IndexJoinSpec) -> Result<Self> {
        let dataset = &spec.dataset;
        let columns = spec
            .keys
            .iter()
            .map(|key| key.column.clone())
            .collect::<Vec<_>>();
        let fields = columns
            .iter()
            .map(|column| {
                let field = dataset.schema().field(column).ok_or_else(|| {
                    lance_core::Error::internal(format!(
                        "index join column '{column}' is missing from the dataset schema"
                    ))
                })?;
                Ok(SortField::new(field.data_type()))
            })
            .collect::<Result<Vec<_>>>()?;
        let converter = RowConverter::new(fields)?;

        let uncovered_ids = spec
            .uncovered_fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect::<RoaringBitmap>();
        // A stale row is read by scanning its fragment's keys; overlays are
        // rare, and the planner bounded what is kept, not what is scanned.
        let mut fragments = spec.uncovered_fragments.clone();
        fragments.extend(
            dataset
                .fragments()
                .iter()
                .filter(|fragment| spec.stale_rows.contains_key(&(fragment.id as u32)))
                .cloned(),
        );

        let mut scanner = dataset.scan();
        scanner
            .with_fragments(fragments)
            .project(&columns)?
            .with_row_id()
            .with_row_address();
        let mut stream = scanner.try_into_stream().await?;
        let mut rows: HashMap<Box<[u8]>, Vec<(u64, u64)>> = HashMap::new();
        while let Some(batch) = stream.try_next().await? {
            let addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
            let keep = addrs
                .values()
                .iter()
                .map(|&addr| {
                    let addr = RowAddress::from(addr);
                    uncovered_ids.contains(addr.fragment_id())
                        || spec
                            .stale_rows
                            .get(&addr.fragment_id())
                            .is_some_and(|offsets| offsets.contains(addr.row_offset()))
                })
                .collect::<BooleanArray>();
            let batch = filter_record_batch(&batch, &keep)?;

            let keys = columns
                .iter()
                .map(|column| batch[column.as_str()].clone())
                .collect::<Vec<_>>();
            let encoded = converter.convert_columns(&keys)?;
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            let row_addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
            for row in 0..batch.num_rows() {
                if keys.iter().any(|key| key.is_null(row)) {
                    continue;
                }
                rows.entry(encoded.row(row).as_ref().into())
                    .or_default()
                    .push((row_ids.value(row), row_addrs.value(row)));
            }
        }
        Ok(Self { converter, rows })
    }
}

struct BatchJoiner {
    spec: Arc<IndexJoinSpec>,
    input_keys: Vec<usize>,
    schema: SchemaRef,
    index_metrics: IndexMetrics,
}

impl BatchJoiner {
    async fn join(&self, batch: RecordBatch, prepared: &Prepared) -> Result<RecordBatch> {
        let keys = self
            .input_keys
            .iter()
            .map(|&column| batch.column(column).clone())
            .collect::<Vec<_>>();

        // (input row, target row id, target row address)
        let mut matches = self.lookup(&keys, prepared).await?;
        if let Some(uncovered) = &prepared.uncovered {
            let encoded = uncovered.converter.convert_columns(&keys)?;
            for row in 0..batch.num_rows() {
                if keys.iter().any(|key| key.is_null(row)) {
                    continue;
                }
                if let Some(targets) = uncovered.rows.get(encoded.row(row).as_ref()) {
                    matches.extend(
                        targets
                            .iter()
                            .map(|&(row_id, row_addr)| (row as u32, row_id, row_addr)),
                    );
                }
            }
        }
        // Keep the input's row order, and make the order of several matches
        // for one input row deterministic.
        matches.sort_unstable();

        self.build_output(&batch, &keys, &matches)
    }

    /// The `(input row, row id, row address)` triples the indices report,
    /// with blocked rows removed.
    async fn lookup(&self, keys: &[ArrayRef], prepared: &Prepared) -> Result<Vec<(u32, u64, u64)>> {
        let mut pairs: Option<Vec<(u32, u64)>> = None;
        for (key, values) in self.spec.keys.iter().zip(keys) {
            let found = key
                .index
                .lookup(values.as_ref(), &self.index_metrics)
                .await?;
            let found = found
                .key_indices
                .values()
                .iter()
                .copied()
                .zip(found.row_ids.values().iter().copied());
            pairs = Some(match pairs {
                None => found
                    .filter(|(_, row_id)| {
                        prepared
                            .lookup_mask
                            .as_ref()
                            .is_none_or(|mask| mask.selected(*row_id))
                    })
                    .collect(),
                // A composite key matches a row only if every column does.
                Some(pairs) => {
                    let found = found.collect::<HashSet<_>>();
                    pairs
                        .into_iter()
                        .filter(|pair| found.contains(pair))
                        .collect()
                }
            });
            if pairs.as_ref().is_some_and(Vec::is_empty) {
                break;
            }
        }
        let pairs = pairs.unwrap_or_default();

        Ok(match &prepared.row_id_index {
            None => pairs
                .into_iter()
                .map(|(row, row_id)| (row, row_id, row_id))
                .collect(),
            Some(row_id_index) => {
                let row_ids = pairs.iter().map(|(_, row_id)| *row_id).collect::<Vec<_>>();
                let addrs = row_id_index.get_many(&row_ids)?;
                pairs
                    .into_iter()
                    .zip(addrs)
                    .filter_map(|((row, row_id), addr)| {
                        addr.map(|addr| (row, row_id, u64::from(addr)))
                    })
                    .collect()
            }
        })
    }

    fn build_output(
        &self,
        batch: &RecordBatch,
        keys: &[ArrayRef],
        matches: &[(u32, u64, u64)],
    ) -> Result<RecordBatch> {
        let keep_unmatched = self.spec.join_type == JoinType::Right;
        let capacity = if keep_unmatched {
            matches.len() + batch.num_rows()
        } else {
            matches.len()
        };
        let mut input_rows = Vec::with_capacity(capacity);
        let mut matched = Vec::with_capacity(capacity);
        let mut next = matches.iter().peekable();
        for row in 0..batch.num_rows() as u32 {
            let mut found = false;
            while let Some(&&(match_row, row_id, row_addr)) = next.peek()
                && match_row == row
            {
                input_rows.push(row);
                matched.push(Some((row_id, row_addr)));
                found = true;
                next.next();
            }
            if !found && keep_unmatched {
                input_rows.push(row);
                matched.push(None);
            }
        }

        let input_rows = UInt32Array::from(input_rows);
        let target_rows = input_rows
            .iter()
            .zip(&matched)
            .map(|(row, target)| target.as_ref().and(row))
            .collect::<UInt32Array>();
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        for column in &self.spec.output_columns {
            columns.push(match column {
                TargetColumn::Key(key) => take(keys[*key].as_ref(), &target_rows, None)?,
                TargetColumn::RowId => Arc::new(
                    matched
                        .iter()
                        .map(|target| target.map(|(row_id, _)| row_id))
                        .collect::<UInt64Array>(),
                ) as ArrayRef,
                TargetColumn::RowAddr => Arc::new(
                    matched
                        .iter()
                        .map(|target| target.map(|(_, row_addr)| row_addr))
                        .collect::<UInt64Array>(),
                ) as ArrayRef,
            });
        }
        for column in batch.columns() {
            columns.push(take(column.as_ref(), &input_rows, None)?);
        }
        Ok(RecordBatch::try_new(self.schema.clone(), columns)?)
    }
}
