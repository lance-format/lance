// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Equi-join a stream against a Lance table through scalar index lookups.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::Arc;

use arrow::compute::{filter_record_batch, take};
use arrow_array::{
    Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array, UInt64Array,
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
};
use arrow_row::{Row, RowConverter, Rows, SortField};
use arrow_schema::SchemaRef;
use datafusion::common::JoinType;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
    metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
    stream::{RecordBatchReceiverStream, RecordBatchStreamAdapter},
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{StreamExt, TryStreamExt};
use lance_core::{ROW_ADDR, ROW_ID, Result, utils::address::RowAddress};
use lance_select::{RowAddrMask, RowAddrTreeMap, RowSetOps};
use lance_table::format::Fragment;
use lance_table::rowids::RowIdIndex;
use roaring::RoaringBitmap;
use rustc_hash::FxBuildHasher;
use tokio::sync::OnceCell;
use tokio::sync::mpsc::Sender;

use super::utils::IndexMetrics;
use crate::Dataset;
use crate::datafusion::planning_context::LookupIndex;
use crate::dataset::rowids::{
    get_row_id_index, row_addrs_to_row_ids, translate_addr_treemap_to_row_ids,
};
use crate::index::prefilter::DatasetPreFilter;

/// What [`IndexJoinExec`] joins against: a Lance table whose join column has
/// a lookup index.
#[derive(Debug)]
pub struct IndexJoinSpec {
    pub dataset: Arc<Dataset>,
    /// [`JoinType::Inner`] or [`JoinType::Right`], with the table on the left.
    pub join_type: JoinType,
    /// The index on the table's join column.
    pub key: Arc<LookupIndex>,
    /// The table columns the join emits ahead of the input's columns.
    pub output_columns: Vec<TargetColumn>,
    /// Fragments the index covers. Index lookups are trusted only here.
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
    /// The join key. A matched row's key equals the input's, so it is copied
    /// from the input rather than read.
    Key,
    RowId,
    RowAddr,
}

impl IndexJoinSpec {
    fn has_uncovered_rows(&self) -> bool {
        !self.uncovered_fragments.is_empty() || !self.stale_rows.is_empty()
    }
}

/// Joins each input batch against a Lance table by looking its keys up in the
/// table's scalar index, so no table data is read for rows the index covers.
///
/// Output is the table's [`TargetColumn`]s followed by the input's columns,
/// the same as a hash join with the table on the left. Unmatched input rows
/// are kept, with null table columns, for [`JoinType::Right`]. Null keys never
/// match.
///
/// Rows the index does not cover (see [`IndexJoinSpec::uncovered_fragments`]
/// and [`IndexJoinSpec::stale_rows`]) have their keys read once, on first
/// execution, into an in-memory map. The planner only chooses this node when
/// that map is small.
#[derive(Debug)]
pub struct IndexJoinExec {
    input: Arc<dyn ExecutionPlan>,
    spec: Arc<IndexJoinSpec>,
    /// Column of the join key in `input`.
    input_key: usize,
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
        input_key: usize,
        schema: SchemaRef,
    ) -> Result<Self> {
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
            input_key,
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
                write!(
                    f,
                    "IndexJoin: type={}, index={}({})",
                    self.spec.join_type, self.spec.key.index_name, self.spec.key.column
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
            self.input_key,
            self.schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let batch_size = context.session_config().batch_size();
        let mut input = self.input.execute(partition, context)?;
        let joiner = BatchJoiner {
            spec: self.spec.clone(),
            input_key: self.input_key,
            schema: self.schema.clone(),
            batch_size,
            index_metrics: IndexMetrics::new(&self.metrics, partition),
        };
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let prepared = self.prepared.clone();
        let spec = self.spec.clone();
        // The joiner runs in its own task because each lookup stream borrows
        // the index and metrics the task owns.
        let mut builder = RecordBatchReceiverStream::builder(self.schema.clone(), 2);
        let output = builder.tx();
        builder.spawn(async move {
            let prepared = prepared
                .get_or_try_init(|| Prepared::load(spec))
                .await?
                .clone();
            while let Some(batch) = input.try_next().await? {
                joiner.join(batch, &prepared, &output).await?;
            }
            Ok(())
        });
        let stream = builder.build().map(move |batch| {
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
    /// Rows a lookup may report that must not match, in the domain the index
    /// reports (row ids or row addresses): deleted rows, rows outside
    /// [`IndexJoinSpec::coverage`] and overlay-stale rows. `None` when nothing
    /// is blocked.
    lookup_mask: Option<Arc<RowAddrMask>>,
    /// Translates the row ids a row-id-domain index reports to addresses
    /// under stable row ids.
    row_id_index: Option<Arc<RowIdIndex>>,
    uncovered: Option<UncoveredRows>,
}

/// The keys of the rows the indices cannot see, for probing in memory.
///
/// Stored flat, one entry per row, so loading allocates per batch rather
/// than per row: the encoded keys share one buffer, and rows with the same
/// key hash are chained through `next`.
#[derive(Debug)]
struct UncoveredRows {
    converter: RowConverter,
    keys: Rows,
    row_ids: Vec<u64>,
    row_addrs: Vec<u64>,
    /// Key hash to the last entry with that hash.
    heads: HashMap<u64, u32, FxBuildHasher>,
    /// The previous entry with the same key hash, or [`END_OF_CHAIN`].
    next: Vec<u32>,
}

const END_OF_CHAIN: u32 = u32::MAX;

impl Prepared {
    async fn load(spec: Arc<IndexJoinSpec>) -> DFResult<Arc<Self>> {
        let results_are_row_addresses = spec.key.index.results_are_row_addresses();
        let lookup_mask = if results_are_row_addresses {
            Self::address_mask(&spec).await?
        } else {
            Self::row_id_mask(&spec).await?
        };
        let row_id_index = if results_are_row_addresses {
            None
        } else {
            get_row_id_index(&spec.dataset).await?
        };
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

    async fn row_id_mask(spec: &IndexJoinSpec) -> Result<Option<Arc<RowAddrMask>>> {
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
            let mut stale = stale_row_addrs(spec);
            if dataset.manifest.uses_stable_row_ids() {
                stale = translate_addr_treemap_to_row_ids(&dataset, &stale).await?;
            }
            Some(RowAddrMask::from_block(stale))
        };
        Ok(match (deletion_mask, stale_block) {
            (None, None) => None,
            (Some(mask), None) => Some(mask),
            (None, Some(block)) => Some(Arc::new(block)),
            (Some(mask), Some(block)) => Some(Arc::new(mask.as_ref().clone() & block)),
        })
    }

    async fn address_mask(spec: &IndexJoinSpec) -> Result<Option<Arc<RowAddrMask>>> {
        let dataset = &spec.dataset;
        let live = dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect::<RoaringBitmap>();
        let mut blocked = stale_row_addrs(spec);
        // Indexed fragments since removed, and live fragments the index does
        // not cover (those are matched through the uncovered map).
        for fragment_id in (&spec.coverage ^ &live).iter() {
            blocked.insert_fragment(fragment_id);
        }
        let deletions =
            futures::stream::iter(dataset.get_fragments().into_iter().filter(|fragment| {
                spec.coverage.contains(fragment.id() as u32)
                    && fragment.metadata().deletion_file.is_some()
            }))
            .map(|fragment| async move {
                let deletion_vector = fragment.get_deletion_vector().await?;
                Ok::<_, lance_core::Error>((fragment.id() as u32, deletion_vector))
            })
            .buffer_unordered(dataset.object_store.io_parallelism());
        let deletions = deletions.try_collect::<Vec<_>>().await?;
        for (fragment_id, deletion_vector) in deletions {
            if let Some(deletion_vector) = deletion_vector {
                let mut deleted = RowAddrTreeMap::new();
                deleted.insert_bitmap(fragment_id, RoaringBitmap::from(deletion_vector.as_ref()));
                blocked |= deleted;
            }
        }
        Ok((!blocked.is_empty()).then(|| Arc::new(RowAddrMask::from_block(blocked))))
    }
}

/// [`IndexJoinSpec::stale_rows`] as row addresses.
fn stale_row_addrs(spec: &IndexJoinSpec) -> RowAddrTreeMap {
    let mut stale = RowAddrTreeMap::new();
    for (&fragment_id, offsets) in &spec.stale_rows {
        stale.insert_bitmap(fragment_id, offsets.clone());
    }
    stale
}

impl UncoveredRows {
    async fn load(spec: &IndexJoinSpec) -> Result<Self> {
        let dataset = &spec.dataset;
        let column = &spec.key.column;
        let field = dataset.schema().field(column).ok_or_else(|| {
            lance_core::Error::internal(format!(
                "index join column '{column}' is missing from the dataset schema"
            ))
        })?;
        let converter = RowConverter::new(vec![SortField::new(field.data_type())])?;

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
            .project(&[column])?
            .with_row_id()
            .with_row_address();
        let mut stream = scanner.try_into_stream().await?;
        let mut uncovered = Self {
            keys: converter.empty_rows(0, 0),
            converter,
            row_ids: Vec::new(),
            row_addrs: Vec::new(),
            heads: HashMap::default(),
            next: Vec::new(),
        };
        while let Some(batch) = stream.try_next().await? {
            let keep = column_u64(&batch, ROW_ADDR)?
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
            let batch = if keep.true_count() == batch.num_rows() {
                batch
            } else {
                filter_record_batch(&batch, &keep)?
            };

            let keys = batch.column_by_name(column).cloned().ok_or_else(|| {
                lance_core::Error::internal(format!(
                    "index join scan is missing key column '{column}'"
                ))
            })?;
            uncovered.insert(
                keys,
                column_u64(&batch, ROW_ID)?,
                column_u64(&batch, ROW_ADDR)?,
            )?;
        }
        Ok(uncovered)
    }

    fn insert(
        &mut self,
        keys: ArrayRef,
        row_ids: &UInt64Array,
        row_addrs: &UInt64Array,
    ) -> Result<()> {
        let first = self.row_ids.len();
        self.converter
            .append(&mut self.keys, std::slice::from_ref(&keys))?;
        self.row_ids.extend_from_slice(row_ids.values());
        self.row_addrs.extend_from_slice(row_addrs.values());
        self.next.reserve(row_ids.len());
        for row in 0..row_ids.len() {
            let entry = u32::try_from(first + row).map_err(|_| {
                lance_core::Error::internal(format!(
                    "index join holds more than {} unindexed rows",
                    u32::MAX
                ))
            })?;
            // Null keys never match, so they are kept out of the chains.
            if keys.is_null(row) {
                self.next.push(END_OF_CHAIN);
                continue;
            }
            let hash = FxBuildHasher.hash_one(self.keys.row(entry as usize).as_ref());
            let previous = self.heads.insert(hash, entry).unwrap_or(END_OF_CHAIN);
            self.next.push(previous);
        }
        Ok(())
    }

    /// The `(row id, row address)` of every row whose key is `key`.
    fn get<'a>(&'a self, key: Row<'a>) -> impl Iterator<Item = (u64, u64)> + 'a {
        let mut entry = self
            .heads
            .get(&FxBuildHasher.hash_one(key.as_ref()))
            .copied()
            .unwrap_or(END_OF_CHAIN);
        std::iter::from_fn(move || {
            while entry != END_OF_CHAIN {
                let current = entry as usize;
                entry = self.next[current];
                if self.keys.row(current) == key {
                    return Some((self.row_ids[current], self.row_addrs[current]));
                }
            }
            None
        })
    }
}

fn column_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            lance_core::Error::internal(format!("index join scan is missing a u64 '{name}' column"))
        })
}

struct BatchJoiner {
    spec: Arc<IndexJoinSpec>,
    input_key: usize,
    schema: SchemaRef,
    /// The most matches one output batch holds.
    batch_size: usize,
    index_metrics: IndexMetrics,
}

impl BatchJoiner {
    /// Join one input batch, sending the output to `output` as it is built.
    ///
    /// Matches arrive from the index in no particular order and are emitted
    /// in batches of at most [`Self::batch_size`] rows. For a right join, the
    /// input rows nothing matched follow once every match is known.
    async fn join(
        &self,
        batch: RecordBatch,
        prepared: &Prepared,
        output: &Sender<DFResult<RecordBatch>>,
    ) -> Result<()> {
        let keys = batch.column(self.input_key).clone();
        let mut matched = vec![false; batch.num_rows()];

        // (input row, row in the index's domain), not yet filtered or resolved
        let mut input_rows = Vec::new();
        let mut rows = Vec::new();
        let mut lookups = self
            .spec
            .key
            .index
            .lookup(keys.clone(), &self.index_metrics)?;
        while let Some(found) = lookups.try_next().await? {
            input_rows.extend_from_slice(found.column(0).as_primitive::<UInt32Type>().values());
            rows.extend_from_slice(found.column(1).as_primitive::<UInt64Type>().values());
            while input_rows.len() >= self.batch_size {
                let rest_input_rows = input_rows.split_off(self.batch_size);
                let rest_rows = rows.split_off(self.batch_size);
                let (input_rows, targets) = self
                    .resolve(
                        std::mem::replace(&mut input_rows, rest_input_rows),
                        std::mem::replace(&mut rows, rest_rows),
                        prepared,
                    )
                    .await?;
                self.send_matches(&batch, &keys, input_rows, targets, &mut matched, output)
                    .await?;
            }
        }
        drop(lookups);
        let (mut input_rows, mut targets) = self.resolve(input_rows, rows, prepared).await?;

        if let Some(uncovered) = &prepared.uncovered {
            let encoded = uncovered
                .converter
                .convert_columns(std::slice::from_ref(&keys))?;
            for row in 0..batch.num_rows() {
                if keys.is_null(row) {
                    continue;
                }
                for target in uncovered.get(encoded.row(row)) {
                    input_rows.push(row as u32);
                    targets.push(target);
                }
            }
        }
        self.send_matches(&batch, &keys, input_rows, targets, &mut matched, output)
            .await?;

        if self.spec.join_type == JoinType::Right {
            let unmatched = (0..batch.num_rows() as u32)
                .filter(|&row| !matched[row as usize])
                .collect::<Vec<_>>();
            if !unmatched.is_empty() {
                let targets = vec![None; unmatched.len()];
                send(
                    output,
                    self.build_output(&batch, &keys, &unmatched, &targets)?,
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Drop the matches the lookup mask blocks and give each remaining one
    /// its `(row id, row address)`.
    async fn resolve(
        &self,
        input_rows: Vec<u32>,
        rows: Vec<u64>,
        prepared: &Prepared,
    ) -> Result<(Vec<u32>, Vec<(u64, u64)>)> {
        let (input_rows, rows): (Vec<u32>, Vec<u64>) = input_rows
            .into_iter()
            .zip(rows)
            .filter(|(_, row)| {
                prepared
                    .lookup_mask
                    .as_ref()
                    .is_none_or(|mask| mask.selected(*row))
            })
            .unzip();

        let dataset = &self.spec.dataset;
        if !dataset.manifest.uses_stable_row_ids() {
            return Ok((input_rows, rows.into_iter().map(|row| (row, row)).collect()));
        }
        let targets: Vec<Option<(u64, u64)>> = if let Some(row_id_index) = &prepared.row_id_index {
            row_id_index
                .get_many(&rows)?
                .into_iter()
                .zip(&rows)
                .map(|(addr, &row_id)| addr.map(|addr| (row_id, u64::from(addr))))
                .collect()
        } else {
            row_addrs_to_row_ids(dataset, rows.iter().map(|&addr| Some(addr)))
                .await?
                .into_iter()
                .zip(&rows)
                .map(|(row_id, &addr)| row_id.map(|row_id| (row_id, addr)))
                .collect()
        };
        Ok(input_rows
            .into_iter()
            .zip(targets)
            .filter_map(|(row, target)| target.map(|target| (row, target)))
            .unzip())
    }

    async fn send_matches(
        &self,
        batch: &RecordBatch,
        keys: &ArrayRef,
        input_rows: Vec<u32>,
        targets: Vec<(u64, u64)>,
        matched: &mut [bool],
        output: &Sender<DFResult<RecordBatch>>,
    ) -> Result<()> {
        for &row in &input_rows {
            matched[row as usize] = true;
        }
        for (input_rows, targets) in input_rows
            .chunks(self.batch_size)
            .zip(targets.chunks(self.batch_size))
        {
            let targets = targets.iter().copied().map(Some).collect::<Vec<_>>();
            send(
                output,
                self.build_output(batch, keys, input_rows, &targets)?,
            )
            .await?;
        }
        Ok(())
    }

    /// Output rows pairing input row `input_rows[i]` with table row
    /// `targets[i]`, given as `(row id, row address)`, or with nulls.
    fn build_output(
        &self,
        batch: &RecordBatch,
        keys: &ArrayRef,
        input_rows: &[u32],
        targets: &[Option<(u64, u64)>],
    ) -> Result<RecordBatch> {
        let input_rows = UInt32Array::from(input_rows.to_vec());
        let target_rows = input_rows
            .iter()
            .zip(targets)
            .map(|(row, target)| target.as_ref().and(row))
            .collect::<UInt32Array>();
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        for column in &self.spec.output_columns {
            columns.push(match column {
                TargetColumn::Key => take(keys.as_ref(), &target_rows, None)?,
                TargetColumn::RowId => Arc::new(
                    targets
                        .iter()
                        .map(|target| target.map(|(row_id, _)| row_id))
                        .collect::<UInt64Array>(),
                ) as ArrayRef,
                TargetColumn::RowAddr => Arc::new(
                    targets
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

async fn send(output: &Sender<DFResult<RecordBatch>>, batch: RecordBatch) -> Result<()> {
    output
        .send(Ok(batch))
        .await
        .map_err(|_| lance_core::Error::internal("index join output was dropped".to_string()))
}
