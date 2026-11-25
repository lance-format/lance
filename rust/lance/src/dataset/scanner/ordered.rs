// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::SortOptions;
use datafusion::physical_plan::ExecutionPlan;
use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::AnyQuery;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_index::scalar::registry::IndexOrderingCapability;
use lance_table::format::IndexMetadata;

use super::{FilterPlan, Scanner};
use crate::index::DatasetIndexExt;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::scalar_logical::load_named_scalar_segments;
use crate::io::exec::ordered_index_scan::OrderedIndexScanExec;
use crate::{Error, Result};

/// Whether a scan may read a B-tree index in key order to serve `order_by`.
///
/// Parse a mode from its name with [`str::parse`], ignoring ASCII case.
///
/// ```
/// # use lance::dataset::scanner::OrderedIndexMode;
/// # use lance::Result;
/// # fn test() -> Result<()> {
/// assert_eq!("off".parse::<OrderedIndexMode>()?, OrderedIndexMode::Off);
/// assert_eq!(OrderedIndexMode::default(), OrderedIndexMode::Auto);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OrderedIndexMode {
    /// Always sort.
    Off,
    /// Read the index in key order when it can serve the ordering, and sort
    /// otherwise.
    #[default]
    Auto,
}

impl OrderedIndexMode {
    const ALL: [Self; 2] = [Self::Off, Self::Auto];

    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
        }
    }
}

impl fmt::Display for OrderedIndexMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for OrderedIndexMode {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                let accepted = Self::ALL.map(Self::name).join(", ");
                Error::invalid_input(format!(
                    "unknown ordered_index mode '{name}', expected one of: {accepted}"
                ))
            })
    }
}

impl Scanner {
    /// Set whether to read a B-tree index in key order to serve `order_by`.
    ///
    /// With [`OrderedIndexMode::Auto`], the default, a scan with a limit that
    /// orders by one column with a B-tree index reads that index in key order
    /// and stops after `offset + limit` live rows, instead of sorting every
    /// row. This applies when the scan has no filter or a filter that the same
    /// index answers exactly. An index named in
    /// [`Self::with_ignored_scalar_indices`] never serves the read.
    /// [`OrderedIndexMode::Off`] always sorts. Other uses of scalar indices are
    /// not affected.
    ///
    /// ```
    /// # use lance::dataset::scanner::{ColumnOrdering, OrderedIndexMode};
    /// # use lance::{Dataset, Result};
    /// # async fn test(dataset: &Dataset) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".to_string())]))?
    ///     .limit(Some(10), None)?
    ///     .ordered_index(OrderedIndexMode::Off);
    /// let newest_ten = scan.try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn ordered_index(&mut self, mode: OrderedIndexMode) -> &mut Self {
        self.ordered_index = mode;
        self
    }

    pub(super) async fn try_ordered_read(
        &self,
        filter_plan: &mut FilterPlan,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        let (Some([ordering]), Some(limit)) = (self.ordering.as_deref(), self.limit) else {
            return None;
        };
        let is_legacy_storage = self
            .dataset
            .manifest()
            .data_storage_format
            .lance_file_format()
            == ConcreteFileVersion::V1;
        // The node applies the dataset's deletion files, not the list's, and has to
        // skip the index rows of every other fragment, which can be most of the index.
        let is_fragment_subset = self
            .fragments
            .as_ref()
            .is_some_and(|fragments| fragments != self.dataset.fragments().as_ref());
        // With stable row ids, an update can add the fragment it writes to the index's
        // fragment bitmap without indexing its rows (issue 10351), so the bitmap
        // cannot prove that the index covers every fragment.
        let has_stable_row_ids = self.dataset.manifest().uses_stable_row_ids();
        if self.ordered_index == OrderedIndexMode::Off
            || !self.use_scalar_index
            || self.external_row_mask.is_some()
            || self.physical_row_addr_prefilter.is_some()
            || self.include_deleted_rows
            || is_legacy_storage
            || is_fragment_subset
            || has_stable_row_ids
        {
            return None;
        }
        let fetch = self
            .offset
            .unwrap_or(0)
            .checked_add(limit)
            .and_then(|fetch| usize::try_from(fetch).ok())?;
        let schema = self.dataset.schema();
        let field = schema
            .fields
            .iter()
            .find(|field| field.name == ordering.column_name)?;
        let expr_filter = &filter_plan.expr_filter_plan;
        let (index_name, query) = match &expr_filter.index_query {
            None if expr_filter.full_expr.is_none() => (None, None),
            Some(ScalarIndexExpr::Query(search)) if expr_filter.is_exact_index_search() => {
                (Some(search.index_name.clone()), Some(search.query.clone()))
            }
            _ => return None,
        };
        let options = SortOptions {
            descending: !ordering.ascending,
            nulls_first: ordering.nulls_first,
        };
        // The sort never needed the index, so an index that fails to open
        // must not fail the scan.
        let scan = match self
            .ordered_index_scan(&field.name, field.id, index_name.as_deref(), query, options)
            .await
        {
            Ok(scan) => scan?,
            Err(error @ Error::NotSupported { .. }) => {
                log::debug!("sorting instead of reading the index in order: {error}");
                return None;
            }
            Err(error) => {
                log::warn!(
                    "sorting instead of reading the index on '{}' in order: {error}",
                    field.name
                );
                return None;
            }
        };
        let plan = scan.with_fetch(Some(fetch))?;
        filter_plan.disable_refine();
        Some(plan)
    }

    async fn ordered_index_scan(
        &self,
        column: &str,
        field_id: i32,
        index_name: Option<&str>,
        query: Option<Arc<dyn AnyQuery>>,
        options: SortOptions,
    ) -> Result<Option<OrderedIndexScanExec>> {
        let indices = self.dataset.load_indices().await?;
        let mut checked = HashSet::new();
        for index in indices.iter() {
            if index.keyed_field() != Some(field_id)
                || index_name.is_some_and(|name| name != index.name)
                || self.ignored_scalar_index_names.contains(&index.name)
                || !checked.insert(index.name.as_str())
            {
                continue;
            }
            let details = IndexDetails(fetch_index_details(&self.dataset, column, index).await?);
            if details.is_vector()
                || details.get_plugin()?.ordering_capability(&details.0)
                    != IndexOrderingCapability::Bidirectional
            {
                continue;
            }
            let segments = load_named_scalar_segments(&self.dataset, column, &index.name).await?;
            let Ok([segment]) = <[IndexMetadata; 1]>::try_from(segments) else {
                continue;
            };
            let Some(coverage) = &segment.fragment_bitmap else {
                continue;
            };
            if !self
                .dataset
                .fragments()
                .iter()
                .all(|fragment| coverage.contains(fragment.id as u32))
            {
                continue;
            }
            let scan =
                OrderedIndexScanExec::try_new(self.dataset.clone(), segment, query, options, None)
                    .await?;
            return Ok(Some(scan));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
        RecordBatchIterator, StringArray, StructArray, UInt64Array,
        cast::AsArray,
        types::{Int32Type, UInt64Type},
    };
    use arrow_schema::{DataType, Field, Schema, SortOptions};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, utils::address::RowAddress};
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
    };
    use lance_table::format::IndexMetadata;
    use rstest::rstest;

    use super::super::{ColumnOrdering, RowAddrMask, RowAddrTreeMap, Scanner};
    use super::OrderedIndexMode;
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::{UpdateBuilder, WriteDestination, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::covering::declare_covering;
    use crate::{Dataset, Error};

    const ROWS: i32 = 300;
    const FRAGMENT_ROWS: usize = 100;
    const INDEX: &str = "ts_idx";

    fn rows(ids: Range<i32>) -> RecordBatch {
        let keys = Int32Array::from_iter(
            ids.clone()
                .map(|id| (id % 40 != 3).then_some(id * 37 % ROWS)),
        );
        let values =
            Int64Array::from_iter(keys.iter().map(|key| key.map(|key| i64::from(key) * 10)));
        let nested = StructArray::from(vec![(
            Arc::new(Field::new("k", DataType::Int32, true)),
            Arc::new(keys.clone()) as ArrayRef,
        )]);
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            2,
            Arc::new(Float32Array::from_iter_values(
                ids.clone().flat_map(|id| [id as f32; 2]),
            )),
            None,
        )
        .unwrap();
        let text = StringArray::from_iter_values(
            ids.clone()
                .map(|id| if id % 2 == 0 { "even" } else { "odd" }),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("ts", DataType::Int32, true)
                .with_metadata(HashMap::from([("unit".into(), "s".into())])),
            Field::new("v", DataType::Int64, true),
            Field::new("s", nested.data_type().clone(), false),
            Field::new("vec", vectors.data_type().clone(), false),
            Field::new("text", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(ids)),
                Arc::new(keys),
                Arc::new(values),
                Arc::new(nested),
                Arc::new(vectors),
                Arc::new(text),
            ],
        )
        .unwrap()
    }

    async fn write(uri: &str, params: WriteParams) -> Dataset {
        let batch = rows(0..ROWS);
        let params = WriteParams {
            max_rows_per_file: FRAGMENT_ROWS,
            ..params
        };
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            uri,
            Some(params),
        )
        .await
        .unwrap()
    }

    fn btree() -> ScalarIndexParams {
        ScalarIndexParams::for_builtin(BuiltinIndexType::BTree)
    }

    async fn delete(dataset: &mut Dataset) {
        dataset
            .delete("ts < 3 OR ts > 296 OR ts % 11 = 5 OR id = 43")
            .await
            .unwrap();
    }

    async fn dataset(params: WriteParams) -> Dataset {
        let mut dataset = write("memory://", params).await;
        dataset
            .create_index(
                &["ts"],
                IndexType::BTree,
                Some(INDEX.into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        delete(&mut dataset).await;
        dataset
    }

    async fn scan(
        dataset: &Dataset,
        configure: &impl Fn(&mut Scanner),
        mode: OrderedIndexMode,
    ) -> (String, RecordBatch) {
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        scanner.ordered_index(mode);
        let plan = scanner.explain_plan(false).await.unwrap();
        (plan, scanner.try_into_batch().await.unwrap())
    }

    enum Expect<'a> {
        OrderedScan(&'a str),
        Sort,
    }

    async fn assert_ordered_read(
        dataset: &Dataset,
        configure: impl Fn(&mut Scanner),
        expect: Expect<'_>,
    ) -> RecordBatch {
        let (plan, rows) = scan(dataset, &configure, OrderedIndexMode::Auto).await;
        let (sorted_plan, sorted_rows) = scan(dataset, &configure, OrderedIndexMode::Off).await;
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        match expect {
            Expect::OrderedScan(node) => {
                assert!(plan.contains(node) && !plan.contains("SortExec"), "{plan}")
            }
            Expect::Sort => assert_eq!(plan, sorted_plan),
        }
        assert_eq!(rows, sorted_rows);
        rows
    }

    #[rstest]
    #[case::off("off", OrderedIndexMode::Off)]
    #[case::auto("auto", OrderedIndexMode::Auto)]
    #[case::upper_case("AUTO", OrderedIndexMode::Auto)]
    #[case::mixed_case("Off", OrderedIndexMode::Off)]
    fn test_ordered_index_mode_parses_names(#[case] name: &str, #[case] mode: OrderedIndexMode) {
        assert_eq!(name.parse::<OrderedIndexMode>().unwrap(), mode);
        assert_eq!(mode.to_string().parse::<OrderedIndexMode>().unwrap(), mode);
    }

    #[rstest]
    #[case::unknown("sorted")]
    #[case::empty("")]
    #[case::padded(" auto")]
    fn test_ordered_index_mode_rejects_unknown_names(#[case] name: &str) {
        let error = name.parse::<OrderedIndexMode>().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let expected = format!("unknown ordered_index mode '{name}', expected one of: off, auto");
        assert!(error.to_string().contains(&expected), "{error}");
    }

    fn order_by_ts(scanner: &mut Scanner) -> &mut Scanner {
        scanner
            .order_by(Some(vec![ColumnOrdering::asc_nulls_last("ts".into())]))
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_read_replaces_the_sort(
        #[values(true, false)] ascending: bool,
        #[values(true, false)] nulls_first: bool,
        #[values(
            None,
            Some("ts BETWEEN 20 AND 80"),
            Some("ts = 42"),
            Some("ts IN (3, 50, 77, 200)"),
            Some("ts IS NULL")
        )]
        filter: Option<&str>,
    ) {
        let dataset = dataset(WriteParams::default()).await;
        let ordering = ColumnOrdering {
            ascending,
            nulls_first,
            column_name: "ts".into(),
        };
        let order = SortOptions {
            descending: !ascending,
            nulls_first,
        };
        for (columns, limit, offset) in [(&["v"][..], 7, 0), (&["ts", "v"][..], 5, 4)] {
            let configure = |scanner: &mut Scanner| {
                scanner
                    .project(columns)
                    .unwrap()
                    .order_by(Some(vec![ordering.clone()]))
                    .unwrap()
                    .limit(Some(limit), Some(offset))
                    .unwrap();
                if let Some(filter) = filter {
                    scanner.filter(filter).unwrap();
                }
            };
            let node = format!(
                "OrderedIndexScan: index={INDEX}, order=[ts {order}], fetch={}",
                offset + limit
            );
            let rows = assert_ordered_read(&dataset, configure, Expect::OrderedScan(&node)).await;
            assert_eq!(rows.schema().fields().len(), columns.len());
            if offset == 0 {
                assert!(rows.num_rows() > 0, "{filter:?} selects live rows");
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_read_takes_row_ids_with_every_fragment_listed(
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let dataset = dataset(WriteParams {
            enable_stable_row_ids,
            ..Default::default()
        })
        .await;
        let fragments = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.metadata().clone())
            .collect::<Vec<_>>();
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .with_row_id()
                .with_fragments(fragments.clone())
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
                .unwrap()
                .limit(Some(20), Some(3))
                .unwrap();
        };
        let node = format!("OrderedIndexScan: index={INDEX}, order=[ts DESC NULLS LAST], fetch=23");
        let expect = if enable_stable_row_ids {
            Expect::Sort
        } else {
            Expect::OrderedScan(&node)
        };
        let rows = assert_ordered_read(&dataset, configure, expect).await;
        assert_eq!(rows.num_rows(), 20);
    }

    #[rstest]
    #[case::scalar_index_disabled(|scanner: &mut Scanner| {
        scanner.use_scalar_index(false);
    })]
    #[case::no_limit(|scanner: &mut Scanner| {
        scanner.limit(None, Some(3)).unwrap();
    })]
    #[case::offset_plus_limit_overflows(|scanner: &mut Scanner| {
        scanner.limit(Some(i64::MAX), Some(i64::MAX)).unwrap();
    })]
    #[case::two_order_columns(|scanner: &mut Scanner| {
        scanner
            .order_by(Some(vec![
                ColumnOrdering::asc_nulls_last("ts".into()),
                ColumnOrdering::asc_nulls_last("id".into()),
            ]))
            .unwrap();
    })]
    #[case::unindexed_order_column(|scanner: &mut Scanner| {
        scanner
            .order_by(Some(vec![ColumnOrdering::asc_nulls_last("v".into())]))
            .unwrap();
    })]
    #[case::filter_on_another_column(|scanner: &mut Scanner| {
        scanner.filter("v > 100").unwrap();
    })]
    #[case::filter_with_a_residual(|scanner: &mut Scanner| {
        scanner.filter("ts > 5 AND v < 2000").unwrap();
    })]
    #[case::filter_of_two_index_queries(|scanner: &mut Scanner| {
        scanner.filter("ts < 10 OR ts > 290").unwrap();
    })]
    #[case::row_id_prefilter(|scanner: &mut Scanner| {
        scanner.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(0..50)));
    })]
    #[case::physical_row_prefilter(|scanner: &mut Scanner| {
        scanner.with_physical_row_addr_prefilter(RowAddrTreeMap::from_iter(0..50));
    })]
    #[case::deleted_rows_included(|scanner: &mut Scanner| {
        scanner.with_row_id().include_deleted_rows();
    })]
    #[case::fragment_subset(|scanner: &mut Scanner| {
        let fragments = scanner.dataset.fragments()[1..].to_vec();
        scanner.with_fragments(fragments);
    })]
    #[case::vector_search(|scanner: &mut Scanner| {
        scanner.nearest("vec", &Float32Array::from(vec![10.0, 10.0]), 30).unwrap();
    })]
    #[tokio::test]
    async fn test_ordered_read_declines_scan_options(#[case] decline: fn(&mut Scanner)) {
        let dataset = dataset(WriteParams::default()).await;
        let configure = |scanner: &mut Scanner| {
            decline(order_by_ts(scanner));
        };
        assert_ordered_read(&dataset, configure, Expect::Sort).await;
    }

    #[rstest]
    #[case::order_column_index(INDEX, Expect::Sort)]
    #[case::unrelated_index(
        "id_idx",
        Expect::OrderedScan("OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=10")
    )]
    #[tokio::test]
    async fn test_ordered_read_skips_ignored_indices(
        #[case] ignored: &str,
        #[case] expect: Expect<'static>,
    ) {
        let mut dataset = dataset(WriteParams::default()).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner).with_ignored_scalar_indices([ignored]);
        };
        assert_ordered_read(&dataset, configure, expect).await;
    }

    /// Overwrites `column` with `value` in the row whose `ts` is `key`.
    async fn overlay(dataset: Dataset, column: &str, key: i32, value: i32) -> Dataset {
        let batch = dataset
            .scan()
            .project(&["ts"])
            .unwrap()
            .with_row_address()
            .filter(&format!("ts = {key}"))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch["ts"].as_primitive::<Int32Type>().value(0), key);
        let addr = batch[ROW_ADDR].as_primitive::<UInt64Type>().value(0);
        let field_id = dataset.schema().field(column).unwrap().id;
        let mut overlay = dataset
            .get_fragment(RowAddress::from(addr).fragment_id() as usize)
            .unwrap()
            .write_overlay(&dataset.schema().project_by_ids(&[field_id], true))
            .await
            .unwrap();
        let values = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                ROW_ADDR_FIELD.clone(),
                Field::from(dataset.schema().field(column).unwrap()),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![addr])),
                Arc::new(Int32Array::from(vec![value])),
            ],
        )
        .unwrap();
        overlay.write_batch(&values).await.unwrap();
        let group = overlay.finish().await.unwrap().unwrap();
        let read_version = dataset.version().version;
        Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![group],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_ordered_read_serves_a_covered_index_by_its_key() {
        let mut dataset = dataset(WriteParams::default()).await;
        let (ts_id, id_id) = declare_covering(&mut dataset, "ts", "id").await;
        let index = dataset.load_indices_by_name(INDEX).await.unwrap().remove(0);
        assert_eq!(index.fields, [ts_id, id_id]);
        let node = "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=10";
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner.project(&["id", "ts"]).unwrap());
        };
        assert_ordered_read(&dataset, configure, Expect::OrderedScan(node)).await;

        let dataset = overlay(dataset, "id", 4, -4).await;
        let rows = assert_ordered_read(&dataset, configure, Expect::OrderedScan(node)).await;
        assert!(
            rows["id"]
                .as_primitive::<Int32Type>()
                .values()
                .contains(&-4)
        );

        let dataset = overlay(dataset, "ts", 6, 1000).await;
        assert_ordered_read(&dataset, configure, Expect::Sort).await;
    }

    #[derive(Debug)]
    enum IndexShape {
        AppendedFragment,
        TwoSegments,
        BitmapIndex,
        NestedKey,
        LegacyStorage,
        NewerKeyOverlay,
        MissingIndexFiles,
        FullTextSearch,
        StableRowIdsClaimingAnUpdatedFragment,
    }

    async fn claim_updated_fragment() -> Dataset {
        let dataset = dataset(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        })
        .await;
        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("ts < 10")
            .unwrap()
            .set("v", "v + 1")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        assert_eq!(updated.rows_updated, 6);
        let mut dataset = Arc::unwrap_or_clone(updated.new_dataset);
        let index = dataset.load_indices_by_name(INDEX).await.unwrap().remove(0);
        let claimed = IndexMetadata {
            fragment_bitmap: Some(
                dataset
                    .fragments()
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect(),
            ),
            ..index.clone()
        };
        let transaction = Transaction::new(
            dataset.version().version,
            Operation::CreateIndex {
                new_indices: vec![claimed],
                removed_indices: vec![index],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();
        dataset
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_read_declines_index_shapes(
        #[values(
            IndexShape::AppendedFragment,
            IndexShape::TwoSegments,
            IndexShape::BitmapIndex,
            IndexShape::NestedKey,
            IndexShape::LegacyStorage,
            IndexShape::NewerKeyOverlay,
            IndexShape::MissingIndexFiles,
            IndexShape::FullTextSearch,
            IndexShape::StableRowIdsClaimingAnUpdatedFragment
        )]
        shape: IndexShape,
    ) {
        let mut order_column = "ts";
        let mut full_text_query = None;
        let tempdir = TempStrDir::default();
        let dataset = match shape {
            IndexShape::AppendedFragment => {
                let mut dataset = dataset(WriteParams::default()).await;
                let batch = rows(ROWS..ROWS + 30);
                dataset
                    .append(
                        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                        None,
                    )
                    .await
                    .unwrap();
                dataset
            }
            IndexShape::TwoSegments => {
                let mut dataset = write("memory://", WriteParams::default()).await;
                let mut segments = Vec::new();
                for fragment_ids in [vec![0, 1], vec![2]] {
                    let segment = dataset
                        .create_index_builder(&["ts"], IndexType::BTree, &btree())
                        .name(INDEX.into())
                        .fragments(fragment_ids)
                        .execute_uncommitted()
                        .await
                        .unwrap();
                    segments.push(segment);
                }
                dataset
                    .commit_existing_index_segments(INDEX, "ts", segments)
                    .await
                    .unwrap();
                delete(&mut dataset).await;
                dataset
            }
            IndexShape::BitmapIndex => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(
                        &["id"],
                        IndexType::Bitmap,
                        None,
                        &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                        false,
                    )
                    .await
                    .unwrap();
                order_column = "id";
                dataset
            }
            IndexShape::NestedKey => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(&["s.k"], IndexType::BTree, None, &btree(), false)
                    .await
                    .unwrap();
                order_column = "s.k";
                dataset
            }
            IndexShape::LegacyStorage => {
                dataset(WriteParams {
                    data_storage_version: Some(LanceFileVersion::Legacy),
                    ..Default::default()
                })
                .await
            }
            IndexShape::NewerKeyOverlay => {
                overlay(dataset(WriteParams::default()).await, "ts", 150, -1).await
            }
            IndexShape::MissingIndexFiles => {
                let mut dataset = write(tempdir.as_str(), WriteParams::default()).await;
                dataset
                    .create_index(
                        &["ts"],
                        IndexType::BTree,
                        Some(INDEX.into()),
                        &btree(),
                        false,
                    )
                    .await
                    .unwrap();
                let segment = dataset.load_indices_by_name(INDEX).await.unwrap().remove(0);
                dataset
                    .object_store
                    .remove_dir_all(dataset.indices_dir().join(segment.uuid.to_string()))
                    .await
                    .unwrap();
                Dataset::open(tempdir.as_str()).await.unwrap()
            }
            IndexShape::FullTextSearch => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(
                        &["text"],
                        IndexType::Inverted,
                        None,
                        &InvertedIndexParams::default(),
                        false,
                    )
                    .await
                    .unwrap();
                full_text_query = Some(FullTextSearchQuery::new("even".into()));
                dataset
            }
            IndexShape::StableRowIdsClaimingAnUpdatedFragment => claim_updated_fragment().await,
        };
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .order_by(Some(vec![ColumnOrdering::asc_nulls_last(
                    order_column.into(),
                )]))
                .unwrap()
                .limit(Some(10), None)
                .unwrap();
            if let Some(query) = &full_text_query {
                scanner.full_text_search(query.clone()).unwrap();
            }
        };
        let rows = assert_ordered_read(&dataset, configure, Expect::Sort).await;
        assert_eq!(rows.num_rows(), 10, "{shape:?}");
    }
}
