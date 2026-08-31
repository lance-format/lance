// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::traits::{import_vec_from_method, import_vec_to_rust};
use arrow::array::Float32Array;
use arrow::{ffi::FFI_ArrowSchema, ffi_stream::FFI_ArrowArrayStream};
use arrow_schema::SchemaRef;
use jni::objects::{JObject, JString, JValueGen};
use jni::sys::{JNI_TRUE, jboolean, jint};
use jni::{JNIEnv, sys::jlong};
use lance::dataset::scanner::{
    AggregateExpr, ColumnOrdering, DatasetRecordBatchStream, ExecutionStatsCallback,
    ExecutionSummaryCounts, MaterializationStyle, Scanner,
};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::{
    DocumentGranularity,
    query::{
        BooleanQuery as FtsBooleanQuery, BoostQuery as FtsBoostQuery, FtsQuery,
        MatchQuery as FtsMatchQuery, MultiMatchQuery as FtsMultiMatchQuery, Occur as FtsOccur,
        PhraseQuery as FtsPhraseQuery,
    },
};
use lance_io::ffi::to_ffi_arrow_array_stream;
use lance_linalg::distance::DistanceType;

use crate::{
    RT, block_on,
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    traits::IntoJava,
    utils::parse_approx_mode,
};

pub const NATIVE_SCANNER: &str = "nativeScannerHandle";

#[derive(Clone)]
pub struct BlockingScanner {
    pub(crate) inner: Arc<Scanner>,
    stats: Arc<Mutex<Option<ExecutionSummaryCounts>>>,
}

impl BlockingScanner {
    pub fn create(mut scanner: Scanner, collect_stats: bool) -> Self {
        let stats = Arc::new(Mutex::new(None));
        if collect_stats {
            let stats_for_callback = stats.clone();
            let callback: ExecutionStatsCallback = Arc::new(move |counts| {
                let mut guard = stats_for_callback.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(counts.clone());
            });
            scanner.scan_stats_callback(callback);
        }

        Self {
            inner: Arc::new(scanner),
            stats,
        }
    }

    fn reset_stats(&self) {
        let mut guard = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }

    pub fn open_stream(&self) -> Result<DatasetRecordBatchStream> {
        self.reset_stats();
        let res = block_on(self.inner.try_into_stream())?;
        Ok(res)
    }

    pub fn schema(&self) -> Result<SchemaRef> {
        let res = block_on(self.inner.schema())?;
        Ok(res)
    }

    pub fn count_rows(&self) -> Result<u64> {
        let res = block_on(self.inner.count_rows())?;
        Ok(res)
    }

    pub fn get_stats(&self) -> Option<ExecutionSummaryCounts> {
        self.stats.lock().unwrap().clone()
    }
}

///////////////////
// Shared Helpers //
///////////////////

/// Build FTS query from Java FullTextQuery object
/// Made pub(crate) to be reused by async_scanner
pub(crate) fn build_full_text_search_query<'a>(
    env: &mut JNIEnv<'a>,
    java_obj: JObject,
) -> Result<FtsQuery> {
    let type_obj = env
        .call_method(
            &java_obj,
            "getType",
            "()Lorg/lance/ipc/FullTextQuery$Type;",
            &[],
        )?
        .l()?;
    let type_name = env.get_string_from_method(&type_obj, "name")?;

    match type_name.as_str() {
        "MATCH" => {
            let query_text = env.get_string_from_method(&java_obj, "getQueryText")?;
            let column = env.get_string_from_method(&java_obj, "getColumn")?;
            let boost = env.get_f32_from_method(&java_obj, "getBoost")?;
            let fuzziness = env.get_optional_u32_from_method(&java_obj, "getFuzziness")?;
            let max_expansions = env.get_int_as_usize_from_method(&java_obj, "getMaxExpansions")?;
            let operator = env.get_fts_operator_from_method(&java_obj)?;
            let prefix_length = env.get_u32_from_method(&java_obj, "getPrefixLength")?;
            let document_granularity = get_document_granularity(env, &java_obj)?;

            let mut query = FtsMatchQuery::new(query_text);
            query = query.with_column(Some(column));
            query = query
                .with_boost(boost)
                .with_fuzziness(fuzziness)
                .with_max_expansions(max_expansions)
                .with_operator(operator)
                .with_prefix_length(prefix_length);
            if let Some(document_granularity) = document_granularity {
                query = query.with_document_granularity(document_granularity);
            }

            Ok(FtsQuery::Match(query))
        }
        "MATCH_PHRASE" => {
            let query_text = env.get_string_from_method(&java_obj, "getQueryText")?;
            let column = env.get_string_from_method(&java_obj, "getColumn")?;
            let slop = env.get_u32_from_method(&java_obj, "getSlop")?;
            let document_granularity = get_document_granularity(env, &java_obj)?;

            let mut query = FtsPhraseQuery::new(query_text);
            query = query.with_column(Some(column));
            query = query.with_slop(slop);
            if let Some(document_granularity) = document_granularity {
                query = query.with_document_granularity(document_granularity);
            }

            Ok(FtsQuery::Phrase(query))
        }
        "MULTI_MATCH" => {
            let query_text = env.get_string_from_method(&java_obj, "getQueryText")?;
            let columns: Vec<String> =
                import_vec_from_method(env, &java_obj, "getColumns", |env, elem| {
                    let jstr = JString::from(elem);
                    let value: String = env.get_string(&jstr)?.into();
                    Ok(value)
                })?;

            let boosts: Option<Vec<f32>> =
                env.get_optional_from_method(&java_obj, "getBoosts", |env, list_obj| {
                    import_vec_to_rust(env, &list_obj, |env, elem| {
                        env.get_f32_from_method(&elem, "floatValue")
                    })
                })?;
            let operator = env.get_fts_operator_from_method(&java_obj)?;

            let mut query = FtsMultiMatchQuery::try_new(query_text, columns)?;
            if let Some(boosts) = boosts {
                query = query.try_with_boosts(boosts)?;
            }
            query = query.with_operator(operator);

            Ok(FtsQuery::MultiMatch(query))
        }
        "BOOST" => {
            let positive_obj = env
                .call_method(
                    &java_obj,
                    "getPositive",
                    "()Lorg/lance/ipc/FullTextQuery;",
                    &[],
                )?
                .l()?;
            if positive_obj.is_null() {
                return Err(Error::input_error(
                    "positive query must not be null in BOOST FullTextQuery".to_string(),
                ));
            }
            let negative_obj = env
                .call_method(
                    &java_obj,
                    "getNegative",
                    "()Lorg/lance/ipc/FullTextQuery;",
                    &[],
                )?
                .l()?;
            if negative_obj.is_null() {
                return Err(Error::input_error(
                    "negative query must not be null in BOOST FullTextQuery".to_string(),
                ));
            }

            let positive = build_full_text_search_query(env, positive_obj)?;
            let negative = build_full_text_search_query(env, negative_obj)?;
            let negative_boost = env.get_f32_from_method(&java_obj, "getNegativeBoost")?;

            let query = FtsBoostQuery::new(positive, negative, Some(negative_boost));
            Ok(FtsQuery::Boost(query))
        }
        "BOOLEAN" => {
            let clauses: Vec<(FtsOccur, FtsQuery)> =
                import_vec_from_method(env, &java_obj, "getClauses", |env, clause_obj| {
                    let occur = env.get_occur_from_method(&clause_obj)?;

                    let query_obj = env
                        .call_method(
                            &clause_obj,
                            "getQuery",
                            "()Lorg/lance/ipc/FullTextQuery;",
                            &[],
                        )?
                        .l()?;
                    if query_obj.is_null() {
                        return Err(Error::input_error(
                            "BooleanClause query must not be null".to_string(),
                        ));
                    }
                    let query = build_full_text_search_query(env, query_obj)?;
                    Ok((occur, query))
                })?;

            let boolean_query = FtsBooleanQuery::new(clauses);
            Ok(FtsQuery::Boolean(boolean_query))
        }
        other => Err(Error::input_error(format!(
            "Unsupported FullTextQuery type: {}",
            other
        ))),
    }
}

fn get_document_granularity(
    env: &mut JNIEnv<'_>,
    java_obj: &JObject,
) -> Result<Option<DocumentGranularity>> {
    env.get_optional_from_method(
        java_obj,
        "getDocumentGranularity",
        |env, granularity_obj| {
            let value = env.get_string_from_method(&granularity_obj, "toRustString")?;
            DocumentGranularity::try_from(value.as_str()).map_err(Error::from)
        },
    )
}

/// Scanner options passed from JNI - shared between blocking and async scanners
pub(crate) struct ScannerOptions<'a> {
    pub fragment_ids_obj: JObject<'a>,
    pub columns_obj: JObject<'a>,
    pub substrait_filter_obj: JObject<'a>,
    pub filter_obj: JObject<'a>,
    pub batch_size_obj: JObject<'a>,
    pub batch_size_bytes_obj: JObject<'a>,
    pub io_buffer_size_obj: JObject<'a>,
    pub limit_obj: JObject<'a>,
    pub offset_obj: JObject<'a>,
    pub query_obj: JObject<'a>,
    pub fts_query_obj: JObject<'a>,
    pub prefilter: jboolean,
    pub with_row_id: jboolean,
    pub with_row_address: jboolean,
    pub batch_readahead: jint,
    pub fragment_readahead_obj: JObject<'a>,
    pub scan_in_order: jboolean,
    pub late_materialization_obj: JObject<'a>,
    pub column_orderings: JObject<'a>,
    pub use_scalar_index: jboolean,
    pub fast_search: jboolean,
    pub substrait_aggregate_obj: JObject<'a>,
    pub include_deleted_rows: jboolean,
    pub strict_batch_size: jboolean,
    pub disable_scoring_autoprojection: jboolean,
}

/// Build a scanner with options applied - shared by blocking and async scanners
pub(crate) fn build_scanner_with_options<'a>(
    env: &mut JNIEnv<'a>,
    dataset: &lance::Dataset,
    options: ScannerOptions<'a>,
) -> Result<Scanner> {
    let mut scanner = dataset.scan();

    // handle fragment_ids
    let fragment_ids_opt = env.get_ints_opt(&options.fragment_ids_obj)?;
    if let Some(fragment_ids) = fragment_ids_opt {
        let mut fragments = Vec::with_capacity(fragment_ids.len());
        for fragment_id in fragment_ids {
            let Some(fragment) = dataset.get_fragment(fragment_id as usize) else {
                return Err(Error::input_error(format!(
                    "Fragment {fragment_id} not found"
                )));
            };
            fragments.push(fragment.metadata().clone());
        }
        scanner.with_fragments(fragments);
    }

    let columns_opt = env.get_strings_opt(&options.columns_obj)?;
    if let Some(columns) = columns_opt {
        scanner.project(&columns)?;
    };

    let substrait_opt = env.get_bytes_opt(&options.substrait_filter_obj)?;
    if let Some(substrait) = substrait_opt {
        block_on(async { scanner.filter_substrait(substrait) })?;
    }

    let filter_opt = env.get_string_opt(&options.filter_obj)?;
    if let Some(filter) = filter_opt {
        scanner.filter(filter.as_str())?;
    }

    let batch_size_opt = env.get_long_opt(&options.batch_size_obj)?;
    if let Some(batch_size) = batch_size_opt {
        scanner.batch_size(batch_size as usize);
    }

    let batch_size_bytes_opt = env.get_long_opt(&options.batch_size_bytes_obj)?;
    if let Some(batch_size_bytes) = batch_size_bytes_opt {
        let batch_size_bytes = u64::try_from(batch_size_bytes).map_err(|_| {
            Error::input_error(format!(
                "batchSizeBytes must be non-negative, got {batch_size_bytes}"
            ))
        })?;
        scanner.batch_size_bytes(batch_size_bytes);
    }

    let io_buffer_size_opt = env.get_long_opt(&options.io_buffer_size_obj)?;
    if let Some(io_buffer_size) = io_buffer_size_opt {
        let io_buffer_size = u64::try_from(io_buffer_size).map_err(|_| {
            Error::input_error(format!(
                "ioBufferSize must be non-negative, got {io_buffer_size}"
            ))
        })?;
        scanner.io_buffer_size(io_buffer_size);
    }

    let limit_opt = env.get_long_opt(&options.limit_obj)?;
    let offset_opt = env.get_long_opt(&options.offset_obj)?;
    scanner
        .limit(limit_opt, offset_opt)
        .map_err(|err| Error::input_error(err.to_string()))?;

    if options.with_row_id == JNI_TRUE {
        scanner.with_row_id();
    }

    if options.with_row_address == JNI_TRUE {
        scanner.with_row_address();
    }

    if options.prefilter == JNI_TRUE {
        scanner.prefilter(true);
    }

    scanner.use_scalar_index(options.use_scalar_index == JNI_TRUE);

    env.get_optional(&options.query_obj, |env, java_obj| {
        // Set column and key for nearest search
        let column = env.get_string_from_method(&java_obj, "getColumn")?;
        let key_array = env.get_vec_f32_from_method(&java_obj, "getKey")?;
        let key = Float32Array::from(key_array);
        let k = env.get_int_as_usize_from_method(&java_obj, "getK")?;
        scanner
            .nearest(&column, &key, k)
            .map_err(|err| Error::input_error(err.to_string()))?;

        let minimum_nprobes = env.get_int_as_usize_from_method(&java_obj, "getMinimumNprobes")?;
        scanner.minimum_nprobes(minimum_nprobes);

        let maximum_nprobes = env.get_optional_usize_from_method(&java_obj, "getMaximumNprobes")?;
        if let Some(maximum_nprobes) = maximum_nprobes {
            scanner.maximum_nprobes(maximum_nprobes);
        }

        if let Some(ef) = env.get_optional_usize_from_method(&java_obj, "getEf")? {
            scanner.ef(ef);
        }

        if let Some(centroid_ef) = env.get_optional_usize_from_method(&java_obj, "getCentroidEf")? {
            scanner.centroid_ef(centroid_ef);
        }

        if let Some(refine_factor) =
            env.get_optional_u32_from_method(&java_obj, "getRefineFactor")?
        {
            scanner.refine(refine_factor);
        }

        if let Some(distance_type_str) =
            env.get_optional_string_from_method(&java_obj, "getDistanceTypeString")?
        {
            let distance_type = DistanceType::try_from(distance_type_str.as_str())?;
            scanner.distance_metric(distance_type);
        }

        let use_index = env.get_boolean_from_method(&java_obj, "isUseIndex")?;
        scanner.use_index(use_index);

        let query_parallelism = env
            .call_method(&java_obj, "getQueryParallelism", "()I", &[])?
            .i()?;
        scanner.query_parallelism(query_parallelism);

        let approx_mode_str = env.get_string_from_method(&java_obj, "getApproxModeString")?;
        scanner.approx_mode(parse_approx_mode(&approx_mode_str)?);
        Ok(())
    })?;

    env.get_optional(&options.fts_query_obj, |env, java_obj| {
        let fts_query = build_full_text_search_query(env, java_obj)?;
        let full_text_query = FullTextSearchQuery::new_query(fts_query);
        scanner.full_text_search(full_text_query)?;
        Ok(())
    })?;

    if options.fast_search == JNI_TRUE {
        scanner.fast_search();
    }

    scanner.batch_readahead(options.batch_readahead as usize);

    let fragment_readahead_opt = env.get_int_opt(&options.fragment_readahead_obj)?;
    if let Some(fragment_readahead) = fragment_readahead_opt {
        let fragment_readahead = usize::try_from(fragment_readahead).map_err(|_| {
            Error::input_error(format!(
                "fragmentReadahead must be greater than 0, got {fragment_readahead}"
            ))
        })?;
        if fragment_readahead == 0 {
            return Err(Error::input_error(
                "fragmentReadahead must be greater than 0, got 0".to_string(),
            ));
        }
        scanner.fragment_readahead(fragment_readahead);
    }

    scanner.scan_in_order(options.scan_in_order == JNI_TRUE);

    env.get_optional(&options.late_materialization_obj, |env, java_obj| {
        let style_name = env.get_string_from_method(&java_obj, "toRustString")?;
        let style = match style_name.as_str() {
            "heuristic" => MaterializationStyle::Heuristic,
            "all_late" => MaterializationStyle::AllLate,
            "all_early" => MaterializationStyle::AllEarly,
            "all_early_except" => {
                let columns: Vec<String> =
                    import_vec_from_method(env, &java_obj, "getColumns", |env, elem| {
                        let jstr = JString::from(elem);
                        Ok(env.get_string(&jstr)?.into())
                    })?;
                MaterializationStyle::all_early_except(&columns, dataset.schema())?
            }
            other => {
                return Err(Error::input_error(format!(
                    "Unsupported materialization style: {other}"
                )));
            }
        };
        scanner.materialization_style(style);
        Ok(())
    })?;

    env.get_optional(&options.column_orderings, |env, java_obj| {
        let list = env.get_list(&java_obj)?;
        let mut iter = list.iter(env)?;
        let mut results = Vec::with_capacity(list.size(env)? as usize);
        while let Some(elem) = iter.next(env)? {
            let column_name = env.get_string_from_method(&elem, "getColumnName")?;
            let nulls_first = env.get_boolean_from_method(&elem, "isNullFirst")?;
            let ascending = env.get_boolean_from_method(&elem, "isAscending")?;
            let col_order = ColumnOrdering {
                ascending,
                nulls_first,
                column_name,
            };
            results.push(col_order)
        }
        scanner.order_by(Some(results))?;
        Ok(())
    })?;

    let substrait_aggregate_opt = env.get_bytes_opt(&options.substrait_aggregate_obj)?;
    if let Some(substrait_aggregate) = substrait_aggregate_opt {
        scanner.aggregate(AggregateExpr::substrait(substrait_aggregate))?;
    }

    if options.include_deleted_rows == JNI_TRUE {
        scanner.include_deleted_rows();
    }

    scanner.strict_batch_size(options.strict_batch_size == JNI_TRUE);

    if options.disable_scoring_autoprojection == JNI_TRUE {
        scanner.disable_scoring_autoprojection();
    }

    Ok(scanner)
}

///////////////////
// Write Methods //
///////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_LanceScanner_createScanner<'local>(
    mut env: JNIEnv<'local>,
    _reader: JObject<'local>,
    jdataset: JObject<'local>,
    fragment_ids_obj: JObject<'local>, // Optional<List<Integer>>
    columns_obj: JObject<'local>,      // Optional<List<String>>
    substrait_filter_obj: JObject<'local>, // Optional<ByteBuffer>
    filter_obj: JObject<'local>,       // Optional<String>
    batch_size_obj: JObject<'local>,   // Optional<Long>
    batch_size_bytes_obj: JObject<'local>, // Optional<Long>
    io_buffer_size_obj: JObject<'local>, // Optional<Long>
    limit_obj: JObject<'local>,        // Optional<Integer>
    offset_obj: JObject<'local>,       // Optional<Integer>
    query_obj: JObject<'local>,        // Optional<Query>
    fts_query_obj: JObject<'local>,    // Optional<FullTextQuery>
    prefilter: jboolean,               // boolean
    with_row_id: jboolean,             // boolean
    with_row_address: jboolean,        // boolean
    batch_readahead: jint,             // int
    fragment_readahead_obj: JObject<'local>, // Optional<Integer>
    scan_in_order: jboolean,           // boolean
    late_materialization_obj: JObject<'local>, // Optional<MaterializationStyle>
    column_orderings: JObject<'local>, // Optional<List<ColumnOrdering>>
    use_scalar_index: jboolean,        // boolean
    fast_search: jboolean,             // boolean
    substrait_aggregate_obj: JObject<'local>, // Optional<ByteBuffer>
    collect_stats: jboolean,           // boolean
    include_deleted_rows: jboolean,    // boolean
    strict_batch_size: jboolean,       // boolean
    disable_scoring_autoprojection: jboolean, // boolean
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_scanner(
            &mut env,
            jdataset,
            fragment_ids_obj,
            columns_obj,
            substrait_filter_obj,
            filter_obj,
            batch_size_obj,
            batch_size_bytes_obj,
            io_buffer_size_obj,
            limit_obj,
            offset_obj,
            query_obj,
            fts_query_obj,
            prefilter,
            with_row_id,
            with_row_address,
            batch_readahead,
            fragment_readahead_obj,
            scan_in_order,
            late_materialization_obj,
            column_orderings,
            use_scalar_index,
            fast_search,
            substrait_aggregate_obj,
            collect_stats,
            include_deleted_rows,
            strict_batch_size,
            disable_scoring_autoprojection,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_scanner<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject<'local>,
    fragment_ids_obj: JObject<'local>,
    columns_obj: JObject<'local>,
    substrait_filter_obj: JObject<'local>,
    filter_obj: JObject<'local>,
    batch_size_obj: JObject<'local>,
    batch_size_bytes_obj: JObject<'local>,
    io_buffer_size_obj: JObject<'local>,
    limit_obj: JObject<'local>,
    offset_obj: JObject<'local>,
    query_obj: JObject<'local>,
    fts_query_obj: JObject<'local>,
    prefilter: jboolean,
    with_row_id: jboolean,
    with_row_address: jboolean,
    batch_readahead: jint,
    fragment_readahead_obj: JObject<'local>,
    scan_in_order: jboolean,
    late_materialization_obj: JObject<'local>,
    column_orderings: JObject<'local>,
    use_scalar_index: jboolean,
    fast_search: jboolean,
    substrait_aggregate_obj: JObject<'local>,
    collect_stats: jboolean,
    include_deleted_rows: jboolean,
    strict_batch_size: jboolean,
    disable_scoring_autoprojection: jboolean,
) -> Result<JObject<'local>> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
    let dataset = dataset_guard.inner.clone();
    drop(dataset_guard);

    let options = ScannerOptions {
        fragment_ids_obj,
        columns_obj,
        substrait_filter_obj,
        filter_obj,
        batch_size_obj,
        batch_size_bytes_obj,
        io_buffer_size_obj,
        limit_obj,
        offset_obj,
        query_obj,
        fts_query_obj,
        prefilter,
        with_row_id,
        with_row_address,
        batch_readahead,
        fragment_readahead_obj,
        scan_in_order,
        late_materialization_obj,
        column_orderings,
        use_scalar_index,
        fast_search,
        substrait_aggregate_obj,
        include_deleted_rows,
        strict_batch_size,
        disable_scoring_autoprojection,
    };

    let scanner = build_scanner_with_options(env, &dataset, options)?;

    let scanner = BlockingScanner::create(scanner, collect_stats == JNI_TRUE);
    scanner.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_LanceScanner_releaseNativeScanner(
    mut env: JNIEnv,
    j_scanner: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_native_scanner(&mut env, j_scanner));
}

fn inner_release_native_scanner(env: &mut JNIEnv, j_scanner: JObject) -> Result<()> {
    let _: BlockingScanner = unsafe { env.take_rust_field(j_scanner, NATIVE_SCANNER) }?;
    Ok(())
}

impl IntoJava for BlockingScanner {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        attach_native_scanner(env, self)
    }
}

fn attach_native_scanner<'local>(
    env: &mut JNIEnv<'local>,
    scanner: BlockingScanner,
) -> Result<JObject<'local>> {
    let j_scanner = create_java_scanner_object(env)?;
    // This block sets a native Rust object (scanner) as a field in the Java object (j_scanner).
    // Caution: This creates a potential for memory leaks. The Rust object (scanner) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_scanner`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    unsafe { env.set_rust_field(&j_scanner, NATIVE_SCANNER, scanner) }?;
    Ok(j_scanner)
}

fn create_java_scanner_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let res = env.new_object("org/lance/ipc/LanceScanner", "()V", &[])?;
    Ok(res)
}

//////////////////
// Read Methods //
//////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_LanceScanner_openStream(
    mut env: JNIEnv,
    j_scanner: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_open_stream(&mut env, j_scanner, stream_addr));
}

fn inner_open_stream(env: &mut JNIEnv, j_scanner: JObject, stream_addr: jlong) -> Result<()> {
    if stream_addr == 0 {
        return Err(Error::input_error(
            "ArrowArrayStream address must not be null".to_string(),
        ));
    }

    // Reject a stream that already holds a producer. We write the C struct in place below with
    // `ptr::write_unaligned`, which does not run any destructor on the previous contents. If the
    // caller passed a stream whose `release` callback is already set (e.g. it was populated by an
    // earlier export and not yet released), overwriting it would drop that callback and leak the
    // first producer's resources. A freshly-allocated `ArrowArrayStream` has a null `release`, per
    // the Arrow C Data Interface, so requiring `release == None` is the contract for "empty".
    //
    // The struct is allocated by Arrow Java inside an ArrowBuf and is not guaranteed to be aligned
    // (hence `write_unaligned` below), so we must not form a reference to it. We read only the
    // `release` field through an unaligned read: `addr_of!` computes the field address without
    // creating an intermediate, possibly-unaligned reference, and the field is an `Option<fn>`
    // which is `Copy` with no destructor, so reading a copy of it leaves the caller's stream
    // untouched.
    let release_is_set = unsafe {
        let stream_ptr = stream_addr as *const FFI_ArrowArrayStream;
        let release = std::ptr::read_unaligned(std::ptr::addr_of!((*stream_ptr).release));
        release.is_some()
    };
    if release_is_set {
        return Err(Error::input_error(
            "ArrowArrayStream is already populated; exporting into it would leak the existing \
             producer. Pass a freshly-allocated, empty stream."
                .to_string(),
        ));
    }

    let record_batch_stream = {
        let scanner_guard =
            unsafe { env.get_rust_field::<_, _, BlockingScanner>(j_scanner, NATIVE_SCANNER) }?;
        scanner_guard.open_stream()?
    };
    let ffi_stream = to_ffi_arrow_array_stream(record_batch_stream, RT.handle().clone())?;
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_LanceScanner_importFfiSchema(
    mut env: JNIEnv,
    j_scanner: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_import_ffi_schema(&mut env, j_scanner, schema_addr)
    );
}

fn inner_import_ffi_schema(env: &mut JNIEnv, j_scanner: JObject, schema_addr: jlong) -> Result<()> {
    let schema = {
        let scanner_guard =
            unsafe { env.get_rust_field::<_, _, BlockingScanner>(j_scanner, NATIVE_SCANNER) }?;
        scanner_guard.schema()?
    };
    let ffi_schema = FFI_ArrowSchema::try_from(&*schema)?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_LanceScanner_nativeCountRows(
    mut env: JNIEnv,
    j_scanner: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_count_rows(&mut env, j_scanner), -1) as jlong
}

fn inner_count_rows(env: &mut JNIEnv, j_scanner: JObject) -> Result<u64> {
    let scanner_guard =
        unsafe { env.get_rust_field::<_, _, BlockingScanner>(j_scanner, NATIVE_SCANNER) }?;
    scanner_guard.count_rows()
}

const SCAN_STATS_CLASS: &str = "org/lance/ipc/ScanStats";
const SCAN_STATS_CONSTRUCTOR_SIG: &str = "(JJJJJJJJLjava/util/Map;Ljava/util/Map;)V";

fn export_usize_map<'a>(env: &mut JNIEnv<'a>, map: &HashMap<String, usize>) -> Result<JObject<'a>> {
    let hash_map = env.new_object("java/util/HashMap", "()V", &[])?;
    for (key, value) in map {
        let java_key: JObject = env.new_string(key)?.into();
        let java_value =
            env.new_object("java/lang/Long", "(J)V", &[JValueGen::Long(*value as i64)])?;
        env.call_method(
            &hash_map,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValueGen::Object(&java_key), JValueGen::Object(&java_value)],
        )?;
    }
    Ok(hash_map)
}

impl IntoJava for &ExecutionSummaryCounts {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let all_counts = export_usize_map(env, &self.all_counts)?;
        let all_times = export_usize_map(env, &self.all_times)?;
        let obj = env.new_object(
            SCAN_STATS_CLASS,
            SCAN_STATS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.iops as i64),
                JValueGen::Long(self.requests as i64),
                JValueGen::Long(self.bytes_read as i64),
                JValueGen::Long(self.indices_loaded as i64),
                JValueGen::Long(self.parts_loaded as i64),
                JValueGen::Long(self.index_comparisons as i64),
                JValueGen::Long(self.index_cache_hits() as i64),
                JValueGen::Long(self.index_cache_misses() as i64),
                JValueGen::Object(&all_counts),
                JValueGen::Object(&all_times),
            ],
        )?;
        Ok(obj)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_LanceScanner_nativeGetStats<'local>(
    mut env: JNIEnv<'local>,
    j_scanner: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_stats(&mut env, j_scanner))
}

fn inner_get_stats<'local>(
    env: &mut JNIEnv<'local>,
    j_scanner: JObject<'local>,
) -> Result<JObject<'local>> {
    let stats = {
        let scanner_guard =
            unsafe { env.get_rust_field::<_, _, BlockingScanner>(j_scanner, NATIVE_SCANNER) }?;
        scanner_guard.get_stats()
    };

    match stats {
        Some(stats) => (&stats).into_java(env),
        None => Ok(JObject::null()),
    }
}
