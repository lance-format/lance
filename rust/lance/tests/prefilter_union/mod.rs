// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{
    RecordBatch, RecordBatchIterator, cast::AsArray, record_batch, types::UInt64Type,
};
use datafusion::{
    datasource::memory::MemorySourceConfig,
    error::{DataFusionError, Result},
    execution::TaskContext,
    physical_plan::{ExecutionPlan, coalesce_partitions::CoalescePartitionsExec, collect},
};
use lance::{
    Dataset,
    dataset::{WriteParams, builder::DatasetBuilder},
    io::exec::PrefilterUnionExec,
};
use lance_core::ROW_ID;
use lance_select::{
    IndexExprResult, RowAddrMask, RowAddrTreeMap, result::IndexExprResultWireFormat,
};
use prost::Message;
use roaring::RoaringBitmap;
use rstest::rstest;

#[derive(Clone, PartialEq, prost::Message)]
struct UnionMetadata {
    #[prost(string, tag = "1")]
    uri: String,
    #[prost(uint64, tag = "2")]
    version: u64,
    #[prost(bytes = "vec", tag = "3")]
    scope: Vec<u8>,
    #[prost(bytes = "vec", tag = "4")]
    coverage: Vec<u8>,
    #[prost(enumeration = "MaskFormat", tag = "5")]
    format: i32,
    #[prost(bool, tag = "6")]
    has_missing: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
enum MaskFormat {
    TwoMask = 0,
    ThreeVariant = 1,
}

// Test the public metadata boundary with serialized children and a fresh snapshot.
fn encode_union(union: &PrefilterUnionExec) -> Result<Vec<u8>> {
    let mut scope = Vec::new();
    union.fragment_scope().serialize_into(&mut scope)?;
    let mut coverage = Vec::new();
    union.fragments_covered().serialize_into(&mut coverage)?;
    Ok(UnionMetadata {
        uri: union.dataset().uri().to_owned(),
        version: union.dataset().version_id(),
        scope,
        coverage,
        format: match union.result_format() {
            IndexExprResultWireFormat::TwoMask => MaskFormat::TwoMask,
            IndexExprResultWireFormat::ThreeVariant => MaskFormat::ThreeVariant,
        } as i32,
        has_missing: union.children().len() == 2,
    }
    .encode_to_vec())
}

fn decode_union(
    buf: &[u8],
    dataset: Arc<Dataset>,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Result<Arc<dyn ExecutionPlan>> {
    let metadata = UnionMetadata::decode(buf)
        .map_err(|error| DataFusionError::Plan(format!("cannot decode union metadata: {error}")))?;
    let expected_inputs = if metadata.has_missing { 2 } else { 1 };
    if inputs.len() != expected_inputs {
        return Err(DataFusionError::Plan(format!(
            "union expected {expected_inputs} inputs, got {}",
            inputs.len()
        )));
    }
    if metadata.uri != dataset.uri() || metadata.version != dataset.version_id() {
        return Err(DataFusionError::Plan(format!(
            "union snapshot uri={} version={} differs from receiver uri={} version={}",
            metadata.uri,
            metadata.version,
            dataset.uri(),
            dataset.version_id()
        )));
    }
    let format = match MaskFormat::try_from(metadata.format) {
        Ok(MaskFormat::TwoMask) => IndexExprResultWireFormat::TwoMask,
        Ok(MaskFormat::ThreeVariant) => IndexExprResultWireFormat::ThreeVariant,
        Err(_) => {
            return Err(DataFusionError::Plan(format!(
                "unsupported union result_format={}",
                metadata.format
            )));
        }
    };
    Ok(Arc::new(PrefilterUnionExec::new(
        dataset,
        inputs[0].clone(),
        inputs.get(1).cloned(),
        RoaringBitmap::deserialize_from(Cursor::new(metadata.scope))?,
        RoaringBitmap::deserialize_from(Cursor::new(metadata.coverage))?,
        format,
    )))
}

fn encode_batch(batch: &RecordBatch) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut writer =
            arrow_ipc::writer::StreamWriter::try_new(&mut bytes, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }
    bytes
}

fn decode_batch(bytes: &[u8]) -> Arc<dyn ExecutionPlan> {
    let reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(bytes), None).unwrap();
    let schema = reader.schema();
    let batches = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    MemorySourceConfig::try_new_exec(&[batches], schema, None).unwrap()
}

#[rstest]
#[case::scoped(false)]
#[case::empty_scope(true)]
#[tokio::test]
async fn prefilter_union_metadata_roundtrip(
    #[values(
        IndexExprResultWireFormat::TwoMask,
        IndexExprResultWireFormat::ThreeVariant
    )]
    format: IndexExprResultWireFormat,
    #[case] empty_scope: bool,
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] has_missing: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let batch = record_batch!(("i", Int32, [0, 1, 2, 3, 4, 5, 6, 7, 8])).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            dir.path().to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let mut scanner = dataset.scan();
    scanner.with_row_id().project(&[ROW_ID]).unwrap();
    let rows = scanner.try_into_batch().await.unwrap();
    let ids = rows[ROW_ID].as_primitive::<UInt64Type>().values();
    // The scalar complement includes excluded fragments and unbounded addresses.
    // Its scope must survive encoding, rather than merely its serialized coverage.
    let scalar_batch =
        IndexExprResult::exact(RowAddrMask::from_block(RowAddrTreeMap::from_iter([ids[3]])))
            .serialize(&[0, 1, 2].into_iter().collect(), format)
            .unwrap();
    let scalar_bytes = encode_batch(&scalar_batch);
    let scalar =
        MemorySourceConfig::try_new_exec(&[vec![scalar_batch]], format.schema().clone(), None)
            .unwrap();
    let missing_bytes =
        has_missing.then(|| encode_batch(&record_batch!((ROW_ID, UInt64, [ids[7]])).unwrap()));
    let missing = missing_bytes.as_deref().map(decode_batch);
    let scope: RoaringBitmap = if empty_scope {
        RoaringBitmap::new()
    } else {
        [1].into_iter().collect()
    };
    let mut coverage = scope.clone();
    if has_missing {
        coverage.insert(2);
    }
    let union = Arc::new(PrefilterUnionExec::new(
        dataset.clone(),
        scalar,
        missing,
        scope.clone(),
        coverage.clone(),
        format,
    ));
    let bytes = encode_union(&union).unwrap();
    drop(union);

    let receiver = Arc::new(
        DatasetBuilder::from_uri(dataset.uri())
            .with_version(dataset.version_id())
            .load()
            .await
            .unwrap(),
    );
    assert!(!Arc::ptr_eq(&receiver, &dataset));
    let context = Arc::new(TaskContext::default());
    let mut children = vec![decode_batch(&scalar_bytes)];
    if let Some(bytes) = missing_bytes {
        children.push(decode_batch(&bytes));
    }
    let decoded: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
        decode_union(&bytes, receiver.clone(), &children).unwrap(),
    ));
    let child = decoded.children()[0].clone();
    let union = child.downcast_ref::<PrefilterUnionExec>().unwrap();
    assert!(Arc::ptr_eq(union.dataset(), &receiver));
    assert_eq!(union.fragment_scope(), &scope);
    assert_eq!(union.fragments_covered(), &coverage);
    assert_eq!(union.result_format(), format);
    assert_eq!(union.children().len(), if has_missing { 2 } else { 1 });

    let batches = collect(decoded, context).await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema(), format.schema().clone());
    let (result, evaluated) = IndexExprResult::deserialize(&batches[0]).unwrap();
    let mut expected = if empty_scope {
        vec![]
    } else {
        vec![ids[4], ids[5]]
    };
    if has_missing {
        expected.push(ids[7]);
    }
    assert!(result.is_exact());
    assert_eq!(evaluated, coverage);
    assert_eq!(
        result.upper,
        RowAddrMask::from_allowed(expected.into_iter().collect())
    );
}

#[cfg(feature = "substrait")]
mod scanner_components {
    use super::*;
    use arrow::compute::{SortColumn, concat_batches, lexsort_to_indices, take};
    use arrow_array::{
        Float32Array,
        types::{Float32Type, Int32Type},
    };
    use datafusion::{physical_plan::displayable, prelude::SessionContext};
    use lance::{
        index::{DatasetIndexExt, DatasetIndexInternalExt, vector::VectorIndexParams},
        io::exec::{
            ANNIvfPartitionExec, ANNIvfSubIndexExec, Planner,
            ann_proto::{
                ann_ivf_partition_exec_from_proto, ann_ivf_partition_exec_to_proto,
                ann_ivf_sub_index_exec_from_proto, ann_ivf_sub_index_exec_to_proto,
            },
            filtered_read::FilteredReadExec,
            filtered_read_proto::{filtered_read_exec_from_proto, filtered_read_exec_to_proto},
            fts::MatchQueryExec,
            scalar_index::ScalarIndexExec,
            utils::PreFilterSource,
        },
    };
    use lance_datafusion::substrait::{
        encode_substrait, parse_substrait, prune_schema_for_substrait,
    };
    use lance_datagen::{RowCount, array, gen_batch};
    use lance_index::{
        IndexType,
        scalar::{
            FullTextSearchQuery, ScalarIndexParams,
            expression::PlannerIndexExt,
            inverted::{
                query::{FtsSearchParams, MatchQuery},
                tokenizer::InvertedIndexParams,
            },
        },
    };
    use lance_linalg::distance::MetricType;
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeSet, HashSet};
    use uuid::Uuid;

    #[derive(Clone, PartialEq, prost::Message)]
    struct ScalarMetadata {
        #[prost(bytes = "vec", tag = "1")]
        substrait_expr: Vec<u8>,
        #[prost(enumeration = "MaskFormat", tag = "2")]
        format: i32,
        #[prost(bytes = "vec", optional, tag = "3")]
        scope: Option<Vec<u8>>,
        #[prost(string, repeated, tag = "4")]
        segments: Vec<String>,
        #[prost(bool, tag = "5")]
        segments_selected: bool,
    }

    #[derive(Debug, Serialize, Deserialize)]
    enum PrefilterKind {
        None,
        Rows,
        Mask,
    }

    #[derive(Serialize, Deserialize)]
    struct MatchMetadata {
        query: MatchQuery,
        params: FtsSearchParams,
        segments: Option<Vec<Uuid>>,
        prefilter: PrefilterKind,
    }

    fn encode_scalar(scalar: &ScalarIndexExec) -> Vec<u8> {
        let scope = scalar.fragment_scope().map(|scope| {
            let mut bytes = Vec::new();
            scope.serialize_into(&mut bytes).unwrap();
            bytes
        });
        let schema = Arc::new(prune_schema_for_substrait(
            &scalar.dataset().schema().into(),
        ));
        ScalarMetadata {
            substrait_expr: encode_substrait(
                scalar.expr().to_expr(),
                schema,
                &SessionContext::new().state(),
            )
            .unwrap(),
            format: match scalar.result_format() {
                IndexExprResultWireFormat::TwoMask => MaskFormat::TwoMask,
                IndexExprResultWireFormat::ThreeVariant => MaskFormat::ThreeVariant,
            } as i32,
            scope,
            segments: scalar
                .segment_uuids()
                .map(|ids| ids.iter().map(Uuid::to_string).collect())
                .unwrap_or_default(),
            segments_selected: scalar.segment_uuids().is_some(),
        }
        .encode_to_vec()
    }

    async fn decode_scalar(bytes: &[u8], dataset: Arc<Dataset>) -> Arc<dyn ExecutionPlan> {
        let metadata = ScalarMetadata::decode(bytes).unwrap();
        let schema = Arc::new(prune_schema_for_substrait(&dataset.schema().into()));
        let expr = parse_substrait(
            &metadata.substrait_expr,
            schema,
            &SessionContext::new().state(),
        )
        .await
        .unwrap();
        let info = dataset.scalar_index_info().await.unwrap();
        let expr = Planner::new(Arc::new(dataset.schema().into()))
            .create_filter_plan(expr, &info, true)
            .unwrap()
            .index_query
            .unwrap();
        let format = match MaskFormat::try_from(metadata.format).unwrap() {
            MaskFormat::TwoMask => IndexExprResultWireFormat::TwoMask,
            MaskFormat::ThreeVariant => IndexExprResultWireFormat::ThreeVariant,
        };
        let mut scalar = ScalarIndexExec::new(dataset, expr, format);
        if let Some(scope) = metadata.scope {
            scalar = scalar
                .with_fragment_scope(RoaringBitmap::deserialize_from(Cursor::new(scope)).unwrap());
        }
        if metadata.segments_selected {
            scalar = scalar.with_segment_uuids(Arc::new(
                metadata
                    .segments
                    .iter()
                    .map(|id| Uuid::parse_str(id).unwrap())
                    .collect(),
            ));
        }
        Arc::new(scalar)
    }

    enum SearchMetadata {
        Ann {
            partitions: Vec<u8>,
            subindex: Vec<u8>,
        },
        Match(Vec<u8>),
    }

    struct SearchComponents {
        union: Vec<u8>,
        scalar: Vec<u8>,
        missing: Vec<u8>,
        search: SearchMetadata,
    }

    impl SearchComponents {
        async fn encode(consumer: &Arc<dyn ExecutionPlan>) -> Self {
            let union_plan = find::<PrefilterUnionExec>(consumer).unwrap();
            let union = union_plan.downcast_ref::<PrefilterUnionExec>().unwrap();
            let children = union.children();
            assert_eq!(children.len(), 2);
            let scalar = children[0].downcast_ref::<ScalarIndexExec>().unwrap();
            let read = children[1].downcast_ref::<FilteredReadExec>().unwrap();
            assert!(read.index_input().is_none() && read.row_stream_input().is_none());
            let search = if let Some(ann) = consumer.downcast_ref::<ANNIvfSubIndexExec>() {
                assert!(matches!(
                    ann.prefilter_source(),
                    PreFilterSource::ScalarIndexQuery(_)
                ));
                let inputs = ann.children();
                assert_eq!(inputs.len(), 2);
                assert!(Arc::ptr_eq(inputs[1], &union_plan));
                let partitions = inputs[0].downcast_ref::<ANNIvfPartitionExec>().unwrap();
                SearchMetadata::Ann {
                    partitions: ann_ivf_partition_exec_to_proto(partitions)
                        .await
                        .unwrap()
                        .encode_to_vec(),
                    subindex: ann_ivf_sub_index_exec_to_proto(ann)
                        .await
                        .unwrap()
                        .encode_to_vec(),
                }
            } else {
                let fts = consumer.downcast_ref::<MatchQueryExec>().unwrap();
                let prefilter = match fts.prefilter_source() {
                    PreFilterSource::ScalarIndexQuery(input) => {
                        assert!(Arc::ptr_eq(input, &union_plan));
                        PrefilterKind::Mask
                    }
                    other => panic!("expected scalar mask prefilter, got {other:?}"),
                };
                SearchMetadata::Match(
                    serde_json::to_vec(&MatchMetadata {
                        query: fts.query().clone(),
                        params: fts.params().clone(),
                        segments: fts.explicit_segment_uuids(),
                        prefilter,
                    })
                    .unwrap(),
                )
            };
            Self {
                union: encode_union(union).unwrap(),
                scalar: encode_scalar(scalar),
                missing: filtered_read_exec_to_proto(read, &SessionContext::new().state())
                    .await
                    .unwrap()
                    .encode_to_vec(),
                search,
            }
        }

        async fn decode(&self, dataset: Arc<Dataset>) -> Arc<dyn ExecutionPlan> {
            let scalar = decode_scalar(&self.scalar, dataset.clone()).await;
            let read = filtered_read_exec_from_proto(
                lance_datafusion::pb::FilteredReadExecProto::decode(self.missing.as_slice())
                    .unwrap(),
                Some(dataset.clone()),
                None,
                &SessionContext::new().state(),
            )
            .await
            .unwrap();
            let union =
                decode_union(&self.union, dataset.clone(), &[scalar, Arc::new(read)]).unwrap();
            match &self.search {
                SearchMetadata::Ann {
                    partitions,
                    subindex,
                } => {
                    let partitions = ann_ivf_partition_exec_from_proto(
                        lance::pb::AnnIvfPartitionExecProto::decode(partitions.as_slice()).unwrap(),
                        Some(dataset.clone()),
                    )
                    .await
                    .unwrap();
                    Arc::new(
                        ann_ivf_sub_index_exec_from_proto(
                            lance::pb::AnnIvfSubIndexExecProto::decode(subindex.as_slice())
                                .unwrap(),
                            Some(dataset),
                            Arc::new(partitions),
                            Some(union),
                        )
                        .await
                        .unwrap(),
                    )
                }
                SearchMetadata::Match(bytes) => {
                    let metadata: MatchMetadata = serde_json::from_slice(bytes).unwrap();
                    assert!(matches!(metadata.prefilter, PrefilterKind::Mask));
                    let prefilter = PreFilterSource::ScalarIndexQuery(union);
                    Arc::new(if let Some(segments) = metadata.segments {
                        MatchQueryExec::new_with_segment_uuids(
                            dataset,
                            metadata.query,
                            metadata.params,
                            prefilter,
                            segments,
                        )
                        .unwrap()
                    } else {
                        MatchQueryExec::new(dataset, metadata.query, metadata.params, prefilter)
                            .unwrap()
                    })
                }
            }
        }
    }

    fn find<T: ExecutionPlan + 'static>(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        if plan.is::<T>() {
            return Some(plan.clone());
        }
        plan.children().into_iter().find_map(find::<T>)
    }

    #[rstest]
    #[case::ann(false)]
    #[case::fts(true)]
    #[tokio::test]
    async fn scanner_hybrid_components_roundtrip(
        #[case] fts: bool,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let batch = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::fill_utf8("token".to_owned()))
            .col("vector", array::rand_vec::<Float32Type>(4.into()))
            .into_batch_rows(RowCount::from(12))
            .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.slice(0, 8))], batch.schema()),
            dir.path().to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch.slice(8, 4))], batch.schema()),
                Some(WriteParams {
                    max_rows_per_file: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        if fts {
            dataset
                .create_index(
                    &["text"],
                    IndexType::Inverted,
                    None,
                    &InvertedIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        } else {
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    None,
                    &VectorIndexParams::ivf_flat(1, MetricType::L2),
                    true,
                )
                .await
                .unwrap();
        }
        let dataset = Arc::new(dataset);
        let mut scanner = dataset.scan();
        scanner
            .with_fragments(dataset.fragments()[1..].to_vec())
            .with_row_id();
        scanner.prefilter(true).filter("i >= 2 AND i < 11").unwrap();
        scanner.project(&[ROW_ID]).unwrap();
        if fts {
            scanner
                .full_text_search(FullTextSearchQuery::new("token".to_owned()))
                .unwrap();
        } else {
            scanner
                .nearest("vector", &Float32Array::from(vec![0.0; 4]), 12)
                .unwrap()
                .nprobes(1);
        }
        let plan = scanner.create_plan().await.unwrap();
        let description = displayable(plan.as_ref()).indent(true).to_string();
        let union = find::<PrefilterUnionExec>(&plan).unwrap_or_else(|| panic!("{description}"));
        let union = union.downcast_ref::<PrefilterUnionExec>().unwrap();
        assert_eq!(union.fragment_scope(), &[1].into_iter().collect());
        assert_eq!(union.fragments_covered(), &[1, 2].into_iter().collect());
        let format = union.result_format();
        let scalar_plan = find::<ScalarIndexExec>(&plan).unwrap();
        let scalar = scalar_plan.downcast_ref::<ScalarIndexExec>().unwrap();
        let scalar_scope = scalar.fragment_scope().cloned();
        let scalar_segments = scalar.segment_uuids().cloned();
        let scalar_expr = scalar.expr().clone();
        assert_eq!(scalar_scope, Some([1].into_iter().collect()));
        assert!(find::<FilteredReadExec>(&plan).is_some(), "{description}");
        assert_eq!(find::<MatchQueryExec>(&plan).is_some(), fts);
        assert_eq!(find::<ANNIvfSubIndexExec>(&plan).is_some(), !fts);
        let consumer = if fts {
            find::<MatchQueryExec>(&plan).unwrap()
        } else {
            find::<ANNIvfSubIndexExec>(&plan).unwrap()
        };
        let components = SearchComponents::encode(&consumer).await;
        let read_plan = find::<FilteredReadExec>(&consumer).unwrap();
        let read = read_plan.downcast_ref::<FilteredReadExec>().unwrap();
        assert_eq!(
            read.options()
                .fragments
                .as_ref()
                .unwrap()
                .iter()
                .map(|fragment| fragment.id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        let expected = collect(consumer, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        let receiver = Arc::new(
            DatasetBuilder::from_uri(dataset.uri())
                .with_version(dataset.version_id())
                .load()
                .await
                .unwrap(),
        );
        assert!(!Arc::ptr_eq(&receiver, &dataset));
        let decoded = components.decode(receiver.clone()).await;
        if fts {
            let search = decoded.downcast_ref::<MatchQueryExec>().unwrap();
            assert!(Arc::ptr_eq(search.dataset(), &receiver));
            assert!(matches!(
                search.prefilter_source(),
                PreFilterSource::ScalarIndexQuery(_)
            ));
        } else {
            let search = decoded.downcast_ref::<ANNIvfSubIndexExec>().unwrap();
            assert!(Arc::ptr_eq(search.dataset(), &receiver));
            assert!(matches!(
                search.prefilter_source(),
                PreFilterSource::ScalarIndexQuery(_)
            ));
            let inputs = search.children();
            assert_eq!(inputs.len(), 2);
            let partitions = inputs[0].downcast_ref::<ANNIvfPartitionExec>().unwrap();
            assert!(Arc::ptr_eq(&partitions.dataset, &receiver));
            assert!(inputs[1].is::<PrefilterUnionExec>());
        }
        let union = find::<PrefilterUnionExec>(&decoded).unwrap();
        let union = union.downcast_ref::<PrefilterUnionExec>().unwrap();
        assert!(Arc::ptr_eq(union.dataset(), &receiver));
        let children = union.children();
        assert_eq!(children.len(), 2);
        assert!(children[0].is::<ScalarIndexExec>());
        assert!(children[1].is::<FilteredReadExec>());
        assert_eq!(union.fragment_scope(), &[1].into_iter().collect());
        assert_eq!(union.fragments_covered(), &[1, 2].into_iter().collect());
        assert_eq!(union.result_format(), format);
        let scalar_plan = find::<ScalarIndexExec>(&decoded).unwrap();
        let scalar = scalar_plan.downcast_ref::<ScalarIndexExec>().unwrap();
        assert!(Arc::ptr_eq(scalar.dataset(), &receiver));
        assert_eq!(scalar.fragment_scope(), scalar_scope.as_ref());
        assert_eq!(scalar.segment_uuids(), scalar_segments.as_ref());
        assert_eq!(scalar.expr(), &scalar_expr);
        assert_eq!(scalar.result_format(), format);
        let read_plan = find::<FilteredReadExec>(&decoded).unwrap();
        let read = read_plan.downcast_ref::<FilteredReadExec>().unwrap();
        assert!(Arc::ptr_eq(read.dataset(), &receiver));
        assert_eq!(
            read.options()
                .fragments
                .as_ref()
                .unwrap()
                .iter()
                .map(|fragment| fragment.id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        let actual = collect(decoded, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        let ids = |batches: &[arrow_array::RecordBatch]| {
            batches
                .iter()
                .flat_map(|batch| {
                    batch[ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .copied()
                })
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(ids(&actual), ids(&expected));
        let mut oracle = receiver.scan();
        oracle.with_fragments(receiver.fragments()[1..].to_vec());
        oracle.use_scalar_index(false).with_row_id();
        oracle.filter("i >= 2 AND i < 11").unwrap();
        oracle.project(&[ROW_ID]).unwrap();
        let oracle = oracle.try_into_batch().await.unwrap();
        let oracle_ids = ids(&[oracle]);
        assert_eq!(ids(&actual), oracle_ids);
        // Compare every returned field (including search scores) independent of
        // the order of equal-distance or equal-score hits.
        let sorted = |batches: &[arrow_array::RecordBatch]| {
            let batch = concat_batches(&batches[0].schema(), batches).unwrap();
            let indices = lexsort_to_indices(
                &[SortColumn {
                    values: batch[ROW_ID].clone(),
                    options: None,
                }],
                None,
            )
            .unwrap();
            let columns = batch
                .columns()
                .iter()
                .map(|column| take(column.as_ref(), &indices, None).unwrap())
                .collect();
            arrow_array::RecordBatch::try_new(batch.schema(), columns).unwrap()
        };
        assert_eq!(sorted(&actual), sorted(&expected));
        assert_eq!(
            actual.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            7
        );
        assert_eq!(ids(&actual).len(), 7);
        if !fts {
            assert_eq!(
                ids(&actual).intersection(&oracle_ids).count() as f64 / oracle_ids.len() as f64,
                1.0
            );
        }

        // Also exercise explicit UUID selection on the real scalar index. The
        // ordinary scanner query selects all committed scalar segments.
        let scalar_uuid = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|index| index.name == "i_idx")
            .unwrap()
            .uuid;
        // An explicitly empty selection must remain distinct from no selection.
        // Also exercise an absent loading scope independently of scanner defaults.
        for segments in [None, Some(Arc::new(HashSet::new()))] {
            let mut scalar = ScalarIndexExec::new(dataset.clone(), scalar_expr.clone(), format);
            if let Some(segments) = segments {
                scalar = scalar.with_segment_uuids(segments);
            }
            let bytes = encode_scalar(&scalar);
            let decoded = decode_scalar(&bytes, receiver.clone()).await;
            let decoded = decoded.downcast_ref::<ScalarIndexExec>().unwrap();
            assert!(Arc::ptr_eq(decoded.dataset(), &receiver));
            assert_eq!(decoded.expr(), scalar.expr());
            assert_eq!(decoded.fragment_scope(), None);
            assert_eq!(decoded.segment_uuids(), scalar.segment_uuids());
            assert_eq!(decoded.result_format(), format);
        }
        let scalar = ScalarIndexExec::new(dataset.clone(), scalar_expr.clone(), format)
            .with_fragment_scope(scalar_scope.unwrap())
            .with_segment_uuids(Arc::new(HashSet::from([scalar_uuid])));
        let scoped: Arc<dyn ExecutionPlan> = Arc::new(PrefilterUnionExec::new(
            dataset.clone(),
            Arc::new(scalar),
            None,
            [1].into_iter().collect(),
            [1].into_iter().collect(),
            format,
        ));
        let union = scoped.downcast_ref::<PrefilterUnionExec>().unwrap();
        let union_bytes = encode_union(union).unwrap();
        let scalar_bytes = encode_scalar(
            union.children()[0]
                .downcast_ref::<ScalarIndexExec>()
                .unwrap(),
        );
        let scalar = decode_scalar(&scalar_bytes, receiver.clone()).await;
        let decoded = decode_union(&union_bytes, receiver.clone(), &[scalar]).unwrap();
        let union = decoded.downcast_ref::<PrefilterUnionExec>().unwrap();
        assert!(Arc::ptr_eq(union.dataset(), &receiver));
        assert_eq!(union.fragment_scope(), &[1].into_iter().collect());
        assert_eq!(union.fragments_covered(), &[1].into_iter().collect());
        assert_eq!(union.result_format(), format);
        assert_eq!(union.children().len(), 1);
        let scalar_plan = find::<ScalarIndexExec>(&decoded).unwrap();
        let scalar = scalar_plan.downcast_ref::<ScalarIndexExec>().unwrap();
        assert!(Arc::ptr_eq(scalar.dataset(), &receiver));
        assert_eq!(scalar.expr(), &scalar_expr);
        assert_eq!(scalar.result_format(), format);
        assert_eq!(scalar.fragment_scope(), Some(&[1].into_iter().collect()));
        assert_eq!(
            scalar.segment_uuids().unwrap().as_ref(),
            &HashSet::from([scalar_uuid])
        );
        let batches = collect(decoded, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        let (result, coverage) = IndexExprResult::deserialize(&batches[0]).unwrap();
        assert!(result.is_exact());
        assert_eq!(coverage, [1].into_iter().collect());
        let mut oracle = dataset.scan();
        oracle.with_fragments(vec![dataset.fragments()[1].clone()]);
        oracle.use_scalar_index(false).with_row_id();
        oracle.filter("i >= 2 AND i < 11").unwrap();
        oracle.project(&[ROW_ID]).unwrap();
        let oracle = oracle.try_into_batch().await.unwrap();
        assert_eq!(
            result.upper,
            RowAddrMask::from_allowed(ids(&[oracle]).into_iter().collect())
        );
    }
}
