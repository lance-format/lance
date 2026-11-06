// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::scalar::inverted::load_segments;
use crate::io::exec::PreFilterSource;
use crate::io::exec::fts::CrossColumnCompoundQueryExec;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use lance_datafusion::exec::execute_plan;
use lance_index::scalar::inverted::query::FtsSearchParams;
use lance_select::{RowAddrMask, RowAddrTreeMap};

fn reconstruct(exec: &CrossColumnCompoundQueryExec) -> CrossColumnCompoundQueryExec {
    let columns = exec
        .column_segments()
        .map(|(column, segments)| (column.to_owned(), segments.to_vec()))
        .collect();
    CrossColumnCompoundQueryExec::new_with_segments(
        exec.dataset().clone(),
        exec.query().clone(),
        exec.params().clone(),
        exec.prefilter_source().clone(),
        columns,
    )
    .unwrap()
    .with_external_mask(
        exec.external_mask()
            .map(|mask| Arc::new(mask.as_ref().clone())),
    )
}

async fn execute(exec: Arc<dyn ExecutionPlan>) -> datafusion::error::Result<Vec<(u64, f32)>> {
    let batches: Vec<RecordBatch> = execute_plan(exec, Default::default())?
        .try_collect()
        .await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            let ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            let scores = batch[SCORE_COL].as_primitive::<Float32Type>();
            (0..batch.num_rows())
                .map(|row| (ids.value(row), scores.value(row)))
                .collect::<Vec<_>>()
        })
        .collect())
}

#[rstest]
#[tokio::test]
async fn composition_cross_column_state(
    #[values(false, true)] different_segment_counts: bool,
    #[values(false, true)] is_boost: bool,
) {
    let mut dataset = write_cross_column_compound_dataset().await;
    let fragments = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<Vec<_>>();
    create_fragmented_fts_index_with_groups(
        &mut dataset,
        "title",
        true,
        vec![vec![fragments[0]], vec![fragments[1]]],
    )
    .await;
    let body_groups = if different_segment_counts {
        vec![fragments.clone()]
    } else {
        vec![vec![fragments[1]], vec![fragments[0]]]
    };
    create_fragmented_fts_index_with_groups(&mut dataset, "body", true, body_groups).await;
    let title = load_segments(&dataset, "title", DocumentGranularity::Row)
        .await
        .unwrap()
        .unwrap();
    let body = load_segments(&dataset, "body", DocumentGranularity::Row)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(title.len(), 2);
    assert_eq!(body.len(), if different_segment_counts { 1 } else { 2 });
    let positive: FtsQuery = BooleanQuery::new([
        (Occur::Should, compound_match_query("alpha", "title", 2.0)),
        (Occur::Should, compound_match_query("gamma", "body", 3.0)),
    ])
    .into();
    let query = if is_boost {
        BoostQuery::new(
            positive,
            compound_match_query("optional", "body", 1.0),
            Some(0.5),
        )
        .into()
    } else {
        positive
    };
    let ids = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids: HashMap<_, _> = (0..ids.num_rows())
        .map(|row| {
            (
                ids["id"].as_primitive::<Int32Type>().value(row),
                ids[ROW_ID].as_primitive::<UInt64Type>().value(row),
            )
        })
        .collect();
    let row_id = |id: i32| row_ids[&id];
    let candidates = arrow_array::record_batch!((
        ROW_ID,
        UInt64,
        (0..10)
            .filter(|id| *id != 1)
            .map(row_id)
            .collect::<Vec<_>>()
    ))
    .unwrap();
    let source =
        MemorySourceConfig::try_new_exec(&[vec![candidates.clone()]], candidates.schema(), None)
            .unwrap();
    let columns = vec![("body".to_owned(), body), ("title".to_owned(), title)];
    let expected_selection = columns
        .iter()
        .map(|(column, segments)| {
            (
                column.clone(),
                segments
                    .iter()
                    .map(|segment| segment.uuid)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let mut oracle = independent_compound_fts_oracle(&dataset, &query).await;
    assert!(
        oracle.remove(&row_id(1)).is_some(),
        "prefilter must exclude a real candidate"
    );
    assert!(
        oracle.remove(&row_id(4)).is_some(),
        "external mask must exclude another candidate"
    );
    let mut expected = sorted_compound_fts_oracle(oracle);
    assert!(expected.len() > 3);
    expected.truncate(3);

    let original = CrossColumnCompoundQueryExec::new_with_segments(
        Arc::new(dataset),
        query,
        FtsSearchParams::default().with_limit(Some(3)),
        PreFilterSource::FilteredRowIds(source),
        columns,
    )
    .unwrap()
    .with_external_mask(Some(Arc::new(RowAddrMask::from_block(
        RowAddrTreeMap::from_iter([row_id(4)]),
    ))));
    let restored = reconstruct(&original);
    let actual_selection = restored
        .column_segments()
        .map(|(column, segments)| {
            (
                column.to_owned(),
                segments
                    .iter()
                    .map(|segment| segment.uuid)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual_selection, expected_selection);
    let actual = execute(Arc::new(restored)).await.unwrap();
    assert_eq!(actual, execute(Arc::new(original)).await.unwrap());
    assert_scored_rows_close("reconstructed cross-column state", &actual, &expected);
}

#[tokio::test]
async fn composition_cross_column_keeps_granularity_validation() {
    let mut title = GenericListBuilder::<i32, _>::new(GenericStringBuilder::<i32>::new());
    title.values().append_value("alpha");
    title.append(true);
    title.append(false);
    let batch = RecordBatch::try_from_iter([
        ("title", Arc::new(title.finish()) as ArrayRef),
        (
            "body",
            Arc::new(StringArray::from(vec!["gamma", "gamma"])) as ArrayRef,
        ),
    ])
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    for (column, granularity) in [
        ("title", DocumentGranularity::ListElement),
        ("body", DocumentGranularity::Row),
    ] {
        dataset
            .create_index(
                &[column],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default().document_granularity(granularity),
                true,
            )
            .await
            .unwrap();
    }
    let query: FtsQuery = BooleanQuery::new([
        (Occur::Must, compound_match_query("alpha", "title", 1.0)),
        (Occur::Must, compound_match_query("gamma", "body", 1.0)),
    ])
    .into();
    let columns = vec![
        (
            "title".into(),
            load_segments(&dataset, "title", DocumentGranularity::ListElement)
                .await
                .unwrap()
                .unwrap(),
        ),
        (
            "body".into(),
            load_segments(&dataset, "body", DocumentGranularity::Row)
                .await
                .unwrap()
                .unwrap(),
        ),
    ];
    let original = CrossColumnCompoundQueryExec::new_with_segments(
        Arc::new(dataset),
        query,
        FtsSearchParams::default().with_limit(Some(2)),
        PreFilterSource::None,
        columns,
    )
    .unwrap();
    let error = execute(Arc::new(reconstruct(&original))).await.unwrap_err();
    assert!(
        matches!(error, datafusion::error::DataFusionError::Execution(_)),
        "{error}"
    );
    assert!(
        error.to_string().contains("ListElement") && error.to_string().contains("Row"),
        "{error}"
    );
}
