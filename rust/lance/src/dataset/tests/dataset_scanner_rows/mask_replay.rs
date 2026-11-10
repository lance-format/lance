// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use lance_datafusion::exec::execute_plan;

#[rstest]
#[case::addresses(false, false)]
#[case::stable_ids(true, false)]
#[case::empty_addresses(false, true)]
#[case::empty_stable_ids(true, true)]
#[tokio::test]
async fn native_mask_plan_replays(#[case] stable: bool, #[case] empty: bool) {
    let mut dataset = gen_batch()
        .col("idx", array::step::<Int32Type>())
        .col("payload", array::step_custom::<Int32Type>(100, 1))
        .into_ram_dataset_with_params(
            FragmentCount::from(3),
            FragmentRowCount::from(2),
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let rows = dataset.scan().with_row_id().try_into_batch().await.unwrap();
    let ids = rows
        .column_by_name(ROW_ID)
        .unwrap()
        .as_primitive::<UInt64Type>();
    let candidates = if empty {
        vec![ids.value(0), ids.value(1)]
    } else {
        vec![ids.value(0), ids.value(1), ids.value(2), ids.value(4)]
    };
    let selection = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([
        ids.value(1),
        ids.value(2),
        ids.value(4),
    ]));
    dataset.delete("idx = 1").await.unwrap();
    let fragments = dataset.get_fragments();
    let mut scanner = dataset.scan();
    scanner.project(&["idx", "payload"]).unwrap();
    scanner.with_fragments(vec![
        fragments[0].metadata().clone(),
        fragments[2].metadata().clone(),
    ]);
    scanner.with_row_id_prefilter(selection);
    scanner
        .filter(&format!(
            "_rowid IN ({})",
            candidates
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ))
        .unwrap();
    let plan = scanner.create_plan().await.unwrap();
    for attempt in 0..2 {
        let current = reset_plan_states(plan.clone()).unwrap();
        let batches = execute_plan(current, Default::default())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_or_else(|error| panic!("literal mask replay {attempt}: {error}"));
        let batch = concat_batches(&plan.schema(), &batches).unwrap();
        let idx = batch
            .column_by_name("idx")
            .unwrap()
            .as_primitive::<Int32Type>();
        let payload = batch
            .column_by_name("payload")
            .unwrap()
            .as_primitive::<Int32Type>();
        assert_eq!(
            idx.values().as_ref(),
            if empty { &[][..] } else { &[4][..] }
        );
        assert_eq!(
            payload.values().as_ref(),
            if empty { &[][..] } else { &[104][..] }
        );
    }
}
