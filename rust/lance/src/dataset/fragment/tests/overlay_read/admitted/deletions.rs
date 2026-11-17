// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use lance_core::utils::deletion::DeletionVector;

#[rstest]
#[tokio::test]
async fn deletion_filter_matches_real_fragment_and_overlay_take(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] nested: bool,
) {
    let mut dataset = write_payload_dataset(version, nested, false).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("delete-old-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([0, 2, 4])),
            vec![payload("old", 3, nested, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("delete-new-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([2, 3, 4])),
            vec![payload("new", 3, nested, 0)],
            version,
        )
        .await;
    }
    // Remove base-only, old-covered and newest-covered rows in both fragments.
    dataset
        .delete("untouched IN ('other0', 'other3', 'other5', 'other6', 'other9', 'other11')")
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, dataset.schema(), lite).await;
        assert_eq!(reader.deletion_vec.as_ref().unwrap().len(), 3);
        for selected in [
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 2, 3, 4],
            vec![1, 2, 4],
            vec![0, 3, 5],
            vec![],
        ] {
            let expected = reader.take_as_batch(&selected, None).await.unwrap();
            let survivor_count = selected
                .iter()
                .filter(|row| ![0, 3, 5].contains(row))
                .count();
            assert_eq!(expected.num_rows(), survivor_count);
            let indices = UInt32Array::from(selected.clone());
            let budget = AllocationBudget::new(65536);
            if survivor_count > 0 && survivor_count < selected.len() {
                // The reduced selection itself must fit before any payload work.
                let short = survivor_count * size_of::<u32>() - 1;
                assert!(
                    reader
                        .try_take_admitted_fullzip(&indices, budget.try_reserve(short).unwrap())
                        .await
                        .unwrap()
                        .is_none()
                );
                assert_eq!(budget.reserved(), 0);
            }
            if survivor_count == 0 {
                // No source payload, but the final Arrow empty arrays still own
                // LargeBinary/Utf8 offsets and optionally a List offset.
                let exact = if nested { 16 } else { 12 };
                assert!(
                    reader
                        .try_take_admitted_fullzip(&indices, budget.try_reserve(exact - 1).unwrap())
                        .await
                        .unwrap()
                        .is_none()
                );
                assert_eq!(budget.reserved(), 0);
                let output = reader
                    .try_take_admitted_fullzip(&indices, budget.try_reserve(exact).unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(output, expected);
                assert_eq!(budget.reserved(), exact);
                drop(output);
                assert_eq!(budget.reserved(), 0);
            }
            let output = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(output, expected);
            let alias = output["payload"].to_data().buffers()[0].slice_with_length(0, 1);
            drop(output);
            assert!(budget.reserved() > 1);
            assert!(budget.try_reserve(65536).is_none());
            drop(alias);
            assert_eq!(budget.reserved(), 0);
            let retried = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(retried, expected);
            drop(retried);
            assert_eq!(budget.reserved(), 0);
        }
        let indices = UInt32Array::from(vec![0, 1, 2, 3, 4, 5]);
        let expected = reader.take_as_batch(indices.values(), None).await.unwrap();
        let budget = AllocationBudget::new(2 * 65536);
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(
                reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
            )
        })
        .await
        .expect("independent deleted-row takes must progress with one CPU worker");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, expected);
        assert_eq!(right, expected);
        drop((left, right));
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn deletion_snapshot_and_ignore_deletions_keep_physical_row_identity() {
    let mut dataset = write_payload_dataset(LanceFileVersion::V2_3, false, false).await;
    dataset.delete("untouched = 'other1'").await.unwrap();
    let first = open(&dataset, 0, dataset.schema(), false).await;
    let selected = UInt32Array::from(vec![0, 1, 2, 3]);
    let first_expected = first.take_as_batch(selected.values(), None).await.unwrap();
    dataset.delete("untouched = 'other2'").await.unwrap();
    let second = open(&dataset, 0, dataset.schema(), false).await;
    let second_expected = second.take_as_batch(selected.values(), None).await.unwrap();
    let ignored = dataset
        .get_fragment(0)
        .unwrap()
        .open(
            dataset.schema(),
            FragReadConfig::default().with_ignore_deletions(true),
        )
        .await
        .unwrap();
    let ignored_expected = ignored
        .take_as_batch(selected.values(), None)
        .await
        .unwrap();
    assert_eq!(
        (
            first_expected.num_rows(),
            second_expected.num_rows(),
            ignored_expected.num_rows()
        ),
        (3, 2, 4)
    );
    let budget = AllocationBudget::new(65536);
    for (reader, expected) in [
        (&first, first_expected),
        (&second, second_expected),
        (&ignored, ignored_expected),
    ] {
        let actual = reader
            .try_take_admitted_fullzip(&selected, budget.try_reserve(65536).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        drop(actual);
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn deletion_pruning_does_not_hide_invalid_selection_or_invalid_markers() {
    let dataset = write_payload_dataset(LanceFileVersion::V2_3, false, false).await;
    let mut reader = open(&dataset, 0, dataset.schema(), false).await;
    // Include the invalid offset in the set so pruning before validation would
    // accidentally hide it. No stored metadata or ordinary reader is altered.
    reader.deletion_vec = Some(Arc::new(DeletionVector::from_iter([1, 6])));
    let budget = AllocationBudget::new(65536);
    for selected in [
        UInt32Array::from(vec![1, 1]),
        UInt32Array::from(vec![6]),
        UInt32Array::from(vec![Some(1), None]),
    ] {
        let error = reader
            .try_take_admitted_fullzip(&selected, budget.try_reserve(65536).unwrap())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("sorted unique"));
        assert_eq!(budget.reserved(), 0);
    }
    reader.with_make_deletions_null();
    let error = reader
        .try_take_admitted_fullzip(
            &UInt32Array::from(vec![1]),
            budget.try_reserve(65536).unwrap(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("markers require"));
    assert_eq!(budget.reserved(), 0);
}
