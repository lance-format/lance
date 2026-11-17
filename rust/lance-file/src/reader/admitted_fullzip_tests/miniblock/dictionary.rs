// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use lance_encoding::constants::{COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY};
use lance_encoding::format::metadata::compressive_encoding::Compression;

#[rstest]
#[tokio::test]
async fn actual_dictionary_scan_take_and_concurrent_progress(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] nested: bool,
) {
    let values = (0..1024)
        .map(|i| {
            if i % 17 == 3 {
                None
            } else {
                Some(format!(
                    "dictionary-value-{:03}-{}",
                    i % 31,
                    "value".repeat(i % 31 + 1)
                ))
            }
        })
        .collect::<Vec<_>>();
    let leaf = Arc::new(LargeStringArray::from(values)) as ArrayRef;
    let expected: ArrayRef = if nested {
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::LargeUtf8, true)),
            OffsetBuffer::new((0..=512).map(|i| i * 2).collect::<Vec<i32>>().into()),
            leaf,
            Some(NullBuffer::from(
                (0..512).map(|i| i % 19 != 3).collect::<Vec<_>>(),
            )),
        ))
    } else {
        leaf
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", expected.data_type().clone(), true).with_metadata(HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
            ),
            (COMPRESSION_META_KEY.to_string(), "none".to_string()),
            (DICT_DIVISOR_META_KEY.to_string(), "1".to_string()),
        ])),
    ]));
    let rows = expected.len();
    let batches = (0..2)
        .map(|i| {
            RecordBatch::try_new(schema.clone(), vec![expected.slice(i * rows / 2, rows / 2)])
                .unwrap()
        })
        .collect();
    let (_fs, reader) = fixture_batches(version, lite, batches, 1).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    assert_eq!(pages.len(), 2);
    for page in pages.iter() {
        let layout = match &page.encoding {
            PageEncoding::Structural(layout) => layout.clone(),
            PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
            _ => panic!("dense writer fixture"),
        };
        let Some(Layout::MiniBlockLayout(layout)) = layout.layout else {
            panic!("stored MiniBlock required");
        };
        assert!(matches!(
            layout
                .dictionary
                .as_ref()
                .and_then(|d| d.compression.as_ref()),
            Some(Compression::Variable(_))
        ));
        assert!(layout.num_dictionary_items > 1);
        assert_eq!(layout.repetition_index_depth, u32::from(nested));
        assert_eq!(page.buffer_offsets_and_sizes.len(), 3 + usize::from(nested));
    }
    const WORKSPACE: usize = 128 * 1024;
    let budget = AllocationBudget::new(WORKSPACE);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..rows as u64, budget.clone(), WORKSPACE, 37)
        .await
        .unwrap();
    let mut position = 0;
    loop {
        match scan.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, position);
                assert_eq!(
                    batch.column(0).to_data(),
                    expected
                        .slice(position as usize, batch.num_rows())
                        .to_data()
                );
                position = rows.end;
                assert!(budget.reserved() <= WORKSPACE);
                drop(batch);
                assert_eq!(budget.reserved(), 0);
            }
            FullZipScanOutcome::Finished => break,
            FullZipScanOutcome::NeedsCapacity => panic!("fixture prefix must fit"),
        }
    }
    assert_eq!(position, rows as u64);
    let indices = [0, 3, 17, rows as u32 / 2, rows as u32 - 1];
    let want = arrow_select::take::take(
        expected.as_ref(),
        &UInt32Array::from(indices.to_vec()),
        None,
    )
    .unwrap();
    let mut work = budget.try_reserve(WORKSPACE).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let batch = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(batch.column(0).to_data(), want.to_data());
    let alias = batch.column(0).to_data().buffers()[0].slice_with_length(0, 1);
    drop((batch, work));
    assert!(budget.reserved() > 1);
    drop(alias);
    assert_eq!(budget.reserved(), 0);

    // Async dictionary reads occur between pure CPU stages. With one worker,
    // neither request may park that worker while holding its original credit.
    let budget = AllocationBudget::new(2 * WORKSPACE);
    let mut first = reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), WORKSPACE, 1)
        .await
        .unwrap();
    let mut second = reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), WORKSPACE, 1)
        .await
        .unwrap();
    let (a, b) = tokio::join!(first.try_next(), second.try_next());
    for outcome in [a, b] {
        let FullZipScanOutcome::Batch { batch, .. } = outcome.unwrap() else {
            panic!("request must make progress")
        };
        assert_eq!(batch.column(0).to_data(), expected.slice(0, 1).to_data());
    }
    assert_eq!(budget.reserved(), 0);
}
