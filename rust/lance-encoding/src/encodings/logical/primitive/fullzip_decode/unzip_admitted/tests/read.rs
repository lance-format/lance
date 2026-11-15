// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Mutex;

use futures::{FutureExt, future::BoxFuture, poll};
use lance_core::utils::allocation::ReservedBuffer;

use super::*;

#[derive(Clone, Copy)]
enum Fault {
    None,
    Error(usize),
    Pending(usize),
}

#[derive(Default)]
struct Reads(Mutex<Vec<Range<u64>>>);

fn reader(
    source: Arc<Vec<u8>>,
    reads: Arc<Reads>,
    fault: Fault,
) -> impl FnMut(Range<u64>, ReservedBuffer) -> BoxFuture<'static, Result<ReservedBuffer>> {
    move |range, mut output| {
        let source = source.clone();
        let call = {
            let mut requests = reads.0.lock().unwrap();
            requests.push(range.clone());
            requests.len() - 1
        };
        async move {
            assert_eq!(range.end - range.start, output.capacity() as u64);
            match fault {
                Fault::Error(at) if at == call => return Err(Error::io("injected read error")),
                Fault::Pending(at) if at == call => std::future::pending::<()>().await,
                _ => {}
            }
            let bytes = source
                .get(range.start as usize..range.end as usize)
                .ok_or_else(|| Error::io("read outside source"))?;
            output.as_mut().copy_from_slice(bytes);
            Ok(output)
        }
        .boxed()
    }
}

fn source(f: &mut Fixture) -> Arc<Vec<u8>> {
    let mut bytes = vec![0; 37];
    bytes.extend_from_slice(f.zipped.as_ref());
    bytes.extend_from_slice(f.index.as_ref());
    f.scheduler.data_buf_position += 37;
    f.scheduler.rep_index.as_mut().unwrap().buf_position += 37;
    bytes.extend_from_slice(&[0; 19]);
    Arc::new(bytes)
}

// Derived from writer row boundaries and source logical values, independently of
// the read/unzip planners. Returns input, unzip and codec output payload bytes.
fn sizes(f: &Fixture, rows: Range<u64>) -> (usize, usize, usize) {
    let values = f.value_starts[rows.start as usize]..f.value_starts[rows.end as usize];
    let levels = f.level_starts[rows.start as usize]..f.level_starts[rows.end as usize];
    let level_bytes =
        levels.len() * 2 * (usize::from(f.rep.is_some()) + usize::from(f.def.is_some()));
    let valid = f.def.as_ref().map_or(values.len(), |defs| {
        defs[levels.clone()].iter().filter(|d| **d == 0).count()
    });
    let input = f.row_offsets[rows.end as usize] - f.row_offsets[rows.start as usize];
    let offsets = (values.len() + 1) * usize::from(f.scheduler.bits_per_offset / 8);
    let encoded = input
        - levels.len() * f.scheduler.details.ctrl_word_parser.bytes_per_word()
        - valid * usize::from(f.scheduler.bits_per_offset / 8);
    let decoded = f.values[values].iter().map(Vec::len).sum::<usize>();
    (
        input,
        offsets + level_bytes + encoded,
        offsets + level_bytes + decoded,
    )
}

#[rstest]
#[tokio::test]
async fn selected_reads_recycle_input_before_decode(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
    #[values("flat", "nullable", "list", "deep")] kind: &str,
    #[values(0..4, 1..3, 3..4, 4..4)] rows: Range<u64>,
) {
    let mut f = fixture(bits, fsst, kind);
    let source = source(&mut f);
    let (input, stage, decoded) = sizes(&f, rows.clone());
    let entry = f.index.len() / 5;
    let envelope = entry.max(input + stage).max(stage + decoded);
    let budget = AllocationBudget::new(envelope);
    let mut work = budget.try_reserve(envelope).unwrap();
    let reads = Arc::new(Reads::default());
    let task = f
        .scheduler
        .try_read_unzip_range_in(
            rows.clone(),
            &mut work,
            reader(source, reads.clone(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), envelope - stage);
    assert_eq!(budget.reserved(), envelope);
    let index = f.scheduler.rep_index.unwrap();
    let expected_requests = if rows.is_empty() {
        vec![]
    } else {
        vec![
            index.buf_position + rows.start * entry as u64
                ..index.buf_position + (rows.end + 1) * entry as u64,
            37 + f.row_offsets[rows.start as usize] as u64
                ..37 + f.row_offsets[rows.end as usize] as u64,
        ]
    };
    assert_eq!(*reads.0.lock().unwrap(), expected_requests);
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_eq!(work.len(), envelope - stage - decoded);
    drop(task);
    assert_eq!(budget.reserved(), envelope - stage);
    let selected = f.value_starts[rows.start as usize]..f.value_starts[rows.end as usize];
    let payload_bytes = (selected.len() + 1) * usize::from(bits / 8)
        + f.values[selected.clone()]
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    let values = f.values[selected]
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(values).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(values).to_data(),
        )
    };
    let arrow = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(arrow, expected);
    let alias = arrow.slice(0, arrow.len());
    drop((arrow, page, work));
    // Nullable levels have separate owners; final Arrow payload aliases retain
    // the exact codec allocation until the last view disappears.
    assert_eq!(budget.reserved(), payload_bytes);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn read_stage_refusal_restores_workspace(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
    #[values("index", "input", "unzip")] stage: &str,
) {
    let mut f = fixture(bits, fsst, "deep");
    let source = source(&mut f);
    let (input, unzip, _) = sizes(&f, 0..4);
    let (available, calls) = match stage {
        "index" => (f.index.len() / 5 - 1, 0),
        "input" => (input - 1, 1),
        _ => (input + unzip - 1, 2),
    };
    let budget = AllocationBudget::new(available);
    let mut work = budget.try_reserve(available).unwrap();
    for _ in 0..2 {
        let reads = Arc::new(Reads::default());
        assert!(
            f.scheduler
                .try_read_unzip_range_in(
                    0..4,
                    &mut work,
                    reader(source.clone(), reads.clone(), Fault::None),
                )
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(reads.0.lock().unwrap().len(), calls);
        assert_eq!(work.len(), available);
        assert_eq!(budget.reserved(), available);
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn read_error_and_cancellation_release_active_owner(
    #[values(0, 1)] at: usize,
    #[values(false, true)] cancel: bool,
) {
    let mut f = fixture(32, false, "deep");
    let source = source(&mut f);
    let (input, stage, _) = sizes(&f, 0..4);
    let budget = AllocationBudget::new(input + stage);
    let mut work = budget.try_reserve(input + stage).unwrap();
    let reads = Arc::new(Reads::default());
    let fault = if cancel {
        Fault::Pending(at)
    } else {
        Fault::Error(at)
    };
    let mut future = Box::pin(f.scheduler.try_read_unzip_range_in(
        0..4,
        &mut work,
        reader(source, reads.clone(), fault),
    ));
    if cancel {
        assert!(poll!(future.as_mut()).is_pending());
        assert_eq!(budget.reserved(), input + stage);
    } else {
        let result = future.as_mut().await;
        let Err(error) = result else {
            panic!("expected read error")
        };
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected read error"));
    }
    assert_eq!(reads.0.lock().unwrap().len(), at + 1);
    drop(future);
    assert_eq!(budget.reserved(), work.len());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::descending(1, 2, true)]
#[case::past_end(1, 2, false)]
#[case::first(0, 2, false)]
#[case::last(1, 4, true)]
#[tokio::test]
async fn invalid_index_is_rejected_before_payload_read(
    #[case] start: u64,
    #[case] end: u64,
    #[case] small: bool,
    #[values(false, true)] combined: bool,
) {
    let mut f = fixture(32, false, "flat");
    let mut source = source(&mut f);
    let index = f.scheduler.rep_index.unwrap();
    let width = index.bytes_per_value as usize;
    let row = if start == 0 { start } else { end };
    let value = if start == 0 {
        1
    } else if small {
        0
    } else {
        f.zipped.len() as u64 + 1
    };
    let offset = index.buf_position as usize + row as usize * width;
    Arc::make_mut(&mut source)[offset..offset + width]
        .copy_from_slice(&value.to_le_bytes()[..width]);
    let available = if combined { f.index.len() } else { width };
    let budget = AllocationBudget::new(available);
    let mut work = budget.try_reserve(available).unwrap();
    let reads = Arc::new(Reads::default());
    let result = f
        .scheduler
        .try_read_unzip_range_in(
            start..end,
            &mut work,
            reader(source, reads.clone(), Fault::None),
        )
        .await;
    let Err(error) = result else {
        panic!("expected corrupt index")
    };
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("repetition index endpoints"));
    assert_eq!(reads.0.lock().unwrap().len(), if combined { 1 } else { 2 });
    assert_eq!(work.len(), available);
}

#[tokio::test]
async fn slow_reader_does_not_block_finite_peers() {
    let mut f = fixture(32, false, "flat");
    let source = source(&mut f);
    let (input, stage, decoded) = sizes(&f, 0..4);
    let envelope = (input + stage).max(stage + decoded);
    let budget = AllocationBudget::new(2 * envelope);
    let mut slow_work = budget.try_reserve(envelope).unwrap();
    let mut slow = Box::pin(f.scheduler.try_read_unzip_range_in(
        0..4,
        &mut slow_work,
        reader(source.clone(), Arc::default(), Fault::Pending(1)),
    ));
    assert!(poll!(slow.as_mut()).is_pending());
    for _ in 0..3 {
        let mut work = budget.try_reserve(envelope).unwrap();
        assert!(budget.try_reserve(1).is_none());
        let task = f
            .scheduler
            .try_read_unzip_range_in(
                0..4,
                &mut work,
                reader(source.clone(), Arc::default(), Fault::None),
            )
            .await
            .unwrap()
            .unwrap();
        let page = task.try_decode_in(&mut work).unwrap().unwrap();
        let arrow = page
            .data
            .try_to_arrow_in(&DataType::Binary, &mut work)
            .unwrap()
            .unwrap();
        let values = f.values.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert_eq!(arrow, BinaryArray::from_vec(values).to_data());
        drop((task, arrow, page, work));
        assert_eq!(budget.reserved(), envelope);
    }
    drop(slow);
    assert_eq!(budget.reserved(), slow_work.len());
    drop(slow_work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn invalid_read_geometry_never_calls_reader(
    #[values(
        "reversed",
        "past_page",
        "index_address",
        "data_address",
        "width",
        "missing"
    )]
    kind: &str,
) {
    let mut f = fixture(32, false, "flat");
    let source = source(&mut f);
    let rows = match kind {
        "reversed" => Range { start: 3, end: 1 },
        "past_page" => 0..5,
        "index_address" => {
            f.scheduler.rep_index.as_mut().unwrap().buf_position = u64::MAX;
            0..4
        }
        "data_address" => {
            f.scheduler.data_buf_position = u64::MAX;
            0..4
        }
        "width" => {
            f.scheduler.rep_index.as_mut().unwrap().bytes_per_value = 3;
            0..4
        }
        _ => {
            f.scheduler.rep_index = None;
            0..4
        }
    };
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let reads = Arc::new(Reads::default());
    let result = f
        .scheduler
        .try_read_unzip_range_in(rows, &mut work, reader(source, reads.clone(), Fault::None))
        .await;
    let Err(error) = result else {
        panic!("expected invalid geometry")
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("FullZip"));
    assert!(reads.0.lock().unwrap().is_empty());
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn corrupt_selected_payload_recycles_exclusive_read() {
    let mut f = fixture(32, false, "flat");
    let mut source = source(&mut f);
    let (input, _, _) = sizes(&f, 0..4);
    // Corrupt a trailing value, while leaving the row-index boundaries intact.
    let offset = 37 + f.row_offsets[3];
    Arc::make_mut(&mut source)[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let budget = AllocationBudget::new(input);
    let mut work = budget.try_reserve(input).unwrap();
    let result = f
        .scheduler
        .try_read_unzip_range_in(0..4, &mut work, reader(source, Arc::default(), Fault::None))
        .await;
    let Err(error) = result else {
        panic!("expected corrupt payload before output refusal")
    };
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("item length"));
    assert_eq!(work.len(), input);
    assert_eq!(budget.reserved(), input);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

mod coalesce;
mod cpu;
mod prefix;
