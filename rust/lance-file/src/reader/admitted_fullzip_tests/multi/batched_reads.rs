// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Fault {
    None,
    Error,
    Pending,
    CorruptIndex,
    Reordered,
}

#[derive(Debug)]
struct BatchIo {
    inner: Arc<dyn EncodingsIo>,
    groups: Mutex<Vec<Vec<Range<u64>>>>,
    singles: AtomicUsize,
    fault: Fault,
}

impl EncodingsIo for BatchIo {
    fn submit_request(
        &self,
        _: Vec<Range<u64>>,
        _: u64,
    ) -> BoxFuture<'static, Result<Vec<bytes::Bytes>>> {
        async { Err(Error::io("unexpected unadmitted payload read")) }.boxed()
    }

    fn submit_single_into(
        &self,
        range: Range<u64>,
        priority: u64,
        output: ReservedBuffer,
    ) -> BoxFuture<'static, Result<ReservedBuffer>> {
        self.singles.fetch_add(1, Ordering::SeqCst);
        self.inner.submit_single_into(range, priority, output)
    }

    fn submit_many_into(
        &self,
        requests: Vec<(Range<u64>, ReservedBuffer)>,
        priority: u64,
    ) -> BoxFuture<'_, Result<Vec<ReservedBuffer>>> {
        let phase = {
            let mut groups = self.groups.lock().unwrap();
            groups.push(requests.iter().map(|(range, _)| range.clone()).collect());
            groups.len()
        };
        async move {
            if phase == 2 {
                match self.fault {
                    Fault::Error => return Err(Error::io("injected batch data failure")),
                    Fault::Pending => std::future::pending::<()>().await,
                    _ => {}
                }
            }
            let mut outputs = self.inner.submit_many_into(requests, priority).await?;
            if phase == 1 && self.fault == Fault::CorruptIndex {
                outputs[1].as_mut().fill(0);
            }
            if phase == 2 && self.fault == Fault::Reordered {
                outputs.reverse();
            }
            Ok(outputs)
        }
        .boxed()
    }
}

fn instrument(reader: &FileReader, fault: Fault) -> (FileReader, Arc<BatchIo>) {
    let io = Arc::new(BatchIo {
        inner: reader.core.scheduler.clone(),
        groups: Mutex::new(Vec::new()),
        singles: AtomicUsize::new(0),
        fault,
    });
    (reader.with_scheduler(io.clone()), io)
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn real_columns_share_two_owned_read_phases(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] nullable_labels: bool,
    #[values(false, true)] large_list: bool,
) {
    let mut expected = batches().remove(0);
    if large_list {
        let frames = expected
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let DataType::List(field) = frames.data_type() else {
            unreachable!()
        };
        let frames: ArrayRef = Arc::new(arrow_array::LargeListArray::new(
            field.clone(),
            OffsetBuffer::new(
                frames
                    .value_offsets()
                    .iter()
                    .map(|v| i64::from(*v))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            frames.values().clone(),
            frames.nulls().cloned(),
        ));
        let mut fields = expected.schema().fields().to_vec();
        fields[1] = Arc::new(
            fields[1]
                .as_ref()
                .clone()
                .with_data_type(frames.data_type().clone()),
        );
        expected = RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            vec![expected.column(0).clone(), frames],
        )
        .unwrap();
    }
    if nullable_labels {
        let mut fields = expected.schema().fields().to_vec();
        fields[0] = Arc::new(fields[0].as_ref().clone().with_nullable(true));
        let labels: ArrayRef = Arc::new(StringArray::from_iter((0..expected.num_rows()).map(
            |row| {
                if row == 1 {
                    None
                } else {
                    Some(format!("label {row} α"))
                }
            },
        )));
        expected = RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            vec![labels, expected.column(1).clone()],
        )
        .unwrap();
    }
    let (_fs, reader) = fixture_batches(version, lite, vec![expected.clone()], 1).await;
    let (reader, io) = instrument(&reader, Fault::None);
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[1, 2, 3], &mut work)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        io.groups
            .lock()
            .unwrap()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        [2, 2]
    );
    assert_eq!(io.singles.load(Ordering::SeqCst), 0);
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_batch(&output, &(1..4), &expected);
    let alias = output.column(1).to_data().child_data()[0].buffers()[1].slice_with_length(0, 1);
    drop((output, work));
    assert!(budget.reserved() > 0);
    assert!(budget.try_reserve(65536).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[4, 5], &mut work)
        .await
        .unwrap()
        .unwrap();
    assert_batch(
        &input.decode_in(&mut work).unwrap().unwrap(),
        &(4..6),
        &expected,
    );
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::read_error(Fault::Error)]
#[case::bad_index(Fault::CorruptIndex)]
#[case::wrong_order(Fault::Reordered)]
#[tokio::test]
async fn failed_batch_releases_compound_attempt(#[case] fault: Fault) {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let (reader, io) = instrument(&reader, fault);
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let error = reader
        .prepare_admitted_fullzip_take_in(&[1, 2, 3], &mut work)
        .await
        .err()
        .unwrap();
    match fault {
        Fault::Error => {
            assert!(matches!(error, Error::IO { .. }));
            assert!(error.to_string().contains("injected batch data failure"));
        }
        Fault::CorruptIndex => {
            assert!(matches!(error, Error::CorruptFile { .. }));
            assert!(
                error
                    .to_string()
                    .contains("invalid repetition index endpoints")
            );
            assert_eq!(io.groups.lock().unwrap().len(), 1);
        }
        Fault::Reordered => {
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(
                error
                    .to_string()
                    .contains("allocations unchanged and in order")
            );
        }
        _ => unreachable!(),
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cancelled_payload_phase_releases_original_workspace() {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let (reader, io) = instrument(&reader, Fault::Pending);
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let mut pending = Box::pin(reader.prepare_admitted_fullzip_take_in(&[1, 2, 3], &mut work));
    tokio::select! {
        _ = &mut pending => panic!("data phase should remain pending"),
        _ = async { while io.groups.lock().unwrap().len() < 2 { tokio::task::yield_now().await; } } => {},
        _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => panic!("batch did not reach payload phase"),
    }
    assert_eq!(budget.reserved(), 65536);
    drop(pending);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

async fn prepare_sequential_row(reader: &FileReader, bytes: usize) -> Option<RecordBatch> {
    let budget = AllocationBudget::new(bytes);
    let mut work = budget.try_reserve(bytes).unwrap();
    let schema = Arc::new(Schema::from(reader.schema().as_ref()));
    let mut inputs = Vec::new();
    for (column, field) in reader.metadata().column_infos.iter().zip(schema.fields()) {
        let page = &column.page_infos[0];
        let scheduler = Arc::new(
            FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap(),
        );
        inputs.push(
            scheduler
                .prepare_array_range_in(
                    0..1,
                    field.data_type().clone(),
                    &mut work,
                    |range, output| {
                        reader
                            .core
                            .scheduler
                            .submit_single_into(range, page.priority, output)
                    },
                )
                .await
                .unwrap()?,
        );
    }
    let arrays = inputs
        .into_iter()
        .map(|input| input.decode_in(&mut work).unwrap())
        .collect::<Option<Vec<_>>>()?;
    Some(RecordBatch::try_new(schema, arrays).unwrap())
}

#[tokio::test]
async fn batching_preserves_smallest_sequential_workspace() {
    let expected = batches().remove(0);
    let (_fs, reader) =
        fixture_batches(ConcreteFileVersion::V2_3, false, vec![expected.clone()], 1).await;
    let (mut low, mut high) = (0, 8192);
    assert!(prepare_sequential_row(&reader, high).await.is_some());
    while high - low > 1 {
        let middle = (low + high) / 2;
        if prepare_sequential_row(&reader, middle).await.is_some() {
            high = middle;
        } else {
            low = middle;
        }
    }
    let budget = AllocationBudget::new(high);
    let mut work = budget.try_reserve(high).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0], &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_batch(&output, &(0..1), &expected);
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}
