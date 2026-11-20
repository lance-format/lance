// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Two-phase reads for independent small FullZip columns, without read-ahead.

use super::*;

struct Plan {
    scheduler: Arc<FullZipScheduler>,
    rows: Range<u64>,
    data_type: DataType,
    index: Range<u64>,
    width: usize,
}

async fn fill<F, Fut>(
    read: &mut F,
    requests: Vec<(Range<u64>, ReservedBuffer)>,
) -> Result<Vec<ReservedBuffer>>
where
    F: FnMut(Vec<(Range<u64>, ReservedBuffer)>) -> Fut,
    Fut: Future<Output = Result<Vec<ReservedBuffer>>>,
{
    let identities: Vec<_> = requests
        .iter()
        .map(|(_, output)| (output.as_ref().as_ptr() as usize, output.capacity()))
        .collect();
    let outputs = read(requests).await?;
    if outputs.len() != identities.len()
        || outputs
            .iter()
            .zip(identities)
            .any(|(output, (pointer, len))| {
                output.as_ref().as_ptr() as usize != pointer || output.capacity() != len
            })
    {
        return Err(Error::invalid_input(
            "FullZip batch reader must return supplied allocations unchanged and in order",
        ));
    }
    Ok(outputs)
}

pub async fn prepare<F, Fut>(
    requests: Vec<(Arc<FullZipScheduler>, Range<u64>, DataType)>,
    workspace: &mut Reservation,
    mut read: F,
) -> Result<Option<Vec<FullZipArrayInput>>>
where
    F: FnMut(Vec<(Range<u64>, ReservedBuffer)>) -> Fut,
    Fut: Future<Output = Result<Vec<ReservedBuffer>>>,
{
    let mut plans = Vec::with_capacity(requests.len());
    let mut index_bytes = 0usize;
    for (scheduler, rows, data_type) in requests {
        Shape::new(&scheduler, &data_type)?;
        scheduler.plan_unzip_range(&[], 0, 0..0)?;
        scheduler
            .data_buf_position
            .checked_add(scheduler.data_buf_size)
            .ok_or_else(|| Error::invalid_input("FullZip data address overflows"))?;
        if rows.start >= rows.end || rows.end > scheduler.rows_in_page {
            return Err(Error::invalid_input(
                "FullZip batch requires nonempty page-local ranges",
            ));
        }
        // The single-column path can already read a whole small page and its
        // adjacent index in one IOP. Do not replace it with two batch phases.
        if rows == (0..scheduler.rows_in_page) {
            return Ok(None);
        }
        let index = scheduler
            .rep_index
            .ok_or_else(|| Error::invalid_input("FullZip batch requires a repetition index"))?;
        if !matches!(index.bytes_per_value, 1 | 2 | 4 | 8) {
            return Err(Error::invalid_input(
                "invalid FullZip repetition index width",
            ));
        }
        let address = |row: u64| {
            row.checked_mul(index.bytes_per_value)
                .and_then(|n| index.buf_position.checked_add(n))
                .ok_or_else(|| Error::invalid_input("FullZip repetition index address overflows"))
        };
        let start = address(rows.start)?;
        let end = address(rows.end)?
            .checked_add(index.bytes_per_value)
            .ok_or_else(|| Error::invalid_input("FullZip repetition index address overflows"))?;
        if end - start > 4096 {
            return Ok(None);
        }
        index_bytes = index_bytes
            .checked_add((end - start) as usize)
            .ok_or_else(|| Error::invalid_input("FullZip batch index size overflows"))?;
        plans.push(Plan {
            scheduler,
            rows,
            data_type,
            index: start..end,
            width: index.bytes_per_value as usize,
        });
    }
    if index_bytes > workspace.len() {
        return Ok(None);
    }
    let mut reads = Vec::with_capacity(plans.len());
    for plan in &plans {
        let output = workspace
            .try_allocate((plan.index.end - plan.index.start) as usize)?
            .ok_or_else(|| Error::internal("pre-admitted FullZip index refused"))?;
        reads.push((plan.index.clone(), output));
    }
    let indices = fill(&mut read, reads).await?;
    let mut ranges = Vec::with_capacity(plans.len());
    let mut input_bytes = 0usize;
    for (plan, index) in plans.iter().zip(indices) {
        let bytes = index.as_ref();
        let endpoint = |bytes: &[u8]| {
            let mut value = [0; 8];
            value[..bytes.len()].copy_from_slice(bytes);
            u64::from_le_bytes(value)
        };
        let start = endpoint(&bytes[..plan.width]);
        let end = endpoint(&bytes[bytes.len() - plan.width..]);
        let scheduler = &plan.scheduler;
        if start >= end
            || end > scheduler.data_buf_size
            || (plan.rows.start == 0 && start != 0)
            || (plan.rows.end == scheduler.rows_in_page && end != scheduler.data_buf_size)
        {
            return Err(Error::corrupt_file_named(
                "FullZip",
                "invalid repetition index endpoints",
            ));
        }
        let data_end = scheduler
            .data_buf_position
            .checked_add(end)
            .ok_or_else(|| Error::invalid_input("FullZip data address overflows"))?;
        let len = usize::try_from(end - start)
            .map_err(|_| Error::invalid_input("FullZip input exceeds addressable size"))?;
        input_bytes = input_bytes
            .checked_add(len)
            .ok_or_else(|| Error::invalid_input("FullZip batch input size overflows"))?;
        ranges.push((data_end - (end - start))..data_end);
        index.recycle(workspace)?;
    }
    // All index scratch is restored before refusing or allocating payloads.
    // The caller may use the sequential path with exactly its original credit.
    if input_bytes > workspace.len() {
        return Ok(None);
    }
    let mut reads = Vec::with_capacity(plans.len());
    for range in ranges {
        let input = workspace
            .try_allocate((range.end - range.start) as usize)?
            .ok_or_else(|| Error::internal("pre-admitted FullZip input refused"))?;
        reads.push((range, input));
    }
    let inputs = fill(&mut read, reads).await?;
    Ok(Some(
        plans
            .into_iter()
            .zip(inputs)
            .map(|(plan, input)| FullZipArrayInput {
                scheduler: plan.scheduler,
                input,
                data_type: plan.data_type,
                count: plan.rows.end - plan.rows.start,
            })
            .collect(),
    ))
}
