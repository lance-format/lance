// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_select::{RowAddrSelection, RowAddrTreeMap};
use roaring::RoaringBitmap;
use std::collections::BTreeMap;

/// A physical row range in a fragment of a particular dataset snapshot.
///
/// Use with [`super::scanner::Scanner::with_fragment_slices`]. For example,
/// `FragmentSlice { fragment_id: 0, row_offset: 10, row_count: 5 }` selects
/// physical offsets 10 through 14, omitting deleted rows during ordinary scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentSlice {
    /// Fragment identifier in the scanner's dataset snapshot.
    pub fragment_id: u32,
    /// Zero-based physical offset; deletions do not compact this coordinate.
    pub row_offset: u64,
    /// Number of physical slots. Zero selects no rows but still validates the offset.
    pub row_count: u64,
}

pub(super) async fn physical_rows_from_slices(
    dataset: &Dataset,
    slices: &[FragmentSlice],
) -> Result<RowAddrTreeMap> {
    let mut grouped = BTreeMap::<u32, Vec<&FragmentSlice>>::new();
    for slice in slices {
        grouped.entry(slice.fragment_id).or_default().push(slice);
    }
    let mut rows = RowAddrTreeMap::new();
    for (fragment_id, slices) in grouped {
        let fragment = dataset.get_fragment(fragment_id as usize).ok_or_else(|| Error::invalid_input(format!(
            "fragment slice references fragment_id={fragment_id}, which is not present in dataset version={}", dataset.version().version
        )))?;
        let physical_row_count = fragment.physical_rows().await? as u64;
        let mut offsets = RoaringBitmap::new();
        for slice in slices {
            let FragmentSlice {
                row_offset,
                row_count,
                ..
            } = *slice;
            let end = row_offset.checked_add(row_count).ok_or_else(|| Error::invalid_input(format!(
                "fragment slice end overflow: fragment_id={fragment_id}, row_offset={row_offset}, row_count={row_count}, physical_row_count={physical_row_count}, dataset_version={}", dataset.version().version
            )))?;
            if end > physical_row_count || end > RowAddress::FRAGMENT_SIZE {
                return Err(Error::invalid_input(format!(
                    "fragment slice is outside fragment bounds: fragment_id={fragment_id}, row_offset={row_offset}, row_count={row_count}, physical_row_count={physical_row_count}, dataset_version={}",
                    dataset.version().version
                )));
            }
            if row_count != 0 {
                // Nonempty, validated ranges start strictly below FRAGMENT_SIZE.
                // Bitmap insertion unions overlapping ranges without enumerating rows.
                offsets.insert_range(row_offset as u32..=(end - 1) as u32);
            }
        }
        if !offsets.is_empty() {
            rows.insert_bitmap(fragment_id, offsets);
        }
    }
    Ok(rows)
}

pub async fn validate_physical_rows(dataset: &Dataset, rows: &RowAddrTreeMap) -> Result<()> {
    for (fragment_id, selection) in rows.iter() {
        let fragment = dataset
                .get_fragment(*fragment_id as usize)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "physical row selection references fragment_id={fragment_id}, which is not present in dataset version={}",
                        dataset.version().version
                    ))
                })?;
        if let RowAddrSelection::Partial(offsets) = selection
            && let Some(max_offset) = offsets.max()
        {
            let physical_row_count = fragment.physical_rows().await? as u64;
            if u64::from(max_offset) >= physical_row_count {
                return Err(Error::invalid_input(format!(
                    "physical row selection for fragment_id={fragment_id} contains row_offset={max_offset}, but the fragment has physical_row_count={physical_row_count} in dataset version={}",
                    dataset.version().version
                )));
            }
        }
    }
    Ok(())
}
