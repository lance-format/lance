// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, num::NonZeroU64, ops::Range};

use lance_core::{Error, Result};
use lance_file::concat::{DataFilePart as OpenedDataFilePart, EncodedFileInput};
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use object_store::path::PathPart;
use serde::{Deserialize, Serialize};

use super::{DataFileTarget, Dataset};

/// Serializable description of a completed staging part, containing its identity
/// and expected metadata rather than file contents or open readers.
///
/// Assembly reopens and validates the actual file. Keep staging files until no
/// worker or checkpoint needs them, then call [`DataFileTarget::finish`] after
/// commit or [`DataFileTarget::cleanup`] when abandoning the target.
///
/// ```
/// # use lance::dataset::DataFilePart;
/// # fn checkpoint(part: &DataFilePart) -> Result<(), serde_json::Error> {
/// let bytes = serde_json::to_vec(part)?;
/// let restored: DataFilePart = serde_json::from_slice(&bytes)?;
/// # let _ = restored;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFilePart {
    pub(super) target_file_name: String,
    pub(super) base_id: Option<u32>,
    pub(super) file_name: String,
    pub(super) blob_ids: Option<Range<u32>>,
    pub(super) num_rows: u64,
    pub(super) size_bytes: NonZeroU64,
}

impl DataFilePart {
    /// Expected physical row count, verified against the footer during assembly.
    pub fn num_rows(&self) -> u64 {
        self.num_rows
    }

    /// Expected complete file size, verified during assembly.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes.get()
    }

    pub(super) async fn open_all(
        dataset: &Dataset,
        target: &DataFileTarget,
        parts: &[Self],
    ) -> Result<Vec<OpenedDataFilePart>> {
        dataset.validate_data_file_target(target)?;
        if parts.is_empty() {
            return Err(Error::invalid_input(
                "concat_data_file_parts requires at least one part",
            ));
        }
        let mut names = HashSet::with_capacity(parts.len());
        let mut ranges = Vec::with_capacity(parts.len());
        for part in parts {
            PathPart::parse(&part.file_name)
                .map_err(|error| Error::invalid_input(format!("invalid part identity: {error}")))?;
            if part.file_name.is_empty() {
                return Err(Error::invalid_input(
                    "part file_name must select a child object",
                ));
            }
            if part.target_file_name != target.file_name || part.base_id != target.base_id {
                return Err(Error::invalid_input(format!(
                    "part '{}' belongs to target {:?} in base {:?}, not {:?} in base {:?}",
                    part.file_name.as_str(),
                    part.target_file_name,
                    part.base_id,
                    target.file_name,
                    target.base_id
                )));
            }
            if !names.insert(part.file_name.as_str()) {
                return Err(Error::invalid_input(format!(
                    "duplicate part '{}'",
                    part.file_name.as_str()
                )));
            }
            if let Some(ids) = &part.blob_ids {
                ranges.push(ids);
            }
        }
        ranges.sort_unstable_by_key(|range| range.start);
        for pair in ranges.windows(2) {
            if pair[1].start < pair[0].end {
                return Err(Error::invalid_input(format!(
                    "part Blob ID ranges overlap: {:?} and {:?}",
                    pair[0], pair[1]
                )));
            }
        }
        let store = dataset.object_store(target.base_id).await?;
        let dir = target.parts_dir(&dataset.data_file_dir_for_base(target.base_id)?);
        let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
        let mut opened = Vec::with_capacity(parts.len());
        for part in parts {
            let path = dir.clone().join(part.file_name.as_str());
            let size = store.size(&path).await?;
            if size != part.size_bytes() {
                return Err(Error::invalid_input(format!(
                    "part at '{path}' has {size} bytes, checkpoint expects {}",
                    part.size_bytes()
                )));
            }
            let file = scheduler
                .open_file(&path, &CachedFileSize::new(size))
                .await?;
            opened.push(
                OpenedDataFilePart::open(
                    EncodedFileInput::new(file).with_expected_num_rows(part.num_rows()),
                    part.blob_ids.clone(),
                    target.blob_target_id(),
                )
                .await?,
            );
        }
        Ok(opened)
    }
}
