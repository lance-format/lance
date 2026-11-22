// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The stored zones of a committed ZoneMap segment, read so that a build of
//! the same index can take some fragments' statistics whole instead of
//! reading those fragments again.

use std::collections::HashMap;

use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_select::{RowAddrSelection, RowAddrTreeMap};
use roaring::RoaringBitmap;

use super::{
    ReusedZones, ZoneFileRead, ZoneMapIndexVersion, ZoneMapStatistics, parse_zones, read_zone_file,
    target_zone,
};
use crate::scalar::IndexStore;

/// The zones and null positions one committed ZoneMap segment stores for a
/// chosen set of its fragments.
///
/// Read with [`Self::load`]: the file is decoded and validated exactly as
/// loading the index does, but nothing is put in an index cache and no
/// address translation is installed, so every zone keeps the address it was
/// written with. Only the requested fragments are kept; the rest of the
/// file is dropped before `load` returns.
#[derive(Debug)]
pub struct StoredZones {
    version: ZoneMapIndexVersion,
    data_type: DataType,
    rows_per_zone: u64,
    store_null_bitmap: bool,
    stores_null_rows: bool,
    /// Zones of the requested fragments, in file order.
    zones: HashMap<u64, Vec<ZoneMapStatistics>>,
    /// Null positions of the requested fragments; `None` when the file
    /// stores none.
    null_rows: Option<RowAddrTreeMap>,
    payload_bytes: usize,
}

/// The outcome of [`StoredZones::load`] when the file could be read.
#[derive(Debug)]
pub enum StoredZonesLoad {
    Loaded(StoredZones),
    /// The bytes read are not a valid zone file under the requested
    /// version; the reason is the validation error's text.
    Rejected(String),
}

impl StoredZones {
    /// Read the zones and null positions of `fragments` from the segment
    /// file in `store`, decoded under `version` (the segment's
    /// `IndexMetadata.index_version`).
    ///
    /// A failed storage or file-reader call is an `Err`; a file whose bytes
    /// were read but do not validate is [`StoredZonesLoad::Rejected`].
    pub async fn load(
        store: &dyn IndexStore,
        version: ZoneMapIndexVersion,
        fragments: &RoaringBitmap,
    ) -> Result<StoredZonesLoad> {
        let file = match read_zone_file(store, true).await? {
            ZoneFileRead::File(file) => file,
            ZoneFileRead::Invalid(error) => {
                return Ok(StoredZonesLoad::Rejected(error.to_string()));
            }
        };
        let (data_type, all_zones) = match parse_zones(&file.zone_maps, version) {
            Ok(parsed) => parsed,
            Err(error) => return Ok(StoredZonesLoad::Rejected(error.to_string())),
        };
        drop(file.zone_maps);
        let mut zones: HashMap<u64, Vec<ZoneMapStatistics>> = HashMap::new();
        for zone in all_zones {
            let wanted = u32::try_from(zone.bound.fragment_id)
                .is_ok_and(|fragment| fragments.contains(fragment));
            if wanted {
                zones.entry(zone.bound.fragment_id).or_default().push(zone);
            }
        }
        let null_rows = file.null_rows.map(|mut rows| {
            rows.retain_fragments(fragments.iter());
            rows
        });
        Ok(StoredZonesLoad::Loaded(Self {
            version,
            data_type,
            rows_per_zone: file.rows_per_zone,
            store_null_bitmap: file.store_null_bitmap,
            stores_null_rows: file.stores_null_rows,
            zones,
            null_rows,
            payload_bytes: file.payload_bytes,
        }))
    }

    pub fn version(&self) -> ZoneMapIndexVersion {
        self.version
    }

    pub fn value_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn rows_per_zone(&self) -> u64 {
        self.rows_per_zone
    }

    /// The null-position policy the segment was configured with.
    pub fn store_null_bitmap(&self) -> bool {
        self.store_null_bitmap
    }

    /// Whether the segment file actually carries null positions.
    pub fn has_null_bitmap(&self) -> bool {
        self.stores_null_rows
    }

    /// Arrow memory of the whole file's zone record batch plus the
    /// serialized length of its null-position buffer, measured before the
    /// unrequested fragments were dropped. Not the memory of the parsed
    /// zones or null positions, and not a peak.
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    /// Why zones of this segment cannot feed a build of `data_type` with
    /// `rows_per_zone` that does (`keep_null_positions`) or does not keep
    /// null positions; `None` when they can.
    pub fn incompatibility(
        &self,
        data_type: &DataType,
        rows_per_zone: u64,
        keep_null_positions: bool,
    ) -> Option<String> {
        if self.version != ZoneMapIndexVersion::CURRENT {
            return Some(format!("index file version {}", self.version as u32));
        }
        if &self.data_type != data_type {
            return Some(format!(
                "value type {:?}, the build's is {data_type:?}",
                self.data_type
            ));
        }
        if self.rows_per_zone != rows_per_zone {
            return Some(format!(
                "rows_per_zone {}, the build's is {rows_per_zone}",
                self.rows_per_zone
            ));
        }
        if self.store_null_bitmap != keep_null_positions {
            return Some(format!(
                "null-position setting {}, the build's is {keep_null_positions}",
                self.store_null_bitmap
            ));
        }
        if keep_null_positions && !self.stores_null_rows {
            return Some("no stored null positions".to_string());
        }
        None
    }
}

/// Whether `records` are exactly the target zones of a fragment of
/// `physical_rows` rows: one record for each interval
/// `[k * rows_per_zone, min((k + 1) * rows_per_zone, physical_rows))`, with
/// no gap, overlap or duplicate. Covering `[0, physical_rows)` is not
/// enough: two shorter records inside one target interval do not match.
pub(super) fn matches_target_zones(
    records: &[ZoneMapStatistics],
    fragment_id: u64,
    rows_per_zone: u64,
    physical_rows: u64,
) -> bool {
    if rows_per_zone == 0 || physical_rows == 0 {
        return false;
    }
    if records.len() as u64 != physical_rows.div_ceil(rows_per_zone) {
        return false;
    }
    let mut records: Vec<&ZoneMapStatistics> = records.iter().collect();
    records.sort_by_key(|record| record.bound.start);
    records.iter().enumerate().all(|(k, record)| {
        let target = target_zone(fragment_id, k as u64, rows_per_zone, physical_rows);
        record.bound == target
    })
}

impl ReusedZones {
    /// Take fragment `fragment_id`'s statistics whole from a committed
    /// segment. Accepted only when the segment holds exactly the fragment's
    /// target zones (see `matches_target_zones`) with a known NaN sign
    /// split, and, when this accumulator keeps null positions, positions
    /// inside the fragment that agree with the records' null counts. The
    /// records are kept as stored. Returns `false`, keeping nothing, when
    /// any check fails; the caller then summarizes the fragment otherwise.
    ///
    /// The caller has checked `source` with [`StoredZones::incompatibility`].
    pub fn reuse_stored(
        &mut self,
        source: &StoredZones,
        fragment_id: u64,
        physical_rows: u64,
    ) -> Result<bool> {
        let reject = |reason: &str| {
            log::debug!("not reusing the stored zones of fragment {fragment_id}: {reason}");
            Ok(false)
        };
        let rows_per_zone = source.rows_per_zone;
        let Some(records) = source.zones.get(&fragment_id) else {
            return reject("the segment stores no zones for it");
        };
        if !matches_target_zones(records, fragment_id, rows_per_zone, physical_rows) {
            return reject("its stored zones are not its target zones");
        }
        if records
            .iter()
            .any(|record| record.negative_nan_count.is_none())
        {
            return reject("a stored zone has no NaN sign split");
        }
        let positions = if self.with_null_positions {
            let Some(null_rows) = source.null_rows.as_ref() else {
                return reject("the segment stores no null positions");
            };
            let fragment = u32::try_from(fragment_id)
                .map_err(|_| Error::internal(format!("fragment id {fragment_id} exceeds u32")))?;
            let bitmap = match null_rows.get(&fragment) {
                None => RoaringBitmap::new(),
                Some(RowAddrSelection::Partial(bitmap)) => bitmap.clone(),
                Some(RowAddrSelection::Full) => {
                    return reject("its null positions are stored as the whole fragment");
                }
            };
            let counted: u64 = records
                .iter()
                .map(|record| u64::from(record.null_count))
                .sum();
            if bitmap.len() != counted {
                return reject("its null positions disagree with the null counts");
            }
            if bitmap
                .max()
                .is_some_and(|offset| u64::from(offset) >= physical_rows)
            {
                return reject("a null position lies beyond the fragment");
            }
            Some((fragment, bitmap))
        } else {
            None
        };
        for record in records {
            let zone_index = record.bound.start / rows_per_zone;
            self.zones.insert((fragment_id, zone_index), record.clone());
        }
        if let Some((fragment, bitmap)) = positions
            && !bitmap.is_empty()
        {
            self.null_rows.insert_bitmap(fragment, bitmap);
        }
        self.stored_fragments.push(fragment_id);
        self.stored_records += records.len();
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
