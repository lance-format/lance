// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan encoded data reads without scheduling decoders or materializing values.

use std::ops::Range;

use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use crate::reader::{CachedFileMetadata, FileReader, ReaderProjection};

/// Encoded buffers needed by a projected physical row selection.
#[derive(Debug, Default)]
pub struct EncodedPagePlan {
    /// Sorted, disjoint file-relative byte ranges, including shared dependencies.
    pub byte_ranges: Vec<Range<u64>>,
    /// Number of projected pages retained, before byte-range coalescing.
    pub selected_pages: u64,
    /// Why the projection required whole-column rather than page-level warming.
    pub fallback_reason: Option<String>,
}

impl EncodedPagePlan {
    /// Bytes covered by the disjoint plan; excludes metadata read while planning.
    pub fn num_bytes(&self) -> u64 {
        self.byte_ranges.iter().map(|r| r.end - r.start).sum()
    }
}

/// Map physical rows to projected encoded pages using file metadata only.
///
/// `None` selects all rows; an empty bitmap selects nothing. Flat columns with
/// one encoded row per top-level row are narrowed to overlapping pages. Nested
/// layouts conservatively retain every projected page because repetition and
/// offset metadata can be needed to map their row coordinates. Shared column
/// and file buffers are retained in either case. Blob payloads must be planned
/// separately: this function rejects blob projections rather than claiming that
/// warming their descriptors also warms the out-of-line payload.
///
/// ```
/// # use lance_file::{prewarm::plan_encoded_pages, reader::{CachedFileMetadata, ReaderProjection}};
/// # fn example(metadata: &CachedFileMetadata, projection: &ReaderProjection) -> lance_core::Result<()> {
/// let plan = plan_encoded_pages(metadata, projection, None)?;
/// for range in plan.byte_ranges { /* fetch through the query's cached reader */ }
/// # Ok(()) }
/// ```
pub fn plan_encoded_pages(
    metadata: &CachedFileMetadata,
    projection: &ReaderProjection,
    rows: Option<&RoaringBitmap>,
) -> Result<EncodedPagePlan> {
    FileReader::validate_projection(projection, metadata)?;
    if projection.schema.fields_pre_order().any(|f| f.is_blob()) {
        return Err(Error::not_supported(
            "encoded page prewarm cannot locate out-of-line blob payloads from footer metadata",
        ));
    }
    let mut plan = EncodedPagePlan::default();
    if rows.is_some_and(RoaringBitmap::is_empty) || metadata.num_rows == 0 {
        return Ok(plan);
    }
    if rows
        .and_then(RoaringBitmap::max)
        .is_some_and(|r| u64::from(r) >= metadata.num_rows)
    {
        return Err(Error::invalid_input(
            "prewarm physical row selection exceeds file row count",
        ));
    }
    let flat = projection
        .schema
        .fields
        .iter()
        .all(|f| f.children.is_empty());
    if !flat {
        plan.fallback_reason =
            Some("nested projection requires whole-column encoded buffers".into());
    }
    let mut push_buffer = |offset: u64, size: u64| -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| Error::invalid_input("prewarm buffer range overflows u64"))?;
        if end > metadata.file_size_bytes {
            return Err(Error::invalid_input(format!(
                "prewarm buffer {offset}..{end} exceeds file size {}",
                metadata.file_size_bytes
            )));
        }
        if size > 0 {
            plan.byte_ranges.push(offset..end);
        }
        Ok(())
    };
    for &column in &projection.column_indices {
        let info = metadata.column_infos.get(column as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "prewarm column {column} is absent from file metadata"
            ))
        })?;
        // A row-count mismatch can mean nested/fixed-size-list coordinates. It
        // must broaden the selection, never silently exclude encoded pages.
        let page_rows = info
            .page_infos
            .iter()
            .try_fold(0u64, |n, p| n.checked_add(p.num_rows))
            .ok_or_else(|| Error::invalid_input("prewarm page row count overflows u64"))?;
        let aligned = flat && page_rows == metadata.num_rows;
        if !aligned && plan.fallback_reason.is_none() {
            plan.fallback_reason =
                Some("encoded page rows do not map directly to top-level rows".into());
        }
        let mut start = 0;
        for page in info.page_infos.iter() {
            let end = start + page.num_rows;
            let selected = !aligned
                || rows.is_none_or(|rows| {
                    let first = rows.rank(start.saturating_sub(1).min(u64::from(u32::MAX)) as u32);
                    let before = if start == 0 { 0 } else { first };
                    let through = rows.rank(end.saturating_sub(1).min(u64::from(u32::MAX)) as u32);
                    through > before
                });
            if selected {
                plan.selected_pages += 1;
                for &(offset, size) in page.buffer_offsets_and_sizes.iter() {
                    push_buffer(offset, size)?;
                }
            }
            start = end;
        }
        for &(offset, size) in info.buffer_offsets_and_sizes.iter() {
            push_buffer(offset, size)?;
        }
    }
    for buffer in &metadata.file_buffers {
        push_buffer(buffer.position, buffer.size)?;
    }
    plan.byte_ranges.sort_unstable_by_key(|r| r.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(plan.byte_ranges.len());
    for range in plan.byte_ranges {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    plan.byte_ranges = merged;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_encoding::{
        decoder::{ColumnInfo, PageEncoding, PageInfo},
        format::{metadata, pb},
    };

    use super::*;
    use crate::{reader::BufferDescriptor, version::ConcreteFileVersion};

    fn fixture() -> (CachedFileMetadata, ReaderProjection) {
        let schema = Arc::new(
            Schema::try_from(&ArrowSchema::new(vec![Field::new(
                "x",
                DataType::Int64,
                true,
            )]))
            .unwrap(),
        );
        let column = |index, base| {
            Arc::new(ColumnInfo {
                index,
                page_infos: (0..3)
                    .map(|page| PageInfo {
                        num_rows: 4,
                        priority: page * 4,
                        encoding: PageEncoding::Structural(metadata::PageLayout::default()),
                        buffer_offsets_and_sizes: Arc::from([(base + page * 100, 100)]),
                    })
                    .collect::<Vec<_>>()
                    .into(),
                buffer_offsets_and_sizes: Arc::from([(base + 400, 20)]),
                encoding: pb::ColumnEncoding::default(),
            })
        };
        let metadata = CachedFileMetadata {
            file_schema: schema.clone(),
            column_metadatas: vec![],
            column_infos: vec![column(0, 0), column(1, 1000)],
            num_rows: 12,
            file_buffers: vec![BufferDescriptor {
                position: 2000,
                size: 20,
            }],
            num_data_bytes: 600,
            num_column_metadata_bytes: 0,
            num_global_buffer_bytes: 20,
            num_footer_bytes: 0,
            major_version: 2,
            minor_version: 1,
            version: ConcreteFileVersion::V2_1,
            file_size_bytes: 2048,
            retained_global_buffers: BTreeMap::new(),
        };
        (
            metadata,
            ReaderProjection {
                schema,
                column_indices: vec![0],
            },
        )
    }

    #[test]
    fn encoded_prewarm_selects_overlapping_pages_and_shared_dependencies() {
        let (metadata, projection) = fixture();
        let rows = RoaringBitmap::from_iter([4, 7]);
        let plan = plan_encoded_pages(&metadata, &projection, Some(&rows)).unwrap();
        assert_eq!(plan.byte_ranges, vec![100..200, 400..420, 2000..2020]);
        assert_eq!(plan.selected_pages, 1);
        assert_eq!(plan.num_bytes(), 140);
        assert!(plan.fallback_reason.is_none());
        // An adjacent selected row crosses the physical page boundary.
        let rows = RoaringBitmap::from_iter([3, 4]);
        assert_eq!(
            plan_encoded_pages(&metadata, &projection, Some(&rows))
                .unwrap()
                .byte_ranges,
            vec![0..200, 400..420, 2000..2020]
        );
    }

    #[test]
    fn encoded_prewarm_empty_is_distinct_from_all_rows() {
        let (metadata, projection) = fixture();
        assert!(
            plan_encoded_pages(&metadata, &projection, Some(&RoaringBitmap::new()))
                .unwrap()
                .byte_ranges
                .is_empty()
        );
        let all = plan_encoded_pages(&metadata, &projection, None).unwrap();
        assert_eq!(all.selected_pages, 3);
        assert_eq!(all.byte_ranges, vec![0..300, 400..420, 2000..2020]);
    }

    #[test]
    fn encoded_prewarm_rejects_out_of_bounds_rows_and_buffers() {
        let (mut metadata, projection) = fixture();
        let err = plan_encoded_pages(
            &metadata,
            &projection,
            Some(&RoaringBitmap::from_iter([12])),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("row selection"));
        metadata.file_buffers[0].size = u64::MAX;
        let err = plan_encoded_pages(&metadata, &projection, None).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("overflows"));
    }

    #[test]
    fn encoded_prewarm_nested_projection_keeps_all_projected_pages() {
        let (metadata, mut projection) = fixture();
        projection.schema = Arc::new(
            Schema::try_from(&ArrowSchema::new(vec![Field::new(
                "items",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            )]))
            .unwrap(),
        );
        let plan = plan_encoded_pages(&metadata, &projection, Some(&RoaringBitmap::from_iter([4])))
            .unwrap();
        assert_eq!(plan.selected_pages, 3);
        assert_eq!(plan.byte_ranges, vec![0..300, 400..420, 2000..2020]);
        assert!(plan.fallback_reason.unwrap().contains("nested"));
    }

    #[test]
    fn encoded_prewarm_blob_descriptors_are_not_reported_as_warm_payloads() {
        let (metadata, mut projection) = fixture();
        let field = Field::new("blob", DataType::LargeBinary, true).with_metadata(
            std::collections::HashMap::from([(
                lance_arrow::BLOB_META_KEY.to_string(),
                "true".into(),
            )]),
        );
        projection.schema = Arc::new(Schema::try_from(&ArrowSchema::new(vec![field])).unwrap());
        let error = plan_encoded_pages(&metadata, &projection, None).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("blob payloads"));
    }
}
