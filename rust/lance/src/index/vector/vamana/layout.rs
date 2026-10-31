// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Where a data file keeps the vectors of a `FixedSizeList<Float32, d>`
//! column, so that a re-score reads the few rows it scores at their byte
//! offsets rather than through a decoder.
//!
//! A full-zip page of a fixed-width value with no repetition to record stores
//! its values flat, one row after another, and Lance's own decoder finds row
//! `r` of the page at `data_buf_position + r * (bytes_per_value +
//! bytes_per_word)` (`FullZipScheduler::schedule_ranges_simple`). Lance writes
//! vectors of 256 bytes or more on full-zip pages. A page that records nulls,
//! of a vector or of a struct around it, puts a control word in front of every
//! vector, and a page that records none puts none. Each page is checked
//! against every part of that arithmetic, and the rows of a page that fails a
//! check are left to take, which reads anything, as is a row whose control
//! word is not zero.

use std::borrow::Cow;
use std::ops::Range;

use lance_core::cache::CacheKey;
use lance_core::deepsize::DeepSizeOf;
use lance_encoding::decoder::{ColumnInfo, PageEncoding, PageInfo};
use lance_encoding::format::metadata;
use lance_file::reader::CachedFileMetadata;
use lance_file::version::ConcreteFileVersion;

use crate::{Error, Result};

/// Where the rows of one vector column of a data file lie.
#[derive(Debug, PartialEq, DeepSizeOf)]
pub struct VectorLayout {
    /// Bytes one vector occupies, four for each of its values.
    stride: u64,
    /// The column's pages in row order.
    pages: Vec<PageSpan>,
}

/// One page of the column: the rows it holds, and where its values lie when
/// they are stored flat.
#[derive(Debug, PartialEq, DeepSizeOf)]
struct PageSpan {
    first_row: u64,
    num_rows: u64,
    /// `None` for a page that does not store its values flat, whose rows are
    /// read through take.
    flat: Option<FlatValues>,
}

/// Where a page that stores its values flat keeps them.
#[derive(Debug, PartialEq, DeepSizeOf)]
struct FlatValues {
    /// Where its first row starts.
    position: u64,
    /// The bytes of the control word in front of each of its vectors: one on
    /// a page that records nulls, of a vector or of a struct around it, none on
    /// any other.
    control: u64,
}

impl PageSpan {
    fn end(&self) -> u64 {
        self.first_row + self.num_rows
    }
}

impl VectorLayout {
    /// The layout of `column`, whose values the caller has established are
    /// `FixedSizeList<Float32, dimension>`, in a data file of `rows` rows.
    ///
    /// `None` when the vectors hold no values, when the pages do not hold
    /// exactly `rows` rows, or when the pages stored flat do not follow one
    /// another through the file, as no Lance writer lays them out: no row of
    /// such a file is read by offset.
    pub fn of_column(column: &ColumnInfo, dimension: u32, rows: u64) -> Option<Self> {
        if dimension == 0 {
            return None;
        }
        let stride = u64::from(dimension) * 4;
        let mut pages = Vec::with_capacity(column.page_infos.len());
        let mut first_row = 0u64;
        let mut written_to = 0u64;
        for page in column.page_infos.iter() {
            let flat = flat_values(page, dimension, stride);
            if let Some(flat) = &flat {
                if flat.position < written_to {
                    return None;
                }
                let bytes = page.num_rows.checked_mul(stride + flat.control)?;
                written_to = flat.position.checked_add(bytes)?;
            }
            pages.push(PageSpan {
                first_row,
                num_rows: page.num_rows,
                flat,
            });
            first_row = first_row.checked_add(page.num_rows)?;
        }
        (first_row == rows).then_some(Self { stride, pages })
    }

    /// The bytes that hold the vector of each of `rows`, which ascend, with
    /// the control word its page puts in front of it, if any (see [`decode`]),
    /// or `None` for a row whose page does not store its values flat.
    pub fn locate(&self, rows: &[u32]) -> Result<Vec<Option<Range<u64>>>> {
        if let Some(pair) = rows.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(Error::internal(format!(
                "IVF_VAMANA locates strictly ascending rows of a data file, got {} then {}",
                pair[0], pair[1]
            )));
        }
        let mut page = 0;
        rows.iter()
            .map(|&row| {
                let row = u64::from(row);
                while self.pages.get(page).is_some_and(|span| row >= span.end()) {
                    page += 1;
                }
                let span = self.pages.get(page).ok_or_else(|| {
                    Error::internal(format!(
                        "IVF_VAMANA located row {row} of a data file holding {} rows",
                        self.pages.last().map_or(0, PageSpan::end)
                    ))
                })?;
                Ok(span.flat.as_ref().map(|flat| {
                    let row_bytes = flat.control + self.stride;
                    let start = flat.position + (row - span.first_row) * row_bytes;
                    start..start + row_bytes
                }))
            })
            .collect()
    }
}

/// The layout of a data file's vector column as a re-score built it from the
/// file's footer, kept in the session's cache of that file's metadata with
/// what it was built for: the version the footer gave, and the rows and
/// dimension the manifest did. A later manifest may disagree, and then the
/// layout is not used.
#[derive(Debug, DeepSizeOf)]
pub struct CachedLayout {
    version: ConcreteFileVersion,
    rows: u64,
    dimension: u32,
    layout: Option<VectorLayout>,
}

impl CachedLayout {
    /// The layout of column `column` of the file `metadata` describes, which
    /// the manifest gives `rows` rows of vectors of `dimension` values.
    pub fn of(metadata: &CachedFileMetadata, column: u32, dimension: u32, rows: u64) -> Self {
        let layout = metadata
            .column_infos
            .get(column as usize)
            .and_then(|info| VectorLayout::of_column(info, dimension, rows));
        Self {
            version: metadata.version,
            rows,
            dimension,
            layout,
        }
    }

    /// The layout, unless the manifest now gives the file another `version`
    /// than its footer does, other `rows`, or vectors of another `dimension`.
    pub fn layout_for(
        &self,
        version: ConcreteFileVersion,
        rows: u64,
        dimension: u32,
    ) -> Option<&VectorLayout> {
        if self.version != version || self.rows != rows || self.dimension != dimension {
            return None;
        }
        self.layout.as_ref()
    }
}

/// The key of a [`CachedLayout`] in the cache of its file's metadata: the
/// column's index among the file's columns.
pub struct LayoutKey {
    pub column: u32,
}

impl CacheKey for LayoutKey {
    type ValueType = CachedLayout;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("vamana-vector-layout/{}", self.column))
    }

    fn type_name() -> &'static str {
        "VamanaVectorLayout"
    }
}

/// Where `page` keeps its values, if it stores them flat: full-zip with no
/// repetition and at most a one-byte control word in front of each value,
/// each value exactly `dimension` uncompressed single-precision floats with no
/// validity bitmap, all in one buffer that holds exactly the page's rows.
///
/// The checks overlap on purpose. The descriptors say what is in front of a
/// value and that a value is its floats, and the buffer's size says the page
/// holds its rows at that stride and nothing else; any one of them failing
/// while the others pass would mean Lance lays pages out in a way this module
/// does not know. Early 2.1 writers did so: some of their pages hold a zero
/// byte in front of each value and record no definition level, and only the
/// buffer's exact size tells them apart.
fn flat_values(page: &PageInfo, dimension: u32, stride: u64) -> Option<FlatValues> {
    let PageEncoding::Structural(layout) = &page.encoding else {
        return None;
    };
    let Some(metadata::page_layout::Layout::FullZipLayout(zip)) = &layout.layout else {
        return None;
    };
    // A control word sits in front of every value of a page that records
    // repetition or a null. Under repetition an empty or null list has a
    // control word and no value, so no stride finds a row. A null alone keeps
    // its value, as every item outside a list does (`max_visible_level`), and
    // its definition level takes one byte up to 8 bits
    // (`ControlWordParser::new`).
    if zip.bits_rep != 0 || zip.bits_def > 8 {
        return None;
    }
    let control = u64::from(zip.bits_def != 0);
    let Some(metadata::full_zip_layout::Details::BitsPerValue(bits)) = zip.details else {
        return None;
    };
    let Some(metadata::compressive_encoding::Compression::FixedSizeList(list)) = zip
        .value_compression
        .as_ref()
        .and_then(|encoding| encoding.compression.as_ref())
    else {
        return None;
    };
    let Some(metadata::compressive_encoding::Compression::Flat(flat)) = list
        .values
        .as_ref()
        .and_then(|encoding| encoding.compression.as_ref())
    else {
        return None;
    };
    if list.has_validity
        || list.items_per_value != u64::from(dimension)
        || flat.bits_per_value != 32
        || flat.data.is_some()
        || u64::from(bits) != stride * 8
        || u64::from(zip.num_items) != page.num_rows
    {
        return None;
    }
    let [(position, size)] = page.buffer_offsets_and_sizes[..] else {
        return None;
    };
    (size == page.num_rows.checked_mul(control + stride)?)
        .then_some(FlatValues { position, control })
}

/// The vectors [`decode`] read.
#[derive(Debug)]
pub struct Decoded {
    /// The values of the vectors read, one after another.
    pub values: Vec<f32>,
    /// For each chunk in turn, whether it holds a vector, rather than a row
    /// whose control word is not zero.
    pub has_vector: Vec<bool>,
}

/// The vectors of `dimension` single-precision floats in `chunks`, one to a
/// chunk as [`VectorLayout::locate`] gave them, little-endian as Lance writes
/// them. A chunk one byte longer than a vector starts with the control word
/// its page puts in front of each vector: the row's definition level, zero
/// when the vector and every struct around it are valid (`is_valid_item`). A
/// row of any other level is for take to read as Lance does.
pub fn decode<'a>(
    chunks: impl ExactSizeIterator<Item = &'a [u8]>,
    dimension: u32,
) -> Result<Decoded> {
    let stride = dimension as usize * 4;
    let mut decoded = Decoded {
        values: Vec::with_capacity(chunks.len() * dimension as usize),
        has_vector: Vec::with_capacity(chunks.len()),
    };
    for chunk in chunks {
        let vector = match chunk {
            vector if vector.len() == stride => Some(vector),
            [control, vector @ ..] if vector.len() == stride => (*control == 0).then_some(vector),
            _ => {
                return Err(Error::internal(format!(
                    "IVF_VAMANA read {} bytes for a vector of {stride}, or of {} with its control \
                     word",
                    chunk.len(),
                    stride + 1
                )));
            }
        };
        decoded.has_vector.push(vector.is_some());
        if let Some(vector) = vector {
            decoded.values.extend(
                vector
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|value| f32::from_le_bytes(*value)),
            );
        }
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_encoding::format::{EncodingUtils, pb};
    use lance_encoding::repdef::DefinitionInterpretation;
    use rstest::rstest;

    use super::*;

    const DIMENSION: u32 = 4;
    const STRIDE: u64 = 16;

    /// A page of `rows` vectors laid out as Lance lays out a page without a
    /// null, its values starting at `position`.
    fn flat_page(rows: u64, position: u64) -> PageInfo {
        let values = EncodingUtils::fsl(u64::from(DIMENSION), false, EncodingUtils::flat(32, None));
        let layout = EncodingUtils::fixed_full_zip_layout(
            0,
            0,
            DIMENSION * 32,
            values,
            &[DefinitionInterpretation::AllValidItem],
            rows as u32,
            rows as u32,
        );
        page(rows, layout, &[(position, rows * STRIDE)])
    }

    /// A page of `rows` vectors laid out as Lance lays out a page that records
    /// nulls in `bits_def` bits, its first control word at `position`.
    fn null_page(rows: u64, position: u64, bits_def: u8) -> PageInfo {
        let layout = null_vectors_layout(rows as u32, bits_def);
        page(rows, layout, &[(position, rows * (STRIDE + 1))])
    }

    fn page(rows: u64, layout: metadata::PageLayout, buffers: &[(u64, u64)]) -> PageInfo {
        PageInfo {
            num_rows: rows,
            priority: 0,
            encoding: PageEncoding::Structural(layout),
            buffer_offsets_and_sizes: Arc::from(buffers),
        }
    }

    fn column(pages: Vec<PageInfo>) -> ColumnInfo {
        ColumnInfo {
            index: 0,
            page_infos: Arc::from(pages),
            buffer_offsets_and_sizes: Arc::from([]),
            encoding: pb::ColumnEncoding::default(),
        }
    }

    /// `flat_page` with one part of its layout changed.
    fn altered(
        rows: u64,
        position: u64,
        alter: impl FnOnce(&mut metadata::FullZipLayout),
    ) -> PageInfo {
        let mut page = flat_page(rows, position);
        let PageEncoding::Structural(metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::FullZipLayout(zip)),
        }) = &mut page.encoding
        else {
            unreachable!()
        };
        alter(zip);
        page
    }

    #[test]
    fn rows_of_flat_pages_are_found_page_by_page() {
        let layout =
            VectorLayout::of_column(&column(vec![flat_page(3, 64), flat_page(2, 1024)]), 4, 5)
                .unwrap();
        let ranges = layout.locate(&[0, 2, 3, 4]).unwrap();
        assert_eq!(
            ranges,
            vec![
                Some(64..80),
                Some(96..112),
                Some(1024..1040),
                Some(1040..1056)
            ]
        );
    }

    /// The rows of a page that records nulls, of its vectors or of the
    /// structs around them, are found with the one-byte control word in front
    /// of each, and the rows of the pages around it where they were.
    #[rstest]
    fn rows_of_a_page_with_a_null_are_found_with_their_control_words(
        #[values(1, 2, 8)] bits_def: u8,
    ) {
        let pages = column(vec![
            flat_page(3, 64),
            null_page(2, 64 + 3 * STRIDE, bits_def),
            flat_page(2, 1024),
        ]);
        let layout = VectorLayout::of_column(&pages, DIMENSION, 7).unwrap();
        let ranges = layout.locate(&[2, 3, 4, 5]).unwrap();
        assert_eq!(
            ranges,
            vec![
                Some(96..112),
                Some(112..129),
                Some(129..146),
                Some(1024..1040)
            ]
        );
    }

    /// A cached layout is used only while the manifest gives its file the
    /// version the footer did, and the rows and dimension it was built for.
    #[test]
    fn a_cached_layout_is_used_only_for_what_it_was_built_for() {
        let pages = column(vec![flat_page(3, 64), flat_page(2, 1024)]);
        let cached = CachedLayout {
            version: ConcreteFileVersion::V2_2,
            rows: 5,
            dimension: DIMENSION,
            layout: VectorLayout::of_column(&pages, DIMENSION, 5),
        };
        assert!(
            cached
                .layout_for(ConcreteFileVersion::V2_2, 5, DIMENSION)
                .is_some()
        );
        for (version, rows, dimension) in [
            (ConcreteFileVersion::V2_1, 5, DIMENSION),
            (ConcreteFileVersion::V2_2, 6, DIMENSION),
            (ConcreteFileVersion::V2_2, 5, 2 * DIMENSION),
        ] {
            assert!(
                cached.layout_for(version, rows, dimension).is_none(),
                "{version:?} {rows} {dimension}"
            );
        }
    }

    /// Each check sends only the rows of its own page to take.
    #[rstest]
    #[case::no_room_for_control_words(altered(2, 128, |zip| zip.bits_def = 1))]
    #[case::unrecorded_control_words(page(2, flat_layout(2), &[(128, 2 * (STRIDE + 1))]))]
    #[case::wide_control_words(page(2, null_vectors_layout(2, 9), &[(128, 2 * (STRIDE + 1))]))]
    #[case::repetition(altered(2, 128, |zip| zip.bits_rep = 1))]
    #[case::validity_bitmap(altered(2, 128, |zip| {
        zip.value_compression = Some(EncodingUtils::fsl(4, true, EncodingUtils::flat(32, None)));
    }))]
    #[case::other_width(altered(2, 128, |zip| {
        zip.value_compression = Some(EncodingUtils::fsl(4, false, EncodingUtils::flat(64, None)));
        zip.details = Some(metadata::full_zip_layout::Details::BitsPerValue(256));
    }))]
    #[case::other_dimension(altered(2, 128, |zip| {
        zip.value_compression = Some(EncodingUtils::fsl(2, false, EncodingUtils::flat(64, None)));
    }))]
    #[case::compressed(altered(2, 128, |zip| {
        zip.value_compression = Some(EncodingUtils::fsl(
            4,
            false,
            EncodingUtils::flat(32, Some(metadata::BufferCompression::default())),
        ));
    }))]
    #[case::variable_width(altered(2, 128, |zip| {
        zip.details = Some(metadata::full_zip_layout::Details::BitsPerOffset(32));
    }))]
    #[case::other_value_width(altered(2, 128, |zip| {
        zip.details = Some(metadata::full_zip_layout::Details::BitsPerValue(DIMENSION * 32 + 8));
    }))]
    #[case::fewer_items(altered(2, 128, |zip| zip.num_items = 1))]
    #[case::short_buffer(page(2, flat_layout(2), &[(128, 31)]))]
    #[case::two_buffers(page(2, flat_layout(2), &[(128, 32), (160, 8)]))]
    #[case::mini_block(page(2, mini_block_layout(2), &[(128, 32)]))]
    #[case::constant(page(2, constant_layout(), &[]))]
    #[case::legacy(PageInfo {
        encoding: PageEncoding::Legacy(pb::ArrayEncoding::default()),
        ..flat_page(2, 128)
    })]
    fn a_page_not_stored_flat_is_left_to_take(#[case] odd: PageInfo) {
        let layout = VectorLayout::of_column(
            &column(vec![flat_page(2, 0), odd, flat_page(2, 256)]),
            DIMENSION,
            6,
        )
        .unwrap();
        let ranges = layout.locate(&[1, 2, 3, 4]).unwrap();
        assert_eq!(ranges, vec![Some(16..32), None, None, Some(256..272)]);
    }

    fn flat_layout(rows: u32) -> metadata::PageLayout {
        let PageEncoding::Structural(layout) = flat_page(u64::from(rows), 0).encoding else {
            unreachable!()
        };
        layout
    }

    /// A page as Lance lays one out when one of its vectors is null: a
    /// control word of `bits_def` bits in front of each vector says whether it
    /// is.
    fn null_vectors_layout(rows: u32, bits_def: u8) -> metadata::PageLayout {
        EncodingUtils::fixed_full_zip_layout(
            0,
            bits_def,
            DIMENSION * 32,
            EncodingUtils::fsl(u64::from(DIMENSION), false, EncodingUtils::flat(32, None)),
            &[DefinitionInterpretation::NullableItem],
            rows,
            rows,
        )
    }

    fn constant_layout() -> metadata::PageLayout {
        metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::ConstantLayout(
                metadata::ConstantLayout::default(),
            )),
        }
    }

    fn mini_block_layout(rows: u64) -> metadata::PageLayout {
        EncodingUtils::miniblock_layout(
            None,
            None,
            EncodingUtils::fsl(4, false, EncodingUtils::flat(32, None)),
            0,
            1,
            None,
            &[DefinitionInterpretation::AllValidItem],
            rows,
            false,
        )
    }

    #[test]
    fn a_column_that_is_not_the_files_rows_or_in_file_order_has_no_layout() {
        let pages = || vec![flat_page(3, 64), flat_page(2, 1024)];
        assert!(VectorLayout::of_column(&column(pages()), DIMENSION, 4).is_none());
        assert!(VectorLayout::of_column(&column(pages()), DIMENSION, 6).is_none());
        let backwards = column(vec![flat_page(3, 1024), flat_page(2, 64)]);
        assert!(VectorLayout::of_column(&backwards, DIMENSION, 5).is_none());
        let overlapping = column(vec![flat_page(3, 64), flat_page(2, 64 + 2 * STRIDE)]);
        assert!(VectorLayout::of_column(&overlapping, DIMENSION, 5).is_none());
        // The last control word's byte is all the overlap.
        let past_control_words = column(vec![
            null_page(2, 64, 1),
            flat_page(2, 64 + 2 * (STRIDE + 1) - 1),
        ]);
        assert!(VectorLayout::of_column(&past_control_words, DIMENSION, 4).is_none());
    }

    /// Vectors without values occupy no bytes, so no offset tells them apart.
    #[test]
    fn vectors_without_values_have_no_layout() {
        let layout = EncodingUtils::fixed_full_zip_layout(
            0,
            0,
            0,
            EncodingUtils::fsl(0, false, EncodingUtils::flat(32, None)),
            &[DefinitionInterpretation::AllValidItem],
            2,
            2,
        );
        let column = column(vec![page(2, layout, &[(128, 0)])]);
        assert!(VectorLayout::of_column(&column, 0, 2).is_none());
    }

    #[test]
    fn rows_out_of_order_or_past_the_file_are_refused() {
        let layout =
            VectorLayout::of_column(&column(vec![flat_page(3, 64)]), DIMENSION, 3).unwrap();
        for rows in [&[1, 1][..], &[2, 1]] {
            let error = layout.locate(rows).unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error:?}");
            assert!(error.to_string().contains("strictly ascending"), "{error}");
        }
        let error = layout.locate(&[0, 3]).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("row 3 of a data file holding 3 rows"),
            "{error}"
        );
    }

    /// Vectors decode to the floats written, with or without the control word
    /// in front, which says a vector is there when it is zero and a null
    /// otherwise.
    #[test]
    fn vectors_decode_to_the_floats_written() {
        let written = [
            [1.5f32, -2.0, 0.25, 7.0],
            [f32::MIN_POSITIVE, 0.0, -0.0, 1e30],
            [3.0, 4.0, 5.0, 6.0],
            [-1.0, -3.5, 9.0, 0.5],
        ];
        let bytes = written
            .iter()
            .map(|vector| {
                vector
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            })
            .collect::<Vec<Vec<u8>>>();
        let chunks = [
            bytes[0].clone(),
            [&[0][..], &bytes[1]].concat(),
            [&[1][..], &bytes[2]].concat(),
            [&[2][..], &bytes[3]].concat(),
        ];
        let decoded = decode(chunks.iter().map(Vec::as_slice), DIMENSION).unwrap();
        assert_eq!(decoded.has_vector, [true, true, false, false]);
        assert_eq!(
            decoded
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            written[..2]
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        for length in [15, 18] {
            let chunk = [&bytes[0][..], &bytes[1][..2]].concat();
            let error = decode([&chunk[..length]].into_iter(), DIMENSION).unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error:?}");
            assert!(
                error.to_string().contains(&format!(
                    "read {length} bytes for a vector of 16, or of 17 with its control word"
                )),
                "{error}"
            );
        }
    }
}
