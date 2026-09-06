// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! Shared input matrix for the sparse structural benchmarks.
//!
//! The sparse structural layout is chosen at write time, not read time: a page is
//! scheduled by [`SparseStructuralScheduler`] whenever it was written with
//! `Layout::SparseLayout`. That happens when a field asks for
//! `lance-encoding:structural-encoding=sparse`, or automatically for a Lance 2.3+
//! nested column whose rep/def levels overflow the dense mini-block budget.
//!
//! Every case therefore records which layout it *expects*, and the benches assert it.
//! Without that check it is easy to benchmark the dense mini-block path by accident and
//! conclude, wrongly, that a sparse change had no effect.

// Shared by the `sparse_decode` and `sparse_footprint` bench targets, each of which uses a
// subset of this module.
#![allow(dead_code)]

use std::{collections::HashMap, sync::Arc};

use arrow_array::{Array, ArrayRef, Int32Array, ListArray, RecordBatch};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Schema};
use lance_encoding::{
    constants::{STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_SPARSE},
    decoder::PageEncoding,
    encoder::{EncodedBatch, EncodingOptions, default_encoding_strategy, encode_batch},
    format::pb21::page_layout::Layout,
    version::LanceFileVersion,
};

/// Sparse structural encoding is only available from this version onwards.
pub const SPARSE_VERSION: LanceFileVersion = LanceFileVersion::V2_3;

const ITEM: &str = "item";
const COLUMN: &str = "l";

/// The structural layout a page was written with, as observed from the encoded batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedLayout {
    Sparse,
    MiniBlock,
    FullZip,
    /// Every visible value is the same scalar or null, so the page stores no chunk index.
    Constant,
    Blob,
    Legacy,
    Other,
}

impl std::fmt::Display for ObservedLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Sparse => "sparse",
            Self::MiniBlock => "miniblock",
            Self::FullZip => "fullzip",
            Self::Constant => "constant",
            Self::Blob => "blob",
            Self::Legacy => "legacy",
            Self::Other => "other",
        };
        f.write_str(name)
    }
}

/// The shape of the list column under test.
///
/// Shape controls both whether the writer picks the sparse layout and, within a sparse
/// page, which chunk-index representation the reader ends up caching: uniform value
/// counts map arithmetically, non-uniform counts need an explicit prefix array, and
/// nested data carries per-chunk row metadata.
#[derive(Debug, Clone, Copy)]
pub enum Shape {
    /// A single row holding a single value.
    SingleValue,
    /// Every row holds exactly `run` values, so all non-final chunks are the same size.
    Uniform { rows: usize, run: usize },
    /// Uniform except for a final oversized row, which breaks uniform chunk sizing.
    UniformWithBigTail {
        rows: usize,
        run: usize,
        tail: usize,
    },
    /// Every `stride`-th row holds `run` values; the rest are empty. Many rep/def levels
    /// per leaf value, which is what pushes the dense mini-block budget over its limit.
    Skewed {
        rows: usize,
        stride: usize,
        run: usize,
    },
    /// Every row is null.
    AllNull { rows: usize },
    /// `List<List<Int32>>`, which forces the nested per-chunk row mapping.
    Nested {
        rows: usize,
        outer: usize,
        inner: usize,
    },
}

impl Shape {
    fn build(&self) -> ArrayRef {
        match *self {
            Self::SingleValue => Arc::new(lists(&[1])),
            Self::Uniform { rows, run } => Arc::new(lists(&vec![run; rows])),
            Self::UniformWithBigTail { rows, run, tail } => {
                let mut counts = vec![run; rows];
                *counts.last_mut().expect("non-empty") = tail;
                Arc::new(lists(&counts))
            }
            Self::Skewed { rows, stride, run } => {
                let counts: Vec<usize> = (0..rows)
                    .map(|row| if row % stride == 0 { run } else { 0 })
                    .collect();
                Arc::new(lists(&counts))
            }
            Self::AllNull { rows } => Arc::new(null_lists(rows)),
            Self::Nested { rows, outer, inner } => Arc::new(nested_lists(rows, outer, inner)),
        }
    }
}

/// A `List<Int32>` array whose row `i` holds `counts[i]` values.
fn lists(counts: &[usize]) -> ListArray {
    let total: usize = counts.iter().sum();
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0i32);
    let mut end = 0i32;
    for count in counts {
        end += *count as i32;
        offsets.push(end);
    }
    ListArray::new(
        Arc::new(Field::new(ITEM, DataType::Int32, true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(Int32Array::from_iter_values((0..total).map(|v| v as i32))),
        None,
    )
}

/// A `List<Int32>` array where every row is null.
fn null_lists(rows: usize) -> ListArray {
    ListArray::new(
        Arc::new(Field::new(ITEM, DataType::Int32, true)),
        OffsetBuffer::new(vec![0i32; rows + 1].into()),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        Some(NullBuffer::new_null(rows)),
    )
}

/// A `List<List<Int32>>` array: `rows` rows, each holding `outer` inner lists of `inner`
/// values.
fn nested_lists(rows: usize, outer: usize, inner: usize) -> ListArray {
    let inner_array = lists(&vec![inner; rows * outer]);
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(0i32);
    for row in 0..rows {
        offsets.push(((row + 1) * outer) as i32);
    }
    ListArray::new(
        Arc::new(Field::new(
            ITEM,
            DataType::List(Arc::new(Field::new(ITEM, DataType::Int32, true))),
            true,
        )),
        OffsetBuffer::new(offsets.into()),
        Arc::new(inner_array),
        None,
    )
}

/// One benchmark input.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    /// Why this case is in the matrix; printed by the footprint report.
    pub note: &'static str,
    pub shape: Shape,
    /// Request the sparse layout explicitly rather than relying on the writer's automatic
    /// selection. Cases that exercise automatic selection deliberately leave this false.
    pub force_sparse: bool,
    /// How many identical columns to encode.
    ///
    /// The scheduler caches its per-page state once per page per column, and
    /// `encode_batch` emits one page per column here (the accumulation queue never splits a
    /// single array), so column count is the knob that multiplies resident state. A wide
    /// table is also the realistic shape for the memory pressure this measures.
    pub columns: usize,
    /// The layout this case is expected to produce.
    pub expect: ObservedLayout,
}

/// Large enough that no case in the matrix is split by the writer's page budget, so page
/// count is determined solely by `Case::columns`.
const PAGE_BUDGET: u64 = 8 * 1024 * 1024;

/// The full input matrix.
///
/// Ordered from degenerate to large so that a failure in the cheap cases surfaces first.
pub fn cases() -> Vec<Case> {
    vec![
        // -- Degenerate shapes ------------------------------------------------------
        // An all-null page never reaches the sparse layout: the writer recognises that it
        // carries no distinct leaf values and emits a constant page, which stores no chunk
        // index. It is kept in the matrix to pin that down, so a future writer change that
        // starts routing it through the sparse layout shows up as a layout assertion failure
        // rather than as a silent memory regression.
        Case {
            name: "degenerate/all_null_lists",
            note: "every row null; writer emits a constant page",
            shape: Shape::AllNull { rows: 200_000 },
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Constant,
        },
        Case {
            name: "degenerate/single_value",
            note: "one row, one value, one chunk; the smallest real sparse index",
            shape: Shape::SingleValue,
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
        Case {
            name: "degenerate/many_tiny_pages",
            note: "64 single-value pages; isolates the per-page floor",
            shape: Shape::SingleValue,
            force_sparse: true,
            columns: 64,
            expect: ObservedLayout::Sparse,
        },
        // -- Sweep over page size ---------------------------------------------------
        // A compact chunk index trades a fixed per-page cost against a per-chunk saving, so
        // the interesting question is where that trade turns positive. These three points
        // plus `degenerate/single_value` and `uniform/many_chunks` give the curve.
        Case {
            name: "sweep/16k_values",
            note: "16k leaf values in one page",
            shape: Shape::Uniform {
                rows: 2_048,
                run: 8,
            },
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
        Case {
            name: "sweep/256k_values",
            note: "256k leaf values in one page",
            shape: Shape::Uniform {
                rows: 32_768,
                run: 8,
            },
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
        // -- Chunk-index representations -------------------------------------------
        Case {
            name: "uniform/many_chunks",
            note: "equal value counts; chunk lookups reduce to arithmetic",
            shape: Shape::Uniform {
                rows: 400_000,
                run: 8,
            },
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
        Case {
            name: "non_uniform/many_chunks",
            note: "one row of a different length; forces per-row count materialization",
            shape: Shape::UniformWithBigTail {
                rows: 400_000,
                run: 8,
                tail: 500_000,
            },
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
        Case {
            name: "nested/list_of_list",
            note: "List<List<Int32>>; per-chunk row metadata is boxed",
            shape: Shape::Nested {
                rows: 100_000,
                outer: 4,
                inner: 4,
            },
            force_sparse: true,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
        // -- Wide, which is what multiplies the resident state ----------------------
        Case {
            name: "wide/32_columns",
            note: "32 sparse columns held open at once",
            shape: Shape::Uniform {
                rows: 50_000,
                run: 8,
            },
            force_sparse: true,
            columns: 32,
            expect: ObservedLayout::Sparse,
        },
        // -- Automatic selection, i.e. what real 2.3 files will do -----------------
        Case {
            name: "automatic/skewed_lists",
            note: "no metadata hint; writer picks sparse from the rep/def budget",
            shape: Shape::Skewed {
                rows: 500_000,
                stride: 64,
                run: 8,
            },
            force_sparse: false,
            columns: 1,
            expect: ObservedLayout::Sparse,
        },
    ]
}

impl Case {
    pub fn batch(&self) -> RecordBatch {
        let array = self.shape.build();
        let mut metadata = HashMap::new();
        if self.force_sparse {
            metadata.insert(
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_SPARSE.to_string(),
            );
        }
        let fields: Vec<Field> = (0..self.columns.max(1))
            .map(|i| {
                Field::new(format!("{COLUMN}{i}"), array.data_type().clone(), true)
                    .with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<ArrayRef> = vec![array; self.columns.max(1)];
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns).expect("valid batch")
    }

    pub fn encoding_options(&self) -> EncodingOptions {
        EncodingOptions {
            cache_bytes_per_column: PAGE_BUDGET,
            max_page_bytes: PAGE_BUDGET,
            version: SPARSE_VERSION,
            ..Default::default()
        }
    }
}

/// Encode a case without checking which layout the writer chose.
pub fn encode_unchecked(case: &Case) -> EncodedBatch {
    let batch = case.batch();
    let lance_schema =
        Arc::new(lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap());
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    rt.block_on(encode_batch(
        &batch,
        lance_schema,
        default_encoding_strategy(SPARSE_VERSION).as_ref(),
        &case.encoding_options(),
    ))
    .unwrap_or_else(|e| panic!("case {} failed to encode: {e}", case.name))
}

/// Encode a case and verify it produced the layout the case claims.
///
/// Panics on a layout mismatch: silently measuring the dense path would invalidate the
/// whole comparison.
pub fn encode(case: &Case) -> EncodedBatch {
    let encoded = encode_unchecked(case);
    let observed = layouts(&encoded);
    assert!(
        observed.iter().all(|l| *l == case.expect),
        "case {} expected every page to be {} but observed {:?}",
        case.name,
        case.expect,
        observed
    );
    encoded
}

/// The layout of every page in the encoded batch.
pub fn layouts(encoded: &EncodedBatch) -> Vec<ObservedLayout> {
    encoded
        .page_table
        .iter()
        .flat_map(|col| col.page_infos.iter())
        .map(|page| match &page.encoding {
            PageEncoding::Legacy(_) => ObservedLayout::Legacy,
            PageEncoding::Structural(layout) => match &layout.layout {
                Some(Layout::SparseLayout(_)) => ObservedLayout::Sparse,
                Some(Layout::MiniBlockLayout(_)) => ObservedLayout::MiniBlock,
                Some(Layout::FullZipLayout(_)) => ObservedLayout::FullZip,
                Some(Layout::ConstantLayout(_)) => ObservedLayout::Constant,
                Some(Layout::BlobLayout(_)) => ObservedLayout::Blob,
                _ => ObservedLayout::Other,
            },
        })
        .collect()
}

/// Number of leaf value slots across every sparse page, used to normalise footprint.
pub fn visible_items(encoded: &EncodedBatch) -> u64 {
    encoded
        .page_table
        .iter()
        .flat_map(|col| col.page_infos.iter())
        .filter_map(|page| match &page.encoding {
            PageEncoding::Structural(layout) => match &layout.layout {
                Some(Layout::SparseLayout(sparse)) => Some(sparse.num_visible_items),
                _ => None,
            },
            PageEncoding::Legacy(_) => None,
        })
        .sum()
}

/// Evenly spaced ascending row indices, which is the shape that drives the per-chunk
/// lookups on the scheduling and decode paths hardest.
pub fn scattered_indices(num_rows: u64, count: u64) -> Vec<u64> {
    if num_rows == 0 || count == 0 {
        return Vec::new();
    }
    let stride = (num_rows / count).max(1);
    (0..num_rows)
        .step_by(stride as usize)
        .take(count as usize)
        .collect()
}
