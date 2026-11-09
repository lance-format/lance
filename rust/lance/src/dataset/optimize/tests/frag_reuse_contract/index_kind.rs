// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What each index type is built on and asked, and the rows it is asked about.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexParams;
use lance_index::IndexType;
use lance_index::scalar::{
    BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
};
use lance_linalg::distance::MetricType;

use crate::index::vector::VectorIndexParams;
use crate::{Dataset, Result};

const DIM: i32 = 4;
const IVF_PARTITIONS: usize = 2;
/// Added to `v` by an in-place update, so updated rows answer different
/// queries than they did when the index was built.
const UPDATED: i32 = 1000;

#[derive(Debug, Clone, Copy)]
pub(super) enum Query {
    Filter(&'static str, fn(&Row) -> bool),
    FullText(&'static str),
    /// `k` close to the live row count makes a candidate the prefilter
    /// should have excluded show up as a missing row.
    Nearest {
        target: f32,
        k: usize,
    },
}

impl Query {
    /// The `i` of every row in `rows` the query should return, computed
    /// without an index.
    pub(super) fn expected<'a>(self, rows: impl IntoIterator<Item = &'a Row>) -> BTreeSet<i32> {
        let rows = rows.into_iter();
        match self {
            Self::Filter(_, predicate) => {
                rows.filter(|row| predicate(row)).map(|row| row.i).collect()
            }
            Self::FullText(token) => rows
                .filter(|row| row.text.split_whitespace().any(|word| word == token))
                .map(|row| row.i)
                .collect(),
            Self::Nearest { target, k } => {
                let mut by_distance: Vec<(f32, i32)> = rows
                    .map(|row| {
                        let distance = (row.vec[0] - target).powi(2)
                            + row.vec[1..].iter().map(|v| v * v).sum::<f32>();
                        (distance, row.i)
                    })
                    .collect();
                by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
                by_distance.into_iter().take(k).map(|(_, i)| i).collect()
            }
        }
    }

    /// The `i` of every row `dataset` returns for the query, through its index.
    pub(super) async fn run(self, dataset: &Dataset) -> Result<BTreeSet<i32>> {
        let mut scanner = dataset.scan();
        match self {
            Self::Filter(filter, _) => {
                scanner.filter(filter)?;
            }
            Self::FullText(token) => {
                scanner.full_text_search(FullTextSearchQuery::new(token.to_string()))?;
            }
            Self::Nearest { target, k } => {
                let mut vector = vec![0.0f32; DIM as usize];
                vector[0] = target;
                scanner
                    .nearest("vec", &Float32Array::from(vector), k)?
                    .minimum_nprobes(IVF_PARTITIONS);
            }
        }
        scanner.project(&["i"])?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch["i"].as_primitive::<Int32Type>().values();
        let unique: BTreeSet<i32> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "{self:?} returned a row twice");
        Ok(unique)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum IndexKind {
    BTree,
    Bitmap,
    ZoneMap,
    BloomFilter,
    LabelList,
    NGram,
    Inverted,
    IvfFlat,
}

impl IndexKind {
    pub(super) fn column(self) -> &'static str {
        match self {
            Self::BTree | Self::Bitmap | Self::ZoneMap | Self::BloomFilter => "v",
            Self::LabelList => "tags",
            Self::NGram | Self::Inverted => "text",
            Self::IvfFlat => "vec",
        }
    }

    /// Each kind is asked about both written and updated rows, so a query
    /// answered from stale index entries returns the wrong rows.
    pub(super) fn queries(self) -> Vec<Query> {
        match self {
            Self::BTree | Self::Bitmap | Self::ZoneMap => vec![
                Query::Filter("v = 13", |row| row.v == 13),
                Query::Filter("v >= 6 AND v < 20", |row| (6..20).contains(&row.v)),
                Query::Filter("v >= 1000", |row| row.v >= UPDATED),
            ],
            Self::BloomFilter => vec![
                Query::Filter("v = 13", |row| row.v == 13),
                Query::Filter("v IN (3, 13, 1013)", |row| [3, 13, 1013].contains(&row.v)),
            ],
            Self::LabelList => vec![
                Query::Filter("array_has_any(tags, ['t1'])", |row| row.tag == "t1"),
                Query::Filter("array_has_any(tags, ['updated'])", |row| {
                    row.tag == "updated"
                }),
            ],
            Self::NGram => vec![
                Query::Filter("contains(text, 'w1')", |row| row.text.contains("w1")),
                Query::Filter("contains(text, 'updated')", |row| {
                    row.text.contains("updated")
                }),
            ],
            Self::Inverted => vec![Query::FullText("w1"), Query::FullText("updated")],
            Self::IvfFlat => vec![
                Query::Nearest { target: 10.3, k: 5 },
                Query::Nearest {
                    target: 10.3,
                    k: 30,
                },
            ],
        }
    }

    pub(super) fn index_type(self) -> IndexType {
        match self {
            Self::BTree => IndexType::BTree,
            Self::Bitmap => IndexType::Bitmap,
            Self::ZoneMap => IndexType::ZoneMap,
            Self::BloomFilter => IndexType::BloomFilter,
            Self::LabelList => IndexType::LabelList,
            Self::NGram => IndexType::NGram,
            Self::Inverted => IndexType::Inverted,
            Self::IvfFlat => IndexType::Vector,
        }
    }

    pub(super) fn params(self) -> Box<dyn IndexParams> {
        let builtin = match self {
            Self::Inverted => return Box::new(InvertedIndexParams::default()),
            Self::IvfFlat => {
                return Box::new(VectorIndexParams::ivf_flat(IVF_PARTITIONS, MetricType::L2));
            }
            Self::BTree => BuiltinIndexType::BTree,
            Self::Bitmap => BuiltinIndexType::Bitmap,
            Self::ZoneMap => BuiltinIndexType::ZoneMap,
            Self::BloomFilter => BuiltinIndexType::BloomFilter,
            Self::LabelList => BuiltinIndexType::LabelList,
            Self::NGram => BuiltinIndexType::NGram,
        };
        Box::new(ScalarIndexParams::for_builtin(builtin))
    }
}

/// The columns the oracle evaluates. `i` is unique and never updated, so it
/// identifies a row across every rewrite; the other columns are indexed and
/// change on an in-place update.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Row {
    pub(super) i: i32,
    pub(super) v: i32,
    /// The single element of the `tags` list.
    pub(super) tag: String,
    pub(super) text: String,
    pub(super) vec: Vec<f32>,
}

impl Row {
    pub(super) fn written(i: i32) -> Self {
        Self {
            i,
            v: i,
            tag: format!("t{}", i % 4),
            text: format!("w{} x{}", i % 7, i % 3),
            vec: vec![i as f32, 0.0, 0.0, 0.0],
        }
    }

    pub(super) fn updated(i: i32) -> Self {
        Self {
            i,
            v: UPDATED + i,
            tag: "updated".to_string(),
            text: format!("updated x{}", i % 3),
            // An updated vector moves next to the query target, so it
            // changes which rows are nearest.
            vec: vec![10.3 + 0.01 * i as f32, 0.0, 0.0, 0.0],
        }
    }

    /// `rows` as a batch, plus a `pad` column no query reads: an update
    /// that leaves it out of the patch rewrites columns in place.
    pub(super) fn to_batch<'a>(rows: impl IntoIterator<Item = &'a Self>) -> RecordBatch {
        let rows: Vec<&Self> = rows.into_iter().collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("pad", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                false,
            ),
        ]));
        let ids = Int32Array::from_iter_values(rows.iter().map(|row| row.i));
        let mut tags = ListBuilder::new(StringBuilder::new());
        for row in &rows {
            tags.values().append_value(&row.tag);
            tags.append(true);
        }
        let vectors =
            Float32Array::from_iter_values(rows.iter().flat_map(|row| row.vec.iter().copied()));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids.clone()),
                Arc::new(ids),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.v))),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|row| row.text.as_str()),
                )),
                Arc::new(tags.finish()),
                Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM).unwrap()),
            ],
        )
        .unwrap()
    }

    pub(super) fn from_batch(batch: &RecordBatch) -> Vec<Self> {
        let ids = batch["i"].as_primitive::<Int32Type>();
        let values = batch["v"].as_primitive::<Int32Type>();
        let texts = batch["text"].as_string::<i32>();
        let tags = batch["tags"].as_list::<i32>();
        let vectors = batch["vec"].as_fixed_size_list();
        (0..batch.num_rows())
            .map(|row| Self {
                i: ids.value(row),
                v: values.value(row),
                tag: tags.value(row).as_string::<i32>().value(0).to_string(),
                text: texts.value(row).to_string(),
                vec: vectors
                    .value(row)
                    .as_primitive::<Float32Type>()
                    .values()
                    .to_vec(),
            })
            .collect()
    }
}
