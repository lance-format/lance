// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-column widths measured from the batches a scan has already produced.
//!
//! A schema fixes no width for strings, binary, lists, maps or dictionaries, so
//! [`super::utils::estimated_bytes_per_row`] reports nothing for them. This module
//! measures what they actually cost, using `get_array_memory_size` -- the same
//! quantity DataFusion reads from the other side of a join.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::{Stream, StreamExt};

use crate::session::caches::DSMetadataCache;
use lance_core::cache::{CacheKey, LanceCache};
use lance_core::deepsize::DeepSizeOf;

/// Arrow bytes and rows observed for one column, summed over batches.
///
/// A running total, not an average of per-batch rates: a batch's per-array
/// overhead is fixed, so small batches would skew the average.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColumnBytes {
    bytes: u64,
    rows: u64,
}

impl ColumnBytes {
    /// Adds what one batch spent on one column.
    pub fn observe(&mut self, bytes: u64, rows: u64) {
        self.bytes += bytes;
        self.rows += rows;
    }

    /// Arrow bytes per row, or `None` when nothing has been observed.
    pub fn bytes_per_row(&self) -> Option<f64> {
        (self.rows > 0).then(|| self.bytes as f64 / self.rows as f64)
    }
}

/// A width measured from data, in Arrow bytes per row.
///
/// A later scan replaces this rather than adding to it; it measured a different
/// slice of the column, not more of this one.
#[derive(Clone, Copy, Debug, DeepSizeOf, PartialEq)]
pub struct MeasuredWidth(pub f64);

/// Cache key for one column's measured width. `DSMetadataCache` supplies the
/// dataset URI prefix.
///
/// Keyed by version because a manifest version is immutable, so an entry can never
/// go stale. Named by column rather than field id because [`measure`] works on
/// whole top-level columns and the node holds a bare Arrow schema.
///
/// `layout` is the column's Arrow type. One name can carry different types within a
/// version -- a partially projected struct, or a blob read as a descriptor rather
/// than its payload -- and a width measured on one says nothing about the other.
#[derive(Debug)]
pub struct MeasuredWidthKey<'a> {
    pub version: u64,
    pub column: &'a str,
    pub layout: &'a DataType,
}

impl CacheKey for MeasuredWidthKey<'_> {
    type ValueType = MeasuredWidth;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "column-width/{}/{}/{:?}",
            self.version, self.column, self.layout
        ))
    }

    fn type_name() -> &'static str {
        "MeasuredWidth"
    }
}

/// Widths already measured for the top-level columns a node reports on.
///
/// Resolved once when the node is built, since `partition_statistics` is sync.
/// Empty on a cold cache, which leaves every column to what its schema fixes.
#[derive(Clone, Debug, Default)]
pub struct MeasuredWidths(HashMap<String, f64>);

/// Identifies a column by name and Arrow layout, which together are what a
/// measured width describes.
fn identity(field: &arrow_schema::Field) -> String {
    format!("{}|{:?}", field.name(), field.data_type())
}

impl MeasuredWidths {
    /// The width measured for `field`, if one has been for this exact layout.
    pub fn get(&self, field: &arrow_schema::Field) -> Option<f64> {
        self.0.get(&identity(field)).copied()
    }

    /// True when nothing has been measured, so every column falls back.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<(arrow_schema::Field, f64)> for MeasuredWidths {
    fn from_iter<I: IntoIterator<Item = (arrow_schema::Field, f64)>>(widths: I) -> Self {
        Self(
            widths
                .into_iter()
                .map(|(field, width)| (identity(&field), width))
                .collect(),
        )
    }
}

/// The Arrow bytes each column of `batch` occupies, paired with its row count.
pub fn measure(batch: &RecordBatch) -> Vec<(u64, u64)> {
    let rows = batch.num_rows() as u64;
    batch
        .columns()
        .iter()
        .map(|column| (column.get_array_memory_size() as u64, rows))
        .collect()
}

/// Wraps a batch stream so what it decodes is measured and recorded on completion.
///
/// Records only when `representative`: a read that filtered, ranged or
/// index-selected its rows measured a sample, and a width published from one is
/// reused by later full scans. A scan abandoned early or failed records nothing
/// either, so a recorded width always describes a whole column.
pub fn measuring<S, E>(
    stream: S,
    cache: Arc<DSMetadataCache>,
    version: u64,
    schema: Arc<ArrowSchema>,
    representative: bool,
) -> impl Stream<Item = Result<RecordBatch, E>>
where
    S: Stream<Item = Result<RecordBatch, E>> + Send + Unpin + 'static,
    E: Send + 'static,
{
    let totals = vec![ColumnBytes::default(); schema.fields().len()];
    futures::stream::unfold(Some((stream, totals)), move |state| {
        let cache = cache.clone();
        let schema = schema.clone();
        async move {
            let (mut inner, mut totals) = state?;
            match inner.next().await {
                Some(Ok(batch)) => {
                    for (total, (bytes, rows)) in totals.iter_mut().zip(measure(&batch)) {
                        total.observe(bytes, rows);
                    }
                    Some((Ok(batch), Some((inner, totals))))
                }
                Some(Err(error)) => Some((Err(error), None)),
                None => {
                    // A read that selected rows saw a sample, not the column. Its
                    // width would be published as the column's and reused by a full
                    // scan, which is how a narrow filtered read under-reports a wide
                    // column by orders of magnitude.
                    if representative {
                        record(&cache, version, &schema, &totals).await;
                    }
                    None
                }
            }
        }
    })
}

/// Writes what a completed scan measured into the dataset's cache, keeping the
/// larger of an existing and an incoming width. A column that produced no rows
/// records nothing rather than recording a zero.
pub async fn record(
    cache: &LanceCache,
    version: u64,
    schema: &ArrowSchema,
    totals: &[ColumnBytes],
) {
    for (field, total) in schema.fields().iter().zip(totals) {
        let Some(width) = total.bytes_per_row() else {
            continue;
        };
        let key = MeasuredWidthKey {
            version,
            column: field.name(),
            layout: field.data_type(),
        };
        // Two observations of one column can disagree -- partitions of a read, or
        // successive reads of different rows -- and a plain insert lets whichever
        // finished last speak for the column. Keep the larger: an over-reported
        // side is merely not collected, while an under-reported one is collected
        // and holds memory proportional to the data it actually carries.
        let width = match cache.get_with_key(&key).await {
            Some(seen) => width.max(seen.0),
            None => width,
        };
        cache
            .insert_with_key(&key, Arc::new(MeasuredWidth(width)))
            .await;
    }
}

/// Reads back the widths already measured for `fields` of this dataset version.
/// A field whose layout differs from the measured one misses, which is correct:
/// the width described the other layout.
pub async fn resolve<'a>(
    cache: &LanceCache,
    version: u64,
    fields: impl Iterator<Item = &'a arrow_schema::Field>,
) -> MeasuredWidths {
    let mut widths = HashMap::new();
    for field in fields {
        let key = MeasuredWidthKey {
            version,
            column: field.name(),
            layout: field.data_type(),
        };
        if let Some(measured) = cache.get_with_key(&key).await {
            widths.insert(identity(field), measured.0);
        }
    }
    MeasuredWidths(widths)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    fn test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "bb", "ccc"])),
            ],
        )
        .unwrap()
    }

    /// The point of measuring: a string column has a real width, and it is the one
    /// the batch actually occupies rather than anything the schema could predict.
    #[test]
    fn a_string_column_is_measured_from_the_batch() {
        let batch = test_batch();

        let measured = measure(&batch);

        assert_eq!(measured.len(), 2, "one entry per column");
        let (name_bytes, rows) = measured[1];
        assert_eq!(rows, 3);
        assert_eq!(name_bytes, batch.column(1).get_array_memory_size() as u64);
        assert!(name_bytes > 0, "a string column must not measure as free");
    }

    /// Batches of different sizes must combine into one width over their total rows.
    /// The rates here differ on purpose -- 10 B/row then 2 B/row -- so that an
    /// unweighted average of the two, 6.0, cannot pass for the right answer.
    #[test]
    fn widths_accumulate_across_batches_by_total_rows() {
        let mut column = ColumnBytes::default();
        column.observe(1000, 100);
        column.observe(600, 300);

        assert_eq!(column.bytes_per_row(), Some(4.0));
    }

    /// A width belongs to the version and the field it was measured on. Reading it
    /// back for any other one must miss: a later version may have rewritten the
    /// column, and a different field is simply different data.
    #[tokio::test]
    async fn a_measured_width_is_scoped_to_its_version_and_field() {
        let cache = lance_core::cache::LanceCache::with_capacity(1024 * 1024);
        let name = arrow_schema::Field::new("name", DataType::Utf8, false);
        let key = MeasuredWidthKey {
            version: 3,
            column: name.name(),
            layout: name.data_type(),
        };
        cache
            .insert_with_key(&key, std::sync::Arc::new(MeasuredWidth(12.5)))
            .await;

        let hit = cache.get_with_key(&key).await;
        assert_eq!(hit.map(|width| width.0), Some(12.5));

        for miss in [
            MeasuredWidthKey {
                version: 4,
                column: name.name(),
                layout: name.data_type(),
            },
            MeasuredWidthKey {
                version: 3,
                column: "title",
                layout: name.data_type(),
            },
        ] {
            assert!(
                cache.get_with_key(&miss).await.is_none(),
                "{miss:?} must not read back the width measured for {key:?}"
            );
        }
    }

    /// Two observations of one column share a cache key: partitions of a read, or
    /// successive reads over different rows. A plain insert lets the last finisher
    /// speak for the column, so a stream of 4 KB values would be replaced by one of
    /// single characters. The larger width wins, because under-reporting is what
    /// gets a large side collected.
    #[tokio::test]
    async fn a_narrow_observation_does_not_shrink_a_wider_one() {
        use futures::TryStreamExt;
        let cache = Arc::new(DSMetadataCache(LanceCache::with_capacity(1024 * 1024)));
        let field = arrow_schema::Field::new("text", DataType::Utf8, false);
        let schema = Arc::new(ArrowSchema::new(vec![field.clone()]));
        let wide = "x".repeat(4_096);
        for value in [wide.as_str(), "a"] {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(arrow_array::StringArray::from(vec![value; 10]))],
            )
            .unwrap();
            measuring(
                futures::stream::iter(vec![Ok::<_, ()>(batch)]),
                cache.clone(),
                3,
                schema.clone(),
                true,
            )
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        }
        let cached = resolve(&cache, 3, [&field].into_iter())
            .await
            .get(&field)
            .unwrap();
        assert!(
            cached > 1_000.0,
            "the 4 KB stream's width must survive the narrow one, got {cached}"
        );
    }

    /// One column name can carry different Arrow layouts within a dataset version
    /// -- a partly projected struct, or a blob read as a descriptor rather than its
    /// payload. A width measured on one says nothing about the other, so reading it
    /// back under a different layout must miss.
    #[tokio::test]
    async fn a_width_is_not_reused_across_layouts() {
        use arrow_schema::Field;

        let cache = lance_core::cache::LanceCache::with_capacity(1024 * 1024);
        let descriptor = Field::new("blob", DataType::UInt64, false);
        let payload = Field::new("blob", DataType::LargeBinary, false);

        cache
            .insert_with_key(
                &MeasuredWidthKey {
                    version: 1,
                    column: descriptor.name(),
                    layout: descriptor.data_type(),
                },
                std::sync::Arc::new(MeasuredWidth(8.0)),
            )
            .await;

        let same = resolve(&cache, 1, [&descriptor].into_iter()).await;
        assert_eq!(same.get(&descriptor), Some(8.0));

        let other = resolve(&cache, 1, [&payload].into_iter()).await;
        assert_eq!(
            other.get(&payload),
            None,
            "a descriptor's width must not be billed as a payload's"
        );
    }

    /// Dividing by a zero row count yields infinity or NaN, and either one would be
    /// billed as a width. Nothing observed means nothing to report.
    #[test]
    fn an_unobserved_column_has_no_width() {
        assert_eq!(ColumnBytes::default().bytes_per_row(), None);

        let mut empty = ColumnBytes::default();
        empty.observe(4096, 0);
        assert_eq!(empty.bytes_per_row(), None);
    }
}
