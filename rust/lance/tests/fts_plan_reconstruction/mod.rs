// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{Field, Schema};
use datafusion::physical_plan::empty::EmptyExec;
use lance::Dataset;
use lance::io::exec::PreFilterSource;
use lance::io::exec::fts::{FlatCombinedFieldsExec, FlatStatsScope};
use lance_index::scalar::inverted::query::{CombinedFieldsQuery, FtsSearchParams};
use roaring::RoaringBitmap;

#[tokio::test]
async fn bm25f_quoted_root_reconstruction() {
    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.values().append_value("alpha");
    builder.append(true);
    let documents = Arc::new(builder.finish());
    let root = Field::new("docs`raw", documents.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![root.clone()]));
    let batch = RecordBatch::try_new(schema.clone(), vec![documents]).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let input_schema = Arc::new(Schema::new(vec![root, lance_core::ROW_ID_FIELD.clone()]));

    // The query encodes a path, while the decoded input carries its literal root.
    FlatCombinedFieldsExec::try_new(
        Arc::new(dataset),
        CombinedFieldsQuery::try_new("alpha".into(), vec!["`docs``raw`".into()]).unwrap(),
        FtsSearchParams::default(),
        Arc::new(EmptyExec::new(input_schema)),
        RoaringBitmap::from_iter([0]),
        FlatStatsScope::WholeCorpus,
        PreFilterSource::None,
    )
    .unwrap();
}
