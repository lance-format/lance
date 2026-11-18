// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{
    Array, ArrayRef, Int32Array, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance::io::exec::PreFilterSource;
use lance::io::exec::fts::{FlatCombinedFieldsExec, FlatStatsScope};
use lance_arrow::ARROW_EXT_NAME_KEY;
use lance_arrow::json::{ARROW_JSON_EXT_NAME, JSON_EXT_NAME, JsonArray, JsonEncoding, json_field};
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::query::{CombinedFieldsQuery, FtsQuery, FtsSearchParams};
use lance_index::scalar::inverted::{Language, tokenizer::InvertedIndexParams};
use roaring::RoaringBitmap;
use rstest::rstest;

#[rstest]
#[case::jsonb(DataType::LargeBinary)]
#[case::arrow_json(DataType::Utf8)]
#[case::large_arrow_json(DataType::LargeUtf8)]
#[tokio::test]
async fn bm25f_json_append_preserves_matches(#[case] storage: DataType) {
    let docs = [
        Some(r#"{"title":"alpha"}"#),
        Some(r#"{"title":"noise"}"#),
        Some(r#"{"title":"alpha"}"#),
        None,
    ];
    let other = [
        Some(r#"{"title":"alpha"}"#),
        None,
        Some(r#"{"title":"noise"}"#),
        Some("{}"),
    ];
    let values = |items: &[Option<&str>]| -> ArrayRef {
        match storage {
            DataType::LargeBinary => Arc::new(
                JsonArray::try_from_iter(items.iter().copied())
                    .unwrap()
                    .into_inner(),
            ),
            DataType::Utf8 => Arc::new(StringArray::from(items.to_vec())),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(items.to_vec())),
            _ => unreachable!(),
        }
    };
    let field = |name| match storage {
        DataType::LargeBinary => json_field(name, true),
        _ => Field::new(name, storage.clone(), true).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_owned(),
            ARROW_JSON_EXT_NAME.to_owned(),
        )])),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        field("doc"),
        field("other"),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
            values(&docs),
            values(&other),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.slice(0, 2))], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let params = InvertedIndexParams::new("simple".into(), Language::English)
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(None);
    for column in ["doc", "other"] {
        dataset
            .create_index(&[column], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
    }
    let search = |dataset: &Dataset| {
        let mut scanner = dataset.scan();
        scanner.project(&["id"]).unwrap();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(FtsQuery::CombinedFields(
                CombinedFieldsQuery::try_new(
                    "title,str,alpha".into(),
                    vec!["doc".into(), "other".into()],
                )
                .unwrap(),
            )))
            .unwrap();
        scanner
    };
    let scores = |batch: &RecordBatch| -> BTreeMap<i32, f32> {
        let ids = batch["id"].as_any().downcast_ref::<Int32Array>().unwrap();
        let scores = batch["_score"]
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap();
        ids.values()
            .iter()
            .copied()
            .zip(scores.values().iter().copied())
            .collect()
    };
    assert_eq!(
        scores(&search(&dataset).try_into_batch().await.unwrap())
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![0]
    );

    dataset
        .append(
            RecordBatchIterator::new([Ok(batch.slice(2, 2))], schema),
            None,
        )
        .await
        .unwrap();
    let plan = search(&dataset).create_plan().await.unwrap();
    let mut flat_nodes = 0;
    plan.apply(|node| {
        if let Some(flat) = node.downcast_ref::<FlatCombinedFieldsExec>() {
            flat_nodes += 1;
            let input = flat.children()[0].schema();
            for column in ["doc", "other"] {
                // Stored JSON reaches the flat converter with its JSONB metadata.
                assert_eq!(
                    JsonEncoding::of_field(input.field_with_name(column).unwrap()),
                    Some(JsonEncoding::Jsonb)
                );
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    assert_eq!(flat_nodes, 1);
    let mixed = scores(&search(&dataset).try_into_batch().await.unwrap());
    assert_eq!(mixed.keys().copied().collect::<Vec<_>>(), vec![0, 2]);

    for column in ["doc", "other"] {
        dataset
            .create_index(&[column], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
    }
    let indexed = scores(&search(&dataset).try_into_batch().await.unwrap());
    assert_eq!(
        mixed.keys().collect::<Vec<_>>(),
        indexed.keys().collect::<Vec<_>>()
    );
    for (id, score) in mixed {
        assert!(
            (score - indexed[&id]).abs() < 1e-6,
            "row {id}: {score} != {}",
            indexed[&id]
        );
    }
}

#[rstest]
#[case::plain_large_binary(DataType::LargeBinary, None)]
#[case::wrong_extension(DataType::LargeBinary, Some("other.binary"))]
#[case::wrong_storage(DataType::Binary, Some(JSON_EXT_NAME))]
#[tokio::test]
async fn bm25f_json_rejects_non_json_binary(
    #[case] storage: DataType,
    #[case] extension: Option<&str>,
) {
    let documents = Arc::new(
        JsonArray::try_from_iter([Some(r#"{"title":"alpha"}"#)])
            .unwrap()
            .into_inner(),
    );
    let schema = Arc::new(Schema::new(vec![json_field("doc", true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![documents]).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let mut field = Field::new("doc", storage, true);
    if let Some(extension) = extension {
        field = field.with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.into(),
            extension.into(),
        )]));
    }
    let input = Arc::new(EmptyExec::new(Arc::new(Schema::new(vec![
        lance_core::ROW_ID_FIELD.clone(),
        field,
    ]))));
    let error = FlatCombinedFieldsExec::try_new(
        Arc::new(dataset),
        CombinedFieldsQuery::try_new("title,str,alpha".into(), vec!["doc".into()]).unwrap(),
        FtsSearchParams::default(),
        input,
        RoaringBitmap::from_iter([0]),
        FlatStatsScope::WholeCorpus,
        PreFilterSource::None,
    )
    .unwrap_err();
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("incompatible type"), "{error}");
}

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
