// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset writes of Arrow JSON columns, through the public write API.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, ListArray, RecordBatch, RecordBatchIterator, StringArray, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use lance::{Dataset, Error};
use lance_arrow::ARROW_EXT_NAME_KEY;
use lance_arrow::json::{ARROW_JSON_EXT_NAME, encode_json};
use rstest::rstest;

#[rstest]
#[case::top_level(false, "field 'extra' at input batch row 1")]
#[case::nested(
    true,
    "field 'media.extra' at input batch row 1, list element positions [1]"
)]
#[tokio::test]
async fn test_write_invalid_json_reports_field_and_row(
    #[case] nested: bool,
    #[case] location: &str,
) {
    let extra_field = ArrowField::new("extra", DataType::Utf8, true).with_metadata(
        [(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        )]
        .into(),
    );
    let extra: ArrayRef = Arc::new(StringArray::from(vec![
        Some(r#"{"codec":"h264"}"#),
        None,
        Some("invalid json {"),
    ]));
    let (field, column) = if nested {
        let item_fields = Fields::from(vec![extra_field]);
        let items = StructArray::new(item_fields.clone(), vec![extra], None);
        let item = Arc::new(ArrowField::new("item", DataType::Struct(item_fields), true));
        // Row 0 holds the first value, row 1 the null and the invalid value.
        let media = ListArray::new(
            item,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 3])),
            Arc::new(items),
            None,
        );
        (
            ArrowField::new("media", media.data_type().clone(), true),
            Arc::new(media) as ArrayRef,
        )
    } else {
        // Drop the first value so the invalid one is at row 1.
        (extra_field, extra.slice(1, 2))
    };
    let schema = Arc::new(ArrowSchema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![column]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

    let error = Dataset::write(reader, "memory://", None).await.unwrap_err();

    let parser_error = encode_json("invalid json {").unwrap_err().to_string();
    let expected = format!("Failed to encode JSON in {location}: {parser_error}");
    assert!(
        matches!(&error, Error::Arrow { message, .. } if message.contains(&expected)),
        "expected an Arrow error containing {expected:?}, got: {error:?}"
    );
}
