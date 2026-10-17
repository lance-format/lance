// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Public zone-statistics APIs used independently of a ZoneMap index.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use lance_index::scalar::{
    SargableQuery,
    seed::IndexSeedWriter,
    zoned::{ZoneBound, ZoneProcessor},
    zonemap::{ZoneMapProcessor, ZoneMapSeedWriter},
};

#[test]
fn accumulate_and_evaluate_zone_statistics() {
    let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(9)]));
    let mut processor = ZoneMapProcessor::new(DataType::Int64).unwrap();
    processor.process_chunk(&values).unwrap();
    let summary = processor
        .finish_zone(ZoneBound {
            fragment_id: 7,
            start: 0,
            length: 3,
        })
        .unwrap();
    assert_eq!(summary.min, ScalarValue::Int64(Some(3)));
    assert_eq!(summary.max, ScalarValue::Int64(Some(9)));
    assert_eq!(summary.null_count, 1);
    assert_eq!(summary.nan_count, 0);
    assert!(
        summary
            .may_match(&DataType::Int64, &SargableQuery::IsNull())
            .unwrap()
    );
    assert!(
        summary
            .may_match(
                &DataType::Int64,
                &SargableQuery::Equals(ScalarValue::Int64(Some(5))),
            )
            .unwrap()
    );
    assert!(
        !summary
            .may_match(
                &DataType::Int64,
                &SargableQuery::Equals(ScalarValue::Int64(Some(12))),
            )
            .unwrap()
    );
}

#[test]
fn decode_zone_seeds_without_opening_an_index() {
    let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(9)]));
    let mut writer = ZoneMapSeedWriter::new("value", 2, DataType::Int64).unwrap();
    writer.observe_batch(&values).unwrap();
    let bytes = writer.finish().unwrap().unwrap();
    let (zones, null_offsets) = ZoneMapSeedWriter::deserialize_seed(7, &bytes, 2).unwrap();
    assert_eq!(zones.len(), 2);
    assert_eq!(zones[0].bound.fragment_id, 7);
    assert_eq!(zones[0].bound.start, 0);
    assert_eq!(zones[0].bound.length, 2);
    assert_eq!(zones[0].min, ScalarValue::Int64(Some(3)));
    assert_eq!(zones[0].max, ScalarValue::Int64(Some(3)));
    assert_eq!(zones[0].null_count, 1);
    assert_eq!(zones[1].bound.start, 2);
    assert_eq!(zones[1].bound.length, 1);
    assert_eq!(zones[1].min, ScalarValue::Int64(Some(9)));
    assert_eq!(null_offsets.unwrap().iter().collect::<Vec<_>>(), vec![1]);
    assert!(writer.finish().unwrap().is_none());
}
