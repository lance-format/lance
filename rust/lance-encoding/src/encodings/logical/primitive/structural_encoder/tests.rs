// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::types::{Int8Type, Int32Type};
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
    PrimitiveArray, StringArray, UInt16Array, new_null_array,
};
use arrow_buffer::{BooleanBuffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field as ArrowField};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use prost::Message;

use crate::EncodingsIo;
use crate::buffer::LanceBuffer;
use crate::compression::DefaultDecompressionStrategy;
use crate::constants::{
    COMPRESSION_LEVEL_META_KEY, COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY,
    DICT_SIZE_RATIO_META_KEY, DICT_VALUES_COMPRESSION_LEVEL_META_KEY,
    DICT_VALUES_COMPRESSION_META_KEY, STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_MINIBLOCK,
};
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::decoder::PageEncoding;
use crate::encodings::logical::primitive::dict::dictionary_encode;
use crate::encodings::logical::primitive::miniblock::{MiniBlockChunk, MiniBlockCompressed};
use crate::encodings::logical::primitive::{MiniBlockScheduler, StructuralPageScheduler};
use crate::format::metadata;
use crate::format::metadata::compressive_encoding::Compression;
use crate::repdef::build_control_word_iterator;
use crate::testing::{
    TestCases, TestEncoding, check_round_trip_encoding_of_data,
    check_round_trip_encoding_of_data_with_expected,
};

use super::{DictionaryEncodingPolicy, MiniblockChunkSize, PrimitiveStructuralEncoder};

#[test]
fn test_is_narrow() {
    let int8_array = Int8Array::from(vec![1, 2, 3]);
    let array_ref: ArrayRef = Arc::new(int8_array);
    let block = DataBlock::from_array(array_ref);

    assert!(PrimitiveStructuralEncoder::is_narrow(&block));

    let string_array = StringArray::from(vec![Some("hello"), Some("world")]);
    let block = DataBlock::from_array(string_array);
    assert!(PrimitiveStructuralEncoder::is_narrow(&block));

    let string_array = StringArray::from(vec![
        Some("hello world".repeat(100)),
        Some("world".to_string()),
    ]);
    let block = DataBlock::from_array(string_array);
    assert!((!PrimitiveStructuralEncoder::is_narrow(&block)));
}

fn valued_dictionary() -> ArrayRef {
    Arc::new(
        DictionaryArray::<Int32Type>::try_new(
            PrimitiveArray::<Int32Type>::from(vec![0]),
            Arc::new(StringArray::from(vec!["a"])),
        )
        .unwrap(),
    )
}

fn empty_dictionary() -> ArrayRef {
    new_null_array(
        &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        1,
    )
}

fn null_valued_dictionary() -> ArrayRef {
    Arc::new(
        DictionaryArray::<Int32Type>::try_new(
            PrimitiveArray::<Int32Type>::from(vec![0]),
            Arc::new(StringArray::from(vec![None::<&str>])),
        )
        .unwrap(),
    )
}

fn sliced_empty_dictionary_with_hidden_key_payload() -> ArrayRef {
    let keys = PrimitiveArray::<Int32Type>::new(
        ScalarBuffer::from(vec![5, 7, 9]),
        Some(NullBuffer::new(BooleanBuffer::new_unset(3))),
    );
    let values = Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef;
    let dictionary = DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();
    Arc::new(dictionary.slice(1, 1))
}

#[rstest::rstest]
#[case::empty_after_value(vec![valued_dictionary(), empty_dictionary()])]
#[case::empty_before_value(vec![empty_dictionary(), valued_dictionary()])]
#[case::null_value_after_value(vec![valued_dictionary(), null_valued_dictionary()])]
#[case::hidden_sliced_after_value(vec![
    valued_dictionary(),
    sliced_empty_dictionary_with_hidden_key_payload(),
])]
#[tokio::test]
async fn test_mixed_valued_and_all_null_dictionary_chunks(#[case] dictionaries: Vec<ArrayRef>) {
    check_round_trip_encoding_of_data(
        dictionaries,
        &TestCases::default()
            .with_structural_encodings()
            .with_page_sizes(vec![1])
            .with_range(0..1)
            .with_range(1..2)
            .with_indices(vec![0])
            .with_indices(vec![1]),
        HashMap::new(),
    )
    .await;
}

#[tokio::test]
async fn test_sliced_empty_dictionary_with_hidden_key_payload() {
    check_round_trip_encoding_of_data(
        vec![sliced_empty_dictionary_with_hidden_key_payload()],
        &TestCases::default()
            .with_structural_encodings()
            .with_page_sizes(vec![4096]),
        HashMap::new(),
    )
    .await;
}

#[test]
fn test_fullzip_fixed_rejects_non_byte_aligned_values() {
    let fixed = FixedWidthDataBlock {
        data: LanceBuffer::from(vec![0_u8]),
        bits_per_value: 1,
        num_values: 8,
        block_info: BlockInfo::new(),
    };
    let repdef = build_control_word_iterator(None, 0, None, 0, u16::MAX, 8);

    let Err(err) = PrimitiveStructuralEncoder::serialize_full_zip_fixed(fixed, repdef, 8) else {
        panic!("expected full-zip to reject 1-bit fixed-width values");
    };
    assert!(
        err.to_string().contains("byte aligned"),
        "unexpected error: {err}"
    );
}

// https://github.com/lance-format/lance/issues/8558
#[tokio::test]
async fn test_take_multichunk_string_lists() {
    let mut list_builder = ListBuilder::new(StringBuilder::new());
    for row in 0..3 {
        for item in 0..512 {
            list_builder
                .values()
                .append_value(format!("r{row:03}i{item:06}xaaaa"));
        }
        list_builder.append(true);
    }
    let list_array: ArrayRef = Arc::new(list_builder.finish());

    let mut metadata = HashMap::new();
    metadata.insert(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
    );
    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU16)
        .with_indices(vec![0, 2]);

    check_round_trip_encoding_of_data(vec![list_array], &test_cases, metadata).await;
}

async fn test_minichunk_size_helper(
    string_data: Vec<Option<String>>,
    minichunk_size: u64,
    encodings: &[TestEncoding],
) {
    use crate::constants::MINICHUNK_SIZE_META_KEY;
    use crate::testing::{TestCases, check_round_trip_encoding_of_data};
    use arrow_array::{ArrayRef, StringArray};
    use std::sync::Arc;

    let string_array: ArrayRef = Arc::new(StringArray::from(string_data));

    let mut metadata = HashMap::new();
    metadata.insert(
        MINICHUNK_SIZE_META_KEY.to_string(),
        minichunk_size.to_string(),
    );
    metadata.insert(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
    );

    let test_cases = TestCases::default()
        .with_encodings(encodings.iter().copied())
        .with_batch_size(1000);

    check_round_trip_encoding_of_data(vec![string_array], &test_cases, metadata).await;
}

#[tokio::test]
async fn test_minichunk_size_roundtrip() {
    // Test that minichunk size can be configured and works correctly in round-trip encoding
    let mut string_data = Vec::new();
    for i in 0..100 {
        string_data.push(Some(format!("test_string_{}", i).repeat(50)));
    }
    // configure minichunk size to 64 bytes (smaller than the default 4kb) for Lance 2.1
    test_minichunk_size_helper(
        string_data,
        64,
        &[
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse,
        ],
    )
    .await;
}

#[tokio::test]
async fn test_minichunk_size_128kb_v2_2() {
    // Test that minichunk size can be configured to 128KB and works correctly with Lance 2.2
    let mut string_data = Vec::new();
    // create a 500kb string array
    for i in 0..10000 {
        string_data.push(Some(format!("test_string_{}", i).repeat(50)));
    }
    test_minichunk_size_helper(
        string_data,
        128 * 1024,
        &[TestEncoding::StructuralU32, TestEncoding::StructuralSparse],
    )
    .await;
}

#[tokio::test]
async fn test_binary_large_minichunk_size_over_max_miniblock_values() {
    let mut string_data = Vec::new();
    // 128kb/chunk / 6 bytes (t_9999) = 21845 items per chunk
    for i in 0..10000 {
        string_data.push(Some(format!("t_{}", i)));
    }
    test_minichunk_size_helper(
        string_data,
        128 * 1024,
        &[TestEncoding::StructuralU32, TestEncoding::StructuralSparse],
    )
    .await;
}

#[tokio::test]
async fn test_large_dictionary_general_compression() {
    use arrow_array::{ArrayRef, StringArray};
    use std::collections::HashMap;
    use std::sync::Arc;

    // Create large string dictionary data (>32KiB) with low cardinality
    // Use 100 unique strings, each 500 bytes long = 50KB dictionary
    let unique_values: Vec<String> = (0..100)
        .map(|i| format!("value_{:04}_{}", i, "x".repeat(500)))
        .collect();

    // Repeat these strings many times to create a large array
    let repeated_strings: Vec<_> = unique_values
        .iter()
        .cycle()
        .take(10_000)
        .map(|s| Some(s.as_str()))
        .collect();

    let string_array = Arc::new(StringArray::from(repeated_strings)) as ArrayRef;

    // Configure test to use V2_2 and verify encoding
    let test_cases = TestCases::default()
        .with_u32_structural_encodings()
        .with_verify_encoding(Arc::new(|cols: &[crate::encoder::EncodedColumn], _| {
            assert_eq!(cols.len(), 1);
            let col = &cols[0];

            // Navigate to the dictionary encoding in the page layout
            if let Some(PageEncoding::Structural(page_layout)) =
                &col.final_pages.first().map(|p| &p.description)
                && let Some(metadata::page_layout::Layout::MiniBlockLayout(mini_block)) =
                    &page_layout.layout
                && let Some(dictionary_encoding) = &mini_block.dictionary
            {
                match dictionary_encoding.compression.as_ref() {
                    Some(Compression::General(general)) => {
                        // Verify it's using LZ4 or Zstd
                        let compression = general.compression.as_ref().unwrap();
                        assert!(
                            compression.scheme()
                                == metadata::CompressionScheme::CompressionAlgorithmLz4
                                || compression.scheme()
                                    == metadata::CompressionScheme::CompressionAlgorithmZstd,
                            "Expected LZ4 or Zstd compression for large dictionary"
                        );
                    }
                    _ => panic!("Expected General compression for large dictionary"),
                }
            }
        }));

    check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
}

fn dictionary_encoding_from_page(
    page: &crate::encoder::EncodedPage,
) -> &crate::format::metadata::CompressiveEncoding {
    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural page encoding");
    };
    let metadata::page_layout::Layout::MiniBlockLayout(layout) = layout.layout.as_ref().unwrap()
    else {
        panic!("Expected mini-block layout");
    };
    layout
        .dictionary
        .as_ref()
        .unwrap_or_else(|| panic!("Expected dictionary encoding"))
}

async fn encode_variable_dict_page(
    metadata: HashMap<String, String>,
) -> crate::encoder::EncodedPage {
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, DictionaryArray, Int32Array, StringArray};

    let values = Arc::new(StringArray::from(
        (0..128)
            .map(|i| format!("value_{i:04}_{}", "x".repeat(256)))
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let keys = Int32Array::from_iter_values((0..20_000).map(|i| i % 128));
    let dict_array =
        Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap()) as ArrayRef;

    let field = arrow_schema::Field::new(
        "dict_col",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        false,
    )
    .with_metadata(metadata);

    encode_first_page(field, dict_array, TestEncoding::StructuralU32).await
}

async fn encode_auto_fixed_dict_page(
    metadata: HashMap<String, String>,
) -> crate::encoder::EncodedPage {
    use arrow_array::{ArrayRef, Decimal128Array};

    // 128-bit fixed-width values with low cardinality to trigger dictionary encoding.
    let values = (0..20_000)
        .map(|i| match i % 3 {
            0 => 10_i128,
            1 => 20_i128,
            _ => 30_i128,
        })
        .collect::<Vec<_>>();
    let decimal = Decimal128Array::from_iter_values(values)
        .with_precision_and_scale(38, 0)
        .unwrap();
    let decimal = Arc::new(decimal) as ArrayRef;

    let mut field_metadata = metadata;
    // Strongly encourage dictionary encoding for this synthetic test data.
    field_metadata.insert(
        "lance-encoding:dict-size-ratio".to_string(),
        "0.99".to_string(),
    );
    let field = arrow_schema::Field::new("fixed_col", DataType::Decimal128(38, 0), false)
        .with_metadata(field_metadata);

    encode_first_page(field, decimal, TestEncoding::StructuralU32).await
}

#[tokio::test]
async fn test_dict_values_general_compression_default_lz4_for_variable_dict_values() {
    let page = encode_variable_dict_page(HashMap::new()).await;
    let dictionary_encoding = dictionary_encoding_from_page(&page);
    let Some(Compression::General(general)) = dictionary_encoding.compression.as_ref() else {
        panic!("Expected General compression for dictionary values");
    };
    let compression = general.compression.as_ref().unwrap();
    assert_eq!(
        compression.scheme(),
        metadata::CompressionScheme::CompressionAlgorithmLz4
    );
}

#[tokio::test]
async fn test_dict_values_general_compression_default_lz4_for_fixed_dict_values() {
    let page = encode_auto_fixed_dict_page(HashMap::new()).await;
    let dictionary_encoding = dictionary_encoding_from_page(&page);
    let Some(Compression::General(general)) = dictionary_encoding.compression.as_ref() else {
        panic!("Expected General compression for dictionary values");
    };
    let compression = general.compression.as_ref().unwrap();
    assert_eq!(
        compression.scheme(),
        metadata::CompressionScheme::CompressionAlgorithmLz4
    );
}

#[tokio::test]
async fn test_dict_values_general_compression_zstd() {
    let mut metadata = HashMap::new();
    metadata.insert(
        DICT_VALUES_COMPRESSION_META_KEY.to_string(),
        "zstd".to_string(),
    );
    let page = encode_variable_dict_page(metadata).await;
    let dictionary_encoding = dictionary_encoding_from_page(&page);
    let Some(Compression::General(general)) = dictionary_encoding.compression.as_ref() else {
        panic!("Expected General compression for dictionary values");
    };
    let compression = general.compression.as_ref().unwrap();
    assert_eq!(
        compression.scheme(),
        metadata::CompressionScheme::CompressionAlgorithmZstd
    );
}

#[tokio::test]
async fn test_dict_values_general_compression_none() {
    let mut metadata = HashMap::new();
    metadata.insert(
        DICT_VALUES_COMPRESSION_META_KEY.to_string(),
        "none".to_string(),
    );
    let page = encode_variable_dict_page(metadata).await;
    let dictionary_encoding = dictionary_encoding_from_page(&page);
    assert!(
        !matches!(
            dictionary_encoding.compression.as_ref(),
            Some(Compression::General(_))
        ),
        "Expected dictionary values to avoid General compression"
    );
}

#[test]
fn test_resolve_dict_values_compression_metadata_defaults_to_lz4() {
    let metadata = PrimitiveStructuralEncoder::resolve_dict_values_compression_metadata(
        &HashMap::new(),
        None,
        None,
    );
    assert_eq!(metadata.get(COMPRESSION_META_KEY), Some(&"lz4".to_string()),);
    assert!(!metadata.contains_key(COMPRESSION_LEVEL_META_KEY));
}

#[test]
fn test_resolve_dict_values_compression_metadata_metadata_overrides_env() {
    let field_metadata = HashMap::from([
        (
            DICT_VALUES_COMPRESSION_META_KEY.to_string(),
            "none".to_string(),
        ),
        (
            DICT_VALUES_COMPRESSION_LEVEL_META_KEY.to_string(),
            "7".to_string(),
        ),
    ]);
    let metadata = PrimitiveStructuralEncoder::resolve_dict_values_compression_metadata(
        &field_metadata,
        Some("zstd".to_string()),
        Some("3".to_string()),
    );
    assert_eq!(
        metadata.get(COMPRESSION_META_KEY),
        Some(&"none".to_string()),
    );
    assert_eq!(
        metadata.get(COMPRESSION_LEVEL_META_KEY),
        Some(&"7".to_string()),
    );
}

#[test]
fn test_resolve_dict_values_compression_metadata_env_fallback() {
    let metadata = PrimitiveStructuralEncoder::resolve_dict_values_compression_metadata(
        &HashMap::new(),
        Some("zstd".to_string()),
        Some("9".to_string()),
    );
    assert_eq!(
        metadata.get(COMPRESSION_META_KEY),
        Some(&"zstd".to_string()),
    );
    assert_eq!(
        metadata.get(COMPRESSION_LEVEL_META_KEY),
        Some(&"9".to_string()),
    );
}

#[tokio::test]
async fn test_dictionary_encode_int64() {
    use crate::constants::{DICT_SIZE_RATIO_META_KEY, STRUCTURAL_ENCODING_META_KEY};
    use crate::testing::{TestCases, check_round_trip_encoding_of_data};
    use arrow_array::{ArrayRef, Int64Array};
    use std::collections::HashMap;
    use std::sync::Arc;

    // Low cardinality with poor RLE opportunity.
    let values = (0..1000)
        .map(|i| match i % 3 {
            0 => 10i64,
            1 => 20i64,
            _ => 30i64,
        })
        .collect::<Vec<_>>();
    let array = Arc::new(Int64Array::from(values)) as ArrayRef;

    let mut metadata = HashMap::new();
    metadata.insert(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
    );
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.99".to_string());

    let test_cases = TestCases::default()
        .with_u32_structural_encodings()
        .with_batch_size(1000)
        .with_range(0..1000)
        .with_indices(vec![0, 1, 10, 999])
        .with_expected_encoding("dictionary");

    check_round_trip_encoding_of_data(vec![array], &test_cases, metadata).await;
}

#[tokio::test]
async fn test_dictionary_encode_float64() {
    use crate::constants::{DICT_SIZE_RATIO_META_KEY, STRUCTURAL_ENCODING_META_KEY};
    use crate::testing::{TestCases, check_round_trip_encoding_of_data};
    use arrow_array::{ArrayRef, Float64Array};
    use std::collections::HashMap;
    use std::sync::Arc;

    // Low cardinality with poor RLE opportunity.
    let values = (0..1000)
        .map(|i| match i % 3 {
            0 => 0.1f64,
            1 => 0.2f64,
            _ => 0.3f64,
        })
        .collect::<Vec<_>>();
    let array = Arc::new(Float64Array::from(values)) as ArrayRef;

    let mut metadata = HashMap::new();
    metadata.insert(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
    );
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.99".to_string());

    let test_cases = TestCases::default()
        .with_u32_structural_encodings()
        .with_batch_size(1000)
        .with_range(0..1000)
        .with_indices(vec![0, 1, 10, 999])
        .with_expected_encoding("dictionary");

    check_round_trip_encoding_of_data(vec![array], &test_cases, metadata).await;
}

#[rstest::rstest]
#[case::dense(false)]
#[case::nullable(true)]
#[tokio::test]
async fn test_planar_dictionary_encode_repeated_float64(#[case] nullable: bool) {
    let num_values = 262_144u64;
    // Spread repeated values across the sample blocks instead of repeating the same
    // short prefix at every block boundary.
    let values = (0..num_values)
        .map(|i| {
            let value = (((i * 7919) ^ (i >> 7)) % 2048) as f64 / 100.0;
            (!nullable || i % 8 != 0).then_some(value)
        })
        .collect::<Vec<_>>();
    let array = Arc::new(Float64Array::from(values)) as ArrayRef;
    let block = DataBlock::from_array(array.clone()).remove_outer_validity();
    let ratio = PrimitiveStructuralEncoder::sample_unique_ratio(&block, 4096)
        .unwrap()
        .unwrap();
    assert!(ratio < 0.98);
    assert!(
        ratio * num_values as f64 > 100_000.0,
        "the old sample projection must reject this 2,048-value dictionary"
    );
    let field = lance_core::datatypes::Field::try_from(&arrow_schema::Field::new(
        "values",
        DataType::Float64,
        nullable,
    ))
    .unwrap();
    let budget = PrimitiveStructuralEncoder::should_dictionary_encode(
        &block,
        &field,
        DictionaryEncodingPolicy::Planar,
    )
    .expect("repeated values should reach dictionary construction");
    assert_eq!(budget.max_dict_entries, 100_000);

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralSparse)
        .with_page_sizes(vec![8 * 1024 * 1024])
        .with_ingest_batch_counts([1])
        .with_slicing_modes([false])
        .with_batch_size(65_536)
        .with_range(0..num_values)
        .with_indices(vec![0, 1, 127, 128, 65_535, 65_536, num_values - 1])
        .with_expected_encoding("dictionary");
    check_round_trip_encoding_of_data(vec![array], &test_cases, HashMap::new()).await;
}

#[rstest::rstest]
#[case::dense(false, false, false, false, "lz4")]
#[case::null_vectors(true, false, false, false, "lz4")]
#[case::null_children(false, true, false, false, "lz4")]
#[case::nested(true, true, true, false, "lz4")]
#[case::sparse(true, true, false, true, "lz4")]
#[case::uncompressed(false, true, false, false, "none")]
#[case::zstd(false, false, false, false, "zstd")]
#[tokio::test]
async fn test_planar_dictionary_encode_repeated_vectors(
    #[case] null_vectors: bool,
    #[case] null_children: bool,
    #[case] nested: bool,
    #[case] sparse: bool,
    #[case] compression: &str,
) {
    const ROWS: usize = 2048;
    const DIMENSION: usize = 512;
    let values = Float32Array::from(
        (0..ROWS * DIMENSION)
            .map(|i| {
                let row = (i / DIMENSION) % 17;
                let column = i % DIMENSION;
                (!null_children || !(row + column).is_multiple_of(7))
                    .then_some((row * DIMENSION + column) as f32 / 10.0)
            })
            .collect::<Vec<_>>(),
    );
    let mut array: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new("item", DataType::Float32, true)),
        if nested { DIMENSION / 2 } else { DIMENSION } as i32,
        Arc::new(values),
        nested.then(|| {
            NullBuffer::from(
                (0..ROWS * 2)
                    .map(|i| !i.is_multiple_of(9))
                    .collect::<Vec<_>>(),
            )
        }),
    ));
    if nested {
        array = Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new(
                "item",
                array.data_type().clone(),
                true,
            )),
            2,
            array,
            None,
        ));
    }
    if null_vectors {
        let data = array
            .to_data()
            .into_builder()
            .nulls(Some(NullBuffer::from(
                (0..ROWS).map(|i| !i.is_multiple_of(8)).collect::<Vec<_>>(),
            )))
            .build()
            .unwrap();
        array = arrow_array::make_array(data);
    }
    let mut metadata = HashMap::from([(
        DICT_VALUES_COMPRESSION_META_KEY.to_string(),
        compression.to_string(),
    )]);
    if sparse {
        metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            "sparse".to_string(),
        );
    }
    let tests = TestCases::default()
        .with_encoding(TestEncoding::StructuralSparse)
        .with_page_sizes(vec![8 * 1024 * 1024])
        .with_ingest_batch_counts([1, 2])
        .with_slicing_modes([false])
        .with_batch_size(256)
        .with_range(0..ROWS as u64)
        .with_indices(vec![0, 1, 7, 8, 127, 128, 1023, 2047])
        .with_expected_encoding("dictionary");
    check_round_trip_encoding_of_data(vec![array], &tests, metadata).await;
}

#[rstest::rstest]
#[case::index_budget(32_768, 64, 100, 8192)]
#[case::item_budget(2048, 64, 100, 512)]
#[case::narrow_vectors(2048, 16, 400, 512)]
#[tokio::test]
async fn test_sparse_vector_dictionary_hidden_payload(
    #[case] rows: usize,
    #[case] dimension: usize,
    #[case] valid: usize,
    #[case] patterns: usize,
    #[values("lz4", "none")] compression: &str,
) {
    // Arrow retains arbitrary child values beneath null vectors. These repeated
    // hidden values must not make a dictionary look cheaper than sparse storage.
    let values = Float32Array::from(
        (0..rows * dimension)
            .map(|i| {
                let row = i / dimension;
                let pattern = if row < valid { 0 } else { 1 + row % patterns };
                let mut bits = (pattern * dimension + i % dimension) as u32;
                bits ^= bits >> 16;
                bits = bits.wrapping_mul(0x7feb_352d);
                bits ^= bits >> 15;
                bits = bits.wrapping_mul(0x846c_a68b);
                bits ^= bits >> 16;
                f32::from_bits(0x3f80_0000 | (bits & 0x007f_ffff))
            })
            .collect::<Vec<_>>(),
    );
    let array = Arc::new(FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new("item", DataType::Float32, true)),
        dimension as i32,
        Arc::new(values),
        Some(NullBuffer::from(
            (0..rows).map(|i| i < valid).collect::<Vec<_>>(),
        )),
    )) as ArrayRef;
    let mut metadata = HashMap::from([(
        DICT_VALUES_COMPRESSION_META_KEY.to_string(),
        compression.to_string(),
    )]);
    // Narrow vectors need an explicit sparse request; wide mostly-null vectors
    // must also be protected when the writer selects sparse storage automatically.
    if dimension < 64 {
        metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            "sparse".to_string(),
        );
    }
    let field = arrow_schema::Field::new("vectors", array.data_type().clone(), true)
        .with_metadata(metadata.clone());
    let candidate = encode_first_page(field, array.clone(), TestEncoding::StructuralSparse).await;
    let mut fallback_metadata = metadata.clone();
    fallback_metadata.insert(DICT_DIVISOR_META_KEY.to_string(), "1000000".to_string());
    let field = arrow_schema::Field::new("vectors", array.data_type().clone(), true)
        .with_metadata(fallback_metadata);
    let fallback = encode_first_page(field, array.clone(), TestEncoding::StructuralSparse).await;
    let candidate_bytes: usize = candidate.data.iter().map(|buffer| buffer.len()).sum();
    let fallback_bytes: usize = fallback.data.iter().map(|buffer| buffer.len()).sum();
    assert!(
        candidate_bytes <= fallback_bytes,
        "dictionary candidate {candidate_bytes} bytes exceeds sparse fallback {fallback_bytes} bytes"
    );
    for (page, is_candidate) in [(&candidate, true), (&fallback, false)] {
        let PageEncoding::Planar(layout) = &page.description else {
            panic!("expected a planar page");
        };
        match layout
            .values
            .as_ref()
            .and_then(|values| values.codec.as_ref())
        {
            Some(crate::format::pb23::value_plane::Codec::Zipped(_)) => {}
            Some(crate::format::pb23::value_plane::Codec::Chunked(values)) => {
                assert_eq!(values.dictionary.is_some(), is_candidate && dimension < 64);
            }
            other => panic!("expected a non-dictionary fallback, got {other:?}"),
        }
    }
    let tests = TestCases::default()
        .with_encoding(TestEncoding::StructuralSparse)
        .with_page_sizes(vec![32 * 1024 * 1024])
        .with_ingest_batch_counts([1])
        .with_slicing_modes([false])
        .with_batch_size(256)
        .with_range(0..rows as u64)
        .with_indices(vec![0, valid as u64 - 1, valid as u64, rows as u64 - 1]);
    check_round_trip_encoding_of_data(vec![array], &tests, metadata).await;
}

#[rstest::rstest]
#[case::rgb("u8", false)]
#[case::depth("u16", true)]
#[case::dense_depth("u16_dense", false)]
#[case::nested_nullable("nested", true)]
#[case::null_children("null_children", false)]
#[tokio::test]
async fn test_tensor_per_value_compression(
    #[case] kind: &str,
    #[case] sparse: bool,
    #[values("zstd", "lz4", "none")] compression: &str,
) {
    const ROWS: usize = 32;
    const DIMENSION: usize = 1030;
    let child: ArrayRef = match kind {
        "u16_dense" => Arc::new(arrow_array::UInt16Array::from_iter_values(
            (0..ROWS * DIMENSION).map(|i| (i / DIMENSION + i % 5) as u16),
        )),
        "u8" => Arc::new(arrow_array::UInt8Array::from_iter_values(
            (0..ROWS * DIMENSION).map(|i| (i / DIMENSION + i % 5) as u8),
        )),
        "null_children" => Arc::new(arrow_array::UInt16Array::new_null(ROWS * DIMENSION)),
        _ => Arc::new(arrow_array::UInt16Array::from_iter(
            (0..ROWS * DIMENSION).map(|i| (i % 7 != 0).then_some((i / DIMENSION + i % 5) as u16)),
        )),
    };
    let dimension = if kind == "nested" {
        DIMENSION / 2
    } else {
        DIMENSION
    };
    let mut array: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new(
            "item",
            child.data_type().clone(),
            true,
        )),
        dimension as i32,
        child,
        None,
    ));
    if kind == "nested" {
        array = Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new(
                "item",
                array.data_type().clone(),
                true,
            )),
            2,
            array,
            None,
        ));
    }
    if sparse {
        array = arrow_array::make_array(
            array
                .to_data()
                .into_builder()
                .nulls(Some(NullBuffer::from(
                    (0..ROWS).map(|i| i % 4 == 0).collect::<Vec<_>>(),
                )))
                .build()
                .unwrap(),
        );
    }
    let metadata = HashMap::from([
        (COMPRESSION_META_KEY.to_string(), compression.to_string()),
        (COMPRESSION_LEVEL_META_KEY.to_string(), "3".to_string()),
        // Exercise the configured codec independently of automatic selection.
        ("lance-encoding:axis-delta".to_string(), "off".to_string()),
    ]);
    let field = arrow_schema::Field::new("tensor", array.data_type().clone(), true)
        .with_metadata(metadata.clone());
    let page = encode_first_page(field, array.clone(), TestEncoding::StructuralSparse).await;
    let PageEncoding::Planar(layout) = page.description else {
        panic!("expected Planar page")
    };
    let Some(crate::format::pb23::value_plane::Codec::Zipped(values)) =
        layout.values.and_then(|v| v.codec)
    else {
        panic!("expected independently addressed tensors");
    };
    assert_eq!(
        matches!(
            values.addressing,
            Some(crate::format::pb23::zipped_values::Addressing::BitsPerOffset(_))
        ),
        compression != "none"
    );
    assert_eq!(
        matches!(
            values.compression.as_ref().unwrap().compression,
            Some(Compression::General(_))
        ),
        compression != "none"
    );
    let tests = TestCases::default()
        .with_encoding(TestEncoding::StructuralSparse)
        .with_page_sizes(vec![8 * 1024])
        .with_ingest_batch_counts([4])
        .with_slicing_modes([true])
        .with_batch_size(7)
        .with_range(3..29)
        .with_indices(vec![0, 3, 4, 7, 8, 17, 28, 31]);
    check_round_trip_encoding_of_data(vec![array], &tests, metadata).await;
}

#[rstest::rstest]
#[case::unique(2048, 2048, 2)]
#[case::entry_budget(128, 64, 8)]
fn test_planar_vector_dictionary_rejects_unhelpful_candidates(
    #[case] rows: usize,
    #[case] distinct: usize,
    #[case] divisor: u64,
) {
    let values = Float64Array::from(
        (0..rows)
            .flat_map(|i| [(i % distinct) as f64, 1.0])
            .collect::<Vec<_>>(),
    );
    let array = Arc::new(FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new("item", DataType::Float64, false)),
        2,
        Arc::new(values),
        None,
    )) as ArrayRef;
    let field =
        arrow_schema::Field::new("vectors", array.data_type().clone(), false).with_metadata(
            HashMap::from([(DICT_DIVISOR_META_KEY.to_string(), divisor.to_string())]),
        );
    let field = lance_core::datatypes::Field::try_from(&field).unwrap();
    assert!(
        PrimitiveStructuralEncoder::try_dictionary_encode(
            &DataBlock::from_array(array),
            &field,
            DictionaryEncodingPolicy::Planar,
            rows as u64,
        )
        .unwrap()
        .is_none()
    );
}

// Dictionary encoding decision tests
fn create_test_fixed_data_block(
    num_values: u64,
    cardinality: u64,
    bits_per_value: u64,
) -> DataBlock {
    assert!(cardinality > 0);
    assert!(cardinality <= num_values);
    let block_info = BlockInfo::default();

    assert_eq!(bits_per_value % 8, 0);
    let data = match bits_per_value {
        32 => {
            let values = (0..num_values)
                .map(|i| (i % cardinality) as u32)
                .collect::<Vec<_>>();
            crate::buffer::LanceBuffer::reinterpret_vec(values)
        }
        64 => {
            let values = (0..num_values).map(|i| i % cardinality).collect::<Vec<_>>();
            crate::buffer::LanceBuffer::reinterpret_vec(values)
        }
        128 => {
            let values = (0..num_values)
                .map(|i| (i % cardinality) as u128)
                .collect::<Vec<_>>();
            crate::buffer::LanceBuffer::reinterpret_vec(values)
        }
        _ => unreachable!(),
    };
    DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value,
        data,
        num_values,
        block_info,
    })
}

/// Helper to create VariableWidth (string) test data block with exact cardinality
fn create_test_variable_width_block(num_values: u64, cardinality: u64) -> DataBlock {
    use arrow_array::StringArray;

    assert!(cardinality <= num_values && cardinality > 0);

    let mut values = Vec::with_capacity(num_values as usize);
    for i in 0..num_values {
        values.push(format!("value_{:016}", i % cardinality));
    }

    let array = StringArray::from(values);
    DataBlock::from_array(Arc::new(array) as ArrayRef)
}

fn create_sorted_string_array(num_values: u64, cardinality: u64) -> ArrayRef {
    use arrow_array::StringArray;

    assert!(cardinality <= num_values && cardinality > 0);

    let mut values = Vec::with_capacity(num_values as usize);
    for i in 0..num_values {
        let value_idx = i * cardinality / num_values;
        values.push(format!("value_{:016}", value_idx));
    }

    Arc::new(StringArray::from(values)) as ArrayRef
}

fn create_sorted_variable_width_block(num_values: u64, cardinality: u64) -> DataBlock {
    DataBlock::from_array(create_sorted_string_array(num_values, cardinality))
}

#[test]
fn test_should_dictionary_encode() {
    use crate::constants::DICT_SIZE_RATIO_META_KEY;
    use lance_core::datatypes::Field as LanceField;

    // Create data where dict encoding saves space
    let block = create_test_variable_width_block(1000, 10);

    let mut metadata = HashMap::new();
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.8".to_string());
    let arrow_field =
        arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    let field = LanceField::try_from(&arrow_field).unwrap();

    let result = PrimitiveStructuralEncoder::should_dictionary_encode(
        &block,
        &field,
        DictionaryEncodingPolicy::LegacyU16,
    );

    assert!(
        result.is_some(),
        "Should use dictionary encode based on size"
    );
}

#[tokio::test]
async fn test_encode_skips_dictionary_above_max_dict_size() {
    use crate::constants::DICT_MAX_SIZE_META_KEY;

    // 2000 values of 64 bytes with 500 distinct: the dictionary alone is 32 KiB,
    // well within the size ratio, so only the byte cap can veto it.
    let array = Arc::new(StringArray::from_iter_values(
        (0..2000).map(|i| format!("{:064}", i % 500)),
    )) as ArrayRef;

    let field = arrow_schema::Field::new("test", DataType::Utf8, false);
    let page = encode_first_page(field, array.clone(), TestEncoding::StructuralU32).await;
    assert!(
        miniblock_layout(&page).dictionary.is_some(),
        "a 32 KiB dictionary must be used under the default cap"
    );

    let metadata = HashMap::from([(DICT_MAX_SIZE_META_KEY.to_string(), "4096".to_string())]);
    let field = arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    let page = encode_first_page(field, array, TestEncoding::StructuralU32).await;
    assert!(
        miniblock_layout(&page).dictionary.is_none(),
        "a dictionary over lance-encoding:dict-max-size must not be written"
    );
}

#[tokio::test]
#[should_panic(expected = "dict-max-size is 8MiB")]
async fn test_encode_rejects_unparsable_max_dict_size() {
    use crate::constants::DICT_MAX_SIZE_META_KEY;

    let array = Arc::new(StringArray::from_iter_values(
        (0..2000).map(|i| format!("{:064}", i % 500)),
    )) as ArrayRef;
    let metadata = HashMap::from([(DICT_MAX_SIZE_META_KEY.to_string(), "8MiB".to_string())]);
    let field = arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    encode_first_page(field, array, TestEncoding::StructuralU32).await;
}

#[test]
fn test_block_sampling_detects_low_cardinality_in_short_sorted_runs() {
    let sample_count: usize = 4096;
    let num_values: u64 = 200_000;
    let cardinality: u64 = 8_000;
    let run_length = num_values / cardinality;
    let stride = num_values as usize / sample_count;
    assert!(
        stride > run_length as usize,
        "test must construct the stride > run_length case"
    );

    let block = create_sorted_variable_width_block(num_values, cardinality);
    let sample_unique_ratio =
        PrimitiveStructuralEncoder::sample_unique_ratio(&block, sample_count).unwrap();

    assert!(
        sample_unique_ratio.is_some_and(|ratio| ratio < 0.98),
        "sorted low-cardinality data must not be classified as near-unique"
    );
}

#[test]
fn test_should_dictionary_encode_sorted_low_cardinality() {
    use crate::constants::DICT_SIZE_RATIO_META_KEY;
    use lance_core::datatypes::Field as LanceField;

    let block = create_sorted_variable_width_block(200_000, 8_000);

    let mut metadata = HashMap::new();
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.8".to_string());
    let arrow_field =
        arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    let field = LanceField::try_from(&arrow_field).unwrap();

    let result = PrimitiveStructuralEncoder::should_dictionary_encode(
        &block,
        &field,
        DictionaryEncodingPolicy::LegacyU32,
    );

    assert!(
        result.is_some(),
        "sorted low-cardinality data should reach dictionary encoding"
    );
}

#[test]
fn test_should_not_dictionary_encode_sorted_high_cardinality_short_runs() {
    use crate::constants::DICT_SIZE_RATIO_META_KEY;
    use lance_core::datatypes::Field as LanceField;

    let num_values = 200_002;
    let cardinality = 100_001;
    let block = create_sorted_variable_width_block(num_values, cardinality);

    let mut metadata = HashMap::new();
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.8".to_string());
    let arrow_field =
        arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    let field = LanceField::try_from(&arrow_field).unwrap();

    let result = PrimitiveStructuralEncoder::should_dictionary_encode(
        &block,
        &field,
        DictionaryEncodingPolicy::LegacyU32,
    );

    assert!(
        result.is_none(),
        "sorted high-cardinality short runs should not trigger a full dictionary probe"
    );
}

#[tokio::test]
async fn test_encode_sorted_low_cardinality_uses_dictionary_layout() {
    use crate::constants::DICT_SIZE_RATIO_META_KEY;

    let mut metadata = HashMap::new();
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.8".to_string());
    let field = arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    let array = create_sorted_string_array(200_000, 8_000);

    let page = encode_first_page(field, array, TestEncoding::StructuralU32).await;
    let _ = dictionary_encoding_from_page(&page);
}

#[test]
fn test_should_not_dictionary_encode_unsupported_bits() {
    use crate::constants::DICT_SIZE_RATIO_META_KEY;
    use lance_core::datatypes::Field as LanceField;

    let block = create_test_fixed_data_block(1000, 1000, 32);

    let mut metadata = HashMap::new();
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.8".to_string());
    let arrow_field =
        arrow_schema::Field::new("test", DataType::Int32, false).with_metadata(metadata);
    let field = LanceField::try_from(&arrow_field).unwrap();

    let result = PrimitiveStructuralEncoder::should_dictionary_encode(
        &block,
        &field,
        DictionaryEncodingPolicy::LegacyU16,
    );

    assert!(
        result.is_none(),
        "Should not use dictionary encode for unsupported bit width"
    );
}

#[rstest::rstest]
#[case::legacy_u16(DictionaryEncodingPolicy::LegacyU16)]
#[case::legacy_u32(DictionaryEncodingPolicy::LegacyU32)]
#[case::planar(DictionaryEncodingPolicy::Planar)]
fn test_should_not_dictionary_encode_near_unique_sample(#[case] policy: DictionaryEncodingPolicy) {
    use crate::constants::DICT_SIZE_RATIO_META_KEY;
    use lance_core::datatypes::Field as LanceField;

    let num_values = 5000;
    let block = create_test_variable_width_block(num_values, num_values);

    let mut metadata = HashMap::new();
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "1.0".to_string());
    let arrow_field =
        arrow_schema::Field::new("test", DataType::Utf8, false).with_metadata(metadata);
    let field = LanceField::try_from(&arrow_field).unwrap();

    let result = PrimitiveStructuralEncoder::should_dictionary_encode(&block, &field, policy);

    assert!(
        result.is_none(),
        "Should not probe dictionary encoding for near-unique data"
    );
}

#[rstest::rstest]
#[case::cardinality(8, 1.0)]
#[case::encoded_size(2, 0.25)]
fn test_planar_dictionary_probe_preserves_budgets(#[case] divisor: u64, #[case] size_ratio: f64) {
    let block = create_test_fixed_data_block(4096, 2048, 64);
    let field =
        arrow_schema::Field::new("values", DataType::UInt64, false).with_metadata(HashMap::from([
            (DICT_DIVISOR_META_KEY.to_string(), divisor.to_string()),
            (DICT_SIZE_RATIO_META_KEY.to_string(), size_ratio.to_string()),
        ]));
    let field = lance_core::datatypes::Field::try_from(&field).unwrap();
    let budget = PrimitiveStructuralEncoder::should_dictionary_encode(
        &block,
        &field,
        DictionaryEncodingPolicy::Planar,
    )
    .expect("a repeated sample should reach bounded dictionary construction");
    assert_eq!(budget.max_dict_entries, (4096 / divisor) as u32);
    assert_eq!(
        budget.max_encoded_size,
        (block.data_size() as f64 * size_ratio) as usize
    );
    assert!(
        dictionary_encode(
            &block,
            budget.max_dict_entries,
            budget.max_encoded_size,
            budget.max_dict_bytes,
        )
        .is_none(),
        "actual construction must abort at the unchanged budget"
    );
    assert!(dictionary_encode(&block, 4096, block.data_size() as usize, usize::MAX).is_some());
}

#[test]
fn test_v2_1_miniblock_serializes_log_num_values_15() {
    let miniblocks = MiniBlockCompressed {
        data: vec![LanceBuffer::from(vec![1_u8; 16])],
        chunks: vec![
            MiniBlockChunk {
                buffer_sizes: vec![8],
                log_num_values: 15,
            },
            MiniBlockChunk {
                buffer_sizes: vec![8],
                log_num_values: 0,
            },
        ],
        num_values: 32_769,
    };

    let serialized = PrimitiveStructuralEncoder::serialize_miniblocks(
        miniblocks,
        None,
        None,
        MiniblockChunkSize::U16,
    )
    .unwrap();

    let chunk_metadata = serialized.metadata.borrow_to_typed_slice::<u16>();
    assert_eq!(chunk_metadata.len(), 2);
    assert_eq!(
        chunk_metadata[0] & 0x0F,
        15,
        "V2.1 metadata should use all 4 bits for log_num_values"
    );
}

async fn encode_first_page(
    field: arrow_schema::Field,
    array: ArrayRef,
    version: TestEncoding,
) -> crate::encoder::EncodedPage {
    use crate::repdef::RepDefBuilder;
    use crate::{
        encoder::{
            ColumnIndexSequence, EncodingOptions, MIN_PAGE_BUFFER_ALIGNMENT, OutOfLineBuffers,
        },
        testing::{create_test_field_encoder, test_encoding_strategy},
    };

    let lance_field = lance_core::datatypes::Field::try_from(&field).unwrap();
    let encoding_strategy = test_encoding_strategy(version);
    let mut column_index_seq = ColumnIndexSequence::default();
    let encoding_options = EncodingOptions {
        cache_bytes_per_column: 1,
        max_page_bytes: 32 * 1024 * 1024,
        keep_original_array: true,
        buffer_alignment: MIN_PAGE_BUFFER_ALIGNMENT,
    };

    let mut encoder = create_test_field_encoder(
        encoding_strategy.as_ref(),
        &lance_field,
        &mut column_index_seq,
        &encoding_options,
    )
    .unwrap();

    let mut external_buffers = OutOfLineBuffers::new(0, MIN_PAGE_BUFFER_ALIGNMENT);
    let repdef = RepDefBuilder::default();
    let num_rows = array.len() as u64;
    let mut pages = Vec::new();
    for task in encoder
        .maybe_encode(array, &mut external_buffers, repdef, 0, num_rows)
        .unwrap()
    {
        pages.push(task.await.unwrap());
    }
    for task in encoder.flush(&mut external_buffers).unwrap() {
        pages.push(task.await.unwrap());
    }
    pages.into_iter().next().unwrap()
}

fn variable_offset_test_field() -> arrow_schema::Field {
    arrow_schema::Field::new("c", DataType::Utf8, false).with_metadata(HashMap::from([
        (
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
        ),
        (COMPRESSION_META_KEY.to_string(), "none".to_string()),
        (DICT_DIVISOR_META_KEY.to_string(), "100000".to_string()),
    ]))
}

fn variable_offset_test_array(lengths: &[usize], num_rows: usize) -> ArrayRef {
    Arc::new(StringArray::from_iter_values((0..num_rows).map(|index| {
        let length = lengths[index % lengths.len()];
        format!("{index:04x}{}", "x".repeat(length - 4))
    })))
}

fn miniblock_layout(page: &crate::encoder::EncodedPage) -> metadata::MiniBlockLayout {
    let layout = match &page.description {
        PageEncoding::Structural(layout) => layout.clone(),
        PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
        _ => panic!("expected structural page encoding"),
    };
    let Some(metadata::page_layout::Layout::MiniBlockLayout(layout)) = layout.layout.as_ref()
    else {
        panic!("expected mini-block page layout");
    };
    layout.clone()
}

fn variable_offset_compression(
    page: &crate::encoder::EncodedPage,
) -> metadata::compressive_encoding::Compression {
    let layout = miniblock_layout(page);
    let value_encoding = layout.value_compression.as_ref().unwrap();
    let Compression::Variable(variable) = value_encoding.compression.as_ref().unwrap() else {
        panic!("expected Variable value compression");
    };
    variable
        .offsets
        .as_deref()
        .and_then(|offsets| offsets.compression.as_ref())
        .unwrap()
        .clone()
}

fn variable_value_wire_bytes(page: &crate::encoder::EncodedPage) -> usize {
    miniblock_layout(page)
        .value_compression
        .as_ref()
        .unwrap()
        .encoded_len()
        + page.data.iter().map(LanceBuffer::len).sum::<usize>()
}

#[derive(Debug)]
struct OffsetIoRecorder {
    data: Bytes,
    requests: Mutex<Vec<Vec<Range<u64>>>>,
}

impl OffsetIoRecorder {
    fn take_requests(&self) -> Vec<Vec<Range<u64>>> {
        std::mem::take(&mut *self.requests.lock().unwrap())
    }
}

impl EncodingsIo for OffsetIoRecorder {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        _priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<Bytes>>> {
        self.requests.lock().unwrap().push(ranges.clone());
        let data = ranges
            .into_iter()
            .map(|range| self.data.slice(range.start as usize..range.end as usize))
            .collect();
        std::future::ready(Ok(data)).boxed()
    }
}

async fn miniblock_take_request_shape(page: &crate::encoder::EncodedPage) -> [usize; 6] {
    let mut position = 0_u64;
    let buffer_offsets_and_sizes = page
        .data
        .iter()
        .map(|buffer| {
            let size = buffer.len() as u64;
            let descriptor = (position, size);
            position += size;
            descriptor
        })
        .collect::<Vec<_>>();
    let mut bytes = Vec::with_capacity(position as usize);
    for buffer in &page.data {
        bytes.extend_from_slice(buffer);
    }

    let recorder = Arc::new(OffsetIoRecorder {
        data: Bytes::from(bytes),
        requests: Mutex::new(Vec::new()),
    });
    let io: Arc<dyn EncodingsIo> = recorder.clone();
    let decompression = DefaultDecompressionStrategy::default();
    let mut cold = MiniBlockScheduler::try_new(
        &buffer_offsets_and_sizes,
        0,
        page.num_rows,
        &miniblock_layout(page),
        &decompression,
    )
    .unwrap();
    let initialization = cold.init_layout().unwrap();
    let ranges = initialization.required_ranges().cloned().collect();
    let buffers = io.submit_request(ranges, 0).await.unwrap();
    let buffers = initialization.with_buffers(buffers).unwrap();
    let cached = cold.init_from_buffers(buffers, &io).await.unwrap();
    cold.try_load(&cached).unwrap();
    let initialize_requests = recorder.take_requests();

    let cold_tasks = cold.schedule_ranges(&[123..124], &io).unwrap();
    let cold_requests = recorder.take_requests();
    for task in cold_tasks {
        task.decoder_fut.await.unwrap();
    }

    let mut warm = MiniBlockScheduler::try_new(
        &buffer_offsets_and_sizes,
        0,
        page.num_rows,
        &miniblock_layout(page),
        &decompression,
    )
    .unwrap();
    warm.try_load(&cached).unwrap();
    let warm_tasks = warm.schedule_ranges(&[123..124], &io).unwrap();
    let warm_requests = recorder.take_requests();
    for task in warm_tasks {
        task.decoder_fut.await.unwrap();
    }

    [
        initialize_requests.len(),
        initialize_requests.iter().map(Vec::len).sum(),
        cold_requests.len(),
        cold_requests.iter().map(Vec::len).sum(),
        warm_requests.len(),
        warm_requests.iter().map(Vec::len).sum(),
    ]
}

#[tokio::test]
async fn test_v2_3_generic_offsets_preserve_legacy_and_take_io() {
    let array = variable_offset_test_array(&[4, 10, 5, 8], 10_000);
    let v2_1 = encode_first_page(
        variable_offset_test_field(),
        array.clone(),
        TestEncoding::StructuralU16,
    )
    .await;
    let v2_2 = encode_first_page(
        variable_offset_test_field(),
        array.clone(),
        TestEncoding::StructuralU32,
    )
    .await;
    let v2_3 = encode_first_page(
        variable_offset_test_field(),
        array,
        TestEncoding::StructuralSparse,
    )
    .await;

    assert!(matches!(
        variable_offset_compression(&v2_1),
        Compression::Flat(_)
    ));
    assert!(matches!(
        variable_offset_compression(&v2_2),
        Compression::Flat(_)
    ));
    assert!(matches!(
        variable_offset_compression(&v2_3),
        Compression::Delta(_)
    ));
    assert!(variable_value_wire_bytes(&v2_3) < variable_value_wire_bytes(&v2_2));
    assert_eq!(miniblock_layout(&v2_2).num_buffers, 1);
    assert_eq!(miniblock_layout(&v2_3).num_buffers, 2);

    let legacy_requests = miniblock_take_request_shape(&v2_2).await;
    let generic_requests = miniblock_take_request_shape(&v2_3).await;
    assert_eq!(legacy_requests, [1, 1, 1, 1, 1, 1]);
    assert_eq!(generic_requests, legacy_requests);
}

#[tokio::test]
async fn test_constant_layout_out_of_line_fixed_size_binary_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    let val = vec![0xABu8; 33];
    let arr: ArrayRef = Arc::new(
        arrow_array::FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            std::iter::repeat_n(Some(val.as_slice()), 256),
            33,
        )
        .unwrap(),
    );
    let field = arrow_schema::Field::new("c", DataType::FixedSizeBinary(33), true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected constant layout in slot 2");
    };
    assert!(layout.inline_value.is_none());
    assert_eq!(page.data.len(), 1);

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_out_of_line_utf8_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    let arr: ArrayRef = Arc::new(arrow_array::StringArray::from_iter_values(
        std::iter::repeat_n("hello", 512),
    ));
    let field = arrow_schema::Field::new("c", DataType::Utf8, true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected constant layout in slot 2");
    };
    assert!(layout.inline_value.is_none());
    assert_eq!(page.data.len(), 1);

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_null_padded_large_value_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    // One large non-null value plus many nulls still qualifies for constant
    // encoding.  Materializing the scalar for every slot (including nulls)
    // would require 32 MiB * 1000 = 32 GiB, overflowing i32 utf8 offsets
    // even within a single decode task.
    let big = "x".repeat(32 * 1024 * 1024);
    let mut values: Vec<Option<&str>> = vec![None; 1000];
    values[17] = Some(big.as_str());
    let arr: ArrayRef = Arc::new(arrow_array::StringArray::from(values));
    let field = arrow_schema::Field::new("c", DataType::Utf8, true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected constant layout in slot 2");
    };
    assert!(layout.inline_value.is_none());

    let test_cases = TestCases::default().with_encoding(TestEncoding::StructuralU32);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_null_padded_binary_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    let big = vec![0xCDu8; 32 * 1024 * 1024];
    let mut values: Vec<Option<&[u8]>> = vec![None; 1200];
    values[0] = Some(big.as_slice());
    values[599] = Some(big.as_slice());
    let arr: ArrayRef = Arc::new(arrow_array::BinaryArray::from(values));
    let field = arrow_schema::Field::new("c", DataType::Binary, true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected constant layout in slot 2");
    };
    assert!(layout.inline_value.is_none());

    let test_cases = TestCases::default().with_encoding(TestEncoding::StructuralU32);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_nullable_item_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    let arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![
        Some(7),
        None,
        Some(7),
        None,
        Some(7),
    ]));
    let field = arrow_schema::Field::new("c", DataType::Int32, true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected constant layout in slot 2");
    };
    assert!(layout.inline_value.is_some());
    assert_eq!(page.data.len(), 2);

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_list_repdef_v2_2() {
    use crate::format::metadata::page_layout::Layout;
    use arrow_array::builder::{Int32Builder, ListBuilder};

    let mut builder = ListBuilder::new(Int32Builder::new());
    builder.values().append_value(7);
    builder.values().append_null();
    builder.values().append_value(7);
    builder.append(true);

    builder.append(true);

    builder.values().append_value(7);
    builder.append(true);

    builder.append_null();

    let arr: ArrayRef = Arc::new(builder.finish());
    let field = arrow_schema::Field::new(
        "c",
        DataType::List(Arc::new(arrow_schema::Field::new(
            "item",
            DataType::Int32,
            true,
        ))),
        true,
    );
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected constant layout in slot 2");
    };
    assert!(layout.inline_value.is_some());
    assert_eq!(page.data.len(), 2);

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_fixed_size_list_not_used_v2_2() {
    use crate::format::metadata::page_layout::Layout;
    use arrow_array::builder::{FixedSizeListBuilder, Int32Builder};

    let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), 3);
    for _ in 0..64 {
        builder.values().append_value(1);
        builder.values().append_null();
        builder.values().append_value(3);
        builder.append(true);
    }
    let arr: ArrayRef = Arc::new(builder.finish());
    let field = arrow_schema::Field::new(
        "c",
        DataType::FixedSizeList(
            Arc::new(arrow_schema::Field::new("item", DataType::Int32, true)),
            3,
        ),
        true,
    );
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    if let PageEncoding::Structural(layout) = &page.description {
        assert!(
            !matches!(layout.layout.as_ref().unwrap(), Layout::ConstantLayout(_)),
            "FixedSizeList should not use constant layout yet"
        );
    }

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_constant_layout_not_written_before_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    let arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![7; 1024]));
    let field = arrow_schema::Field::new("c", DataType::Int32, true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU16).await;

    let PageEncoding::Structural(layout) = &page.description else {
        return;
    };
    assert!(
        !matches!(layout.layout.as_ref().unwrap(), Layout::ConstantLayout(_)),
        "Should not emit constant layout before v2.2"
    );

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU16)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

#[tokio::test]
async fn test_all_null_constant_layout_still_works_v2_2() {
    use crate::format::metadata::page_layout::Layout;

    let arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![None, None, None]));
    let field = arrow_schema::Field::new("c", DataType::Int32, true);
    let page = encode_first_page(field, arr.clone(), TestEncoding::StructuralU32).await;

    let PageEncoding::Structural(layout) = &page.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout) = layout.layout.as_ref().unwrap() else {
        panic!("Expected layout in slot 2");
    };
    assert!(layout.inline_value.is_none());
    assert_eq!(page.data.len(), 0);

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);
    check_round_trip_encoding_of_data(vec![arr], &test_cases, HashMap::new()).await;
}

fn hand_built_dictionary_with_out_of_range_null_keys() -> ArrayRef {
    use arrow_array::{DictionaryArray, Int32Array, types::Int32Type};
    use arrow_buffer::NullBuffer;

    let keys = Int32Array::new(
        vec![0, 7, 7].into(),
        Some(NullBuffer::from(vec![true, false, false])),
    );
    let values = Arc::new(StringArray::from(vec!["a"]));
    Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap()) as ArrayRef
}

fn concatenated_dictionary_with_out_of_range_null_keys() -> ArrayRef {
    use arrow_array::{builder::StringDictionaryBuilder, new_null_array, types::Int32Type};

    let mut builder = StringDictionaryBuilder::<Int32Type>::new();
    builder.append_value("a");
    for _ in 0..7 {
        builder.append_null();
    }
    let valued = Arc::new(builder.finish()) as ArrayRef;
    let all_null = new_null_array(valued.data_type(), 8);
    arrow_select::concat::concat(&[valued.as_ref(), all_null.as_ref()]).unwrap()
}

#[rstest::rstest]
#[case::hand_built(hand_built_dictionary_with_out_of_range_null_keys(), 1)]
#[case::concatenated(concatenated_dictionary_with_out_of_range_null_keys(), 8)]
#[tokio::test]
async fn test_dictionary_out_of_range_null_keys_round_trip(
    #[case] dictionary: ArrayRef,
    #[case] split_at: usize,
) {
    let num_rows = dictionary.len();
    let dictionaries = vec![
        dictionary.slice(0, split_at),
        dictionary.slice(split_at, num_rows - split_at),
    ];
    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![1])
        .with_range(split_at as u64..num_rows as u64)
        .with_indices(vec![split_at as u64]);

    check_round_trip_encoding_of_data(dictionaries, &test_cases, HashMap::new()).await;
}

#[test]
fn test_encode_decode_complex_all_null_vals_roundtrip() {
    use crate::compression::{DecompressionStrategy, DefaultDecompressionStrategy};

    let values: Arc<[u16]> = Arc::from((0..2048).map(|i| (i % 5) as u16).collect::<Vec<u16>>());

    let compression_strategy = crate::testing::test_compression_strategy(
        TestEncoding::StructuralU16,
        crate::compression_config::CompressionParams::default(),
    );
    let decompression_strategy = DefaultDecompressionStrategy::default();

    let (compressed_buf, encoding) = PrimitiveStructuralEncoder::encode_complex_all_null_vals(
        &values,
        compression_strategy.as_ref(),
    )
    .unwrap();

    let decompressor = decompression_strategy
        .create_block_decompressor(&encoding)
        .unwrap();
    let decompressed = decompressor
        .decompress(Some(compressed_buf), values.len() as u64)
        .unwrap();
    let decompressed_fixed_width = decompressed.as_fixed_width().unwrap();
    assert_eq!(decompressed_fixed_width.num_values, values.len() as u64);
    assert_eq!(decompressed_fixed_width.bits_per_value, 16);
    let rep_result = decompressed_fixed_width.data.borrow_to_typed_slice::<u16>();
    assert_eq!(rep_result.as_ref(), values.as_ref());
}

#[tokio::test]
async fn test_complex_all_null_compression_gated_by_version() {
    use crate::format::metadata::page_layout::Layout;
    use arrow_array::ListArray;

    let list_array = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(
        (0..1000).map(|i| if i % 2 == 0 { None } else { Some(vec![]) }),
    );
    let arr: ArrayRef = Arc::new(list_array);
    let field = arrow_schema::Field::new(
        "c",
        DataType::List(Arc::new(arrow_schema::Field::new(
            "item",
            DataType::Int32,
            true,
        ))),
        true,
    );

    let page_v21 = encode_first_page(field.clone(), arr.clone(), TestEncoding::StructuralU16).await;
    let PageEncoding::Structural(layout_v21) = &page_v21.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout_v21) = layout_v21.layout.as_ref().unwrap() else {
        panic!("Expected constant layout");
    };
    assert!(layout_v21.rep_compression.is_none());
    assert!(layout_v21.def_compression.is_none());
    assert_eq!(layout_v21.num_rep_values, 0);
    assert_eq!(layout_v21.num_def_values, 0);

    let page_v22 = encode_first_page(field, arr, TestEncoding::StructuralU32).await;
    let PageEncoding::Structural(layout_v22) = &page_v22.description else {
        panic!("Expected structural encoding");
    };
    let Layout::ConstantLayout(layout_v22) = layout_v22.layout.as_ref().unwrap() else {
        panic!("Expected constant layout");
    };
    assert!(layout_v22.def_compression.is_some());
    assert!(layout_v22.num_def_values > 0);
}

#[tokio::test]
async fn test_complex_all_null_round_trip() {
    use arrow_array::ListArray;

    let list_array = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(
        (0..1000).map(|i| if i % 2 == 0 { None } else { Some(vec![]) }),
    );

    let test_cases = TestCases::default().with_u32_structural_encodings();
    check_round_trip_encoding_of_data(vec![Arc::new(list_array)], &test_cases, HashMap::new())
        .await;
}

#[tokio::test]
async fn test_complex_all_null_constant_def_round_trip() {
    use arrow_array::ListArray;

    // Every row is a null list => constant def levels => a single RLE run,
    // exercising the lazy run-form decode end to end.
    let list_array = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(
        (0..5000).map(|_| None::<Vec<Option<i32>>>),
    );

    let test_cases = TestCases::default().with_u32_structural_encodings();
    check_round_trip_encoding_of_data(vec![Arc::new(list_array)], &test_cases, HashMap::new())
        .await;
}

/// Builds low-cardinality rows over keys that stress the dictionary maps:
/// empty, a prefix chain, long shared prefixes and multi-byte UTF-8, with a
/// null every 17th row.
fn tricky_dictionary_rows(num_rows: usize) -> Vec<Option<String>> {
    let shared = "p".repeat(300);
    let mut keys: Vec<String> = vec![String::new(), "a".into(), "aa".into(), "aaa".into()];
    keys.extend(["héllo", "日本語", "🦀"].map(String::from));
    keys.extend((0..8).map(|i| format!("{shared}{i}")));
    (0..num_rows)
        .map(|i| (i % 17 != 0).then(|| keys[(i * 7 + i / 5) % keys.len()].clone()))
        .collect()
}

fn dictionary_miniblock_metadata() -> HashMap<String, String> {
    use crate::constants::{DICT_SIZE_RATIO_META_KEY, STRUCTURAL_ENCODING_META_KEY};
    let mut metadata = HashMap::new();
    metadata.insert(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
    );
    metadata.insert(DICT_SIZE_RATIO_META_KEY.to_string(), "0.99".to_string());
    metadata
}

/// Strings (32- and 64-bit offsets) and binary values with nulls round-trip
/// through the dictionary encoder, and the pages really are dictionary pages.
#[tokio::test]
async fn test_dictionary_round_trip_variable_width_with_nulls_and_tricky_keys() {
    use arrow_array::{BinaryArray, LargeStringArray};

    let rows = tricky_dictionary_rows(10_000);
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(rows.clone())),
        Arc::new(LargeStringArray::from(rows.clone())),
        Arc::new(BinaryArray::from_iter(
            rows.iter().map(|r| r.as_ref().map(|s| s.as_bytes())),
        )),
    ];
    for array in arrays {
        let test_cases = TestCases::default()
            .with_encodings([TestEncoding::StructuralU16, TestEncoding::StructuralU32])
            .with_batch_size(1000)
            .with_range(0..1000)
            .with_indices(vec![0, 1, 16, 17, 34, 5_000, 9_999])
            .with_expected_encoding("dictionary");
        check_round_trip_encoding_of_data(
            vec![array],
            &test_cases,
            dictionary_miniblock_metadata(),
        )
        .await;
    }
}

/// 64-bit values with nulls, including the extremes, round-trip through the
/// fixed-width dictionary encoder.
#[tokio::test]
async fn test_dictionary_round_trip_u64_with_nulls() {
    use arrow_array::UInt64Array;

    let keys = [0u64, 1, u64::MAX, 1 << 63, 0xdead_beef];
    let values: Vec<Option<u64>> = (0..10_000)
        .map(|i| (i % 13 != 0).then(|| keys[(i * 3 + i / 7) % keys.len()]))
        .collect();
    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_batch_size(1000)
        .with_range(0..1000)
        .with_indices(vec![0, 1, 13, 26, 5_000, 9_999])
        .with_expected_encoding("dictionary");
    check_round_trip_encoding_of_data(
        vec![Arc::new(UInt64Array::from(values)) as ArrayRef],
        &test_cases,
        dictionary_miniblock_metadata(),
    )
    .await;
}

/// Enough distinct values that a page may exceed the dictionary budget part way
/// through building its map; whichever encoding wins, values round-trip.
#[tokio::test]
async fn test_dictionary_round_trip_high_cardinality_strings() {
    let values: Vec<Option<String>> = (0..20_000)
        .map(|i| (i % 29 != 0).then(|| format!("value-{:06}", (i * 7919) % 6_000)))
        .collect();
    let test_cases = TestCases::default()
        .with_encodings([TestEncoding::StructuralU16, TestEncoding::StructuralU32])
        .with_batch_size(1000)
        .with_range(0..2000)
        .with_indices(vec![0, 29, 10_000, 19_999]);
    check_round_trip_encoding_of_data(
        vec![Arc::new(StringArray::from(values)) as ArrayRef],
        &test_cases,
        dictionary_miniblock_metadata(),
    )
    .await;
}

/// Three single-value dictionary chunks merged into one page leave the merged
/// dictionary values as a run of identical entries, so the writer RLE-compresses
/// the dictionary values block.  The mini-block reader must accept the encodings
/// the writer emits (regression: it rejected `Compression::Rle`, so no scan
/// could read a file the writer had just produced).
#[tokio::test]
async fn test_dictionary_round_trip_rle_encoded_dictionary_values() {
    use arrow_array::{UInt32Array, UInt64Array, types::UInt32Type};

    let single_value = |rows: usize| -> ArrayRef {
        let values = Arc::new(UInt64Array::from(vec![5_881_210_131_331_364_753u64]));
        let keys = UInt32Array::from(vec![0u32; rows]);
        Arc::new(DictionaryArray::<UInt32Type>::try_new(keys, values).unwrap())
    };
    let chunks: Vec<ArrayRef> = (0..3).map(|_| single_value(2)).collect();

    let test_cases = TestCases::default()
        .with_encoding(TestEncoding::StructuralU32)
        .with_page_sizes(vec![4096]);

    check_round_trip_encoding_of_data(chunks, &test_cases, HashMap::new()).await;
}

/// Values above the 32 KiB per-value threshold take the per-value zstd path.
/// Mixed with nulls, empty and small values, at several levels (and the
/// default when no level is set), every page is zstd and every value survives.
#[tokio::test]
async fn test_per_value_zstd_round_trip_large_values_with_nulls() {
    use crate::constants::{COMPRESSION_LEVEL_META_KEY, COMPRESSION_META_KEY};

    let values: Vec<Option<String>> = (0..120)
        .map(|i| match i % 6 {
            0 => None,
            1 => Some(String::new()),
            2 => Some(format!("small-{i}")),
            _ => Some(format!("{i:05}-").repeat(8_000 + i * 97)),
        })
        .collect();
    for level in [None, Some("1"), Some("3"), Some("19")] {
        let mut metadata = HashMap::new();
        metadata.insert(COMPRESSION_META_KEY.to_string(), "zstd".to_string());
        if let Some(level) = level {
            metadata.insert(COMPRESSION_LEVEL_META_KEY.to_string(), level.to_string());
        }
        let test_cases = TestCases::default()
            .with_encodings([TestEncoding::StructuralU16, TestEncoding::StructuralU32])
            .with_page_sizes(vec![64 * 1024 * 1024])
            .with_batch_size(10)
            .with_range(0..30)
            .with_indices(vec![0, 1, 2, 3, 59, 119])
            .with_expected_encoding("zstd");
        check_round_trip_encoding_of_data(
            vec![Arc::new(StringArray::from(values.clone())) as ArrayRef],
            &test_cases,
            metadata,
        )
        .await;
    }
}

#[rstest::rstest]
#[tokio::test]
async fn test_axis_delta_per_value_compression(
    #[values("off", "on:1", "auto:1", "auto", "default")] mode: &str,
    #[values("zstd", "lz4", "none")] compression: &str,
    #[values(false, true)] nullable: bool,
    #[values("fullzip", "sparse")] structure: &str,
) {
    const ROWS: usize = 16;
    const DIM: usize = 3 * 17 * 23;
    let child: ArrayRef = Arc::new(arrow_array::Int32Array::from(
        (0..ROWS * DIM)
            .map(|i| (i as i32 * 107) - 1_000_000)
            .collect::<Vec<_>>(),
    ));
    let array: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new(
            "item",
            child.data_type().clone(),
            true,
        )),
        DIM as i32,
        child,
        nullable.then(|| NullBuffer::from((0..ROWS).map(|i| i % 3 != 0).collect::<Vec<_>>())),
    ));
    let mut metadata = HashMap::from([
        (
            "lance-encoding:structural-encoding".to_string(),
            structure.to_string(),
        ),
        ("lance-encoding:axis-delta".to_string(), mode.to_string()),
        (COMPRESSION_META_KEY.to_string(), compression.to_string()),
    ]);
    if mode == "default" {
        metadata.remove("lance-encoding:axis-delta");
    }
    metadata.insert(
        "ARROW:extension:name".into(),
        "arrow.fixed_shape_tensor".into(),
    );
    metadata.insert(
        "ARROW:extension:metadata".into(),
        r#"{"shape":[23,17,3]}"#.into(),
    );
    let field = arrow_schema::Field::new("pixels", array.data_type().clone(), true)
        .with_metadata(metadata.clone());
    let page = encode_first_page(field, array.clone(), TestEncoding::StructuralSparse).await;
    let PageEncoding::Planar(layout) = page.description else {
        panic!("expected Planar")
    };
    let Some(crate::format::pb23::value_plane::Codec::Zipped(values)) =
        layout.values.and_then(|v| v.codec)
    else {
        panic!("expected Zipped")
    };
    let mut codec = values.compression.as_ref().unwrap();
    if let Some(Compression::General(general)) = &codec.compression {
        codec = general.values.as_ref().unwrap();
    }
    assert_eq!(
        matches!(codec.compression, Some(Compression::AxisDelta(_))),
        mode != "off"
            && !((mode.starts_with("auto") || mode == "default") && compression == "none")
    );
    let tests = TestCases::default()
        .with_encoding(TestEncoding::StructuralSparse)
        .with_page_sizes(vec![8 * 1024])
        .with_ingest_batch_counts([4])
        .with_slicing_modes([true])
        .with_batch_size(3)
        .with_range(3..15)
        .with_indices(vec![0, 1, 3, 8, 14, 15]);
    check_round_trip_encoding_of_data(vec![array], &tests, metadata).await;
}

#[rstest::rstest]
#[case::scalar("scalar")]
#[case::strings("strings")]
#[case::float_vector("float")]
#[case::noise("noise")]
#[case::repeated_vector("repeated")]
#[case::inner_nulls("inner_nulls")]
#[case::small_tensor("small")]
#[case::invalid_extension("invalid_extension")]
#[tokio::test]
async fn test_default_axis_delta_preserves_other_encodings(
    #[case] kind: &str,
    #[values("lz4", "zstd", "none", "unset")] compression: &str,
    #[values(
        TestEncoding::StructuralSparse,
        TestEncoding::StructuralU32,
        TestEncoding::StructuralU16
    )]
    version: TestEncoding,
) {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(601);
    const ROWS: usize = 128;
    let dimension = if kind == "small" { 2 } else { 1024 };
    let array: ArrayRef = match kind {
        "scalar" => Arc::new(arrow_array::Int32Array::from(
            (0..ROWS as i32).collect::<Vec<_>>(),
        )),
        "strings" => Arc::new(StringArray::from(
            (0..ROWS)
                .map(|i| format!("value {i} {}", "abc".repeat(i % 7)))
                .collect::<Vec<_>>(),
        )),
        _ => {
            let values: ArrayRef = if kind == "float" {
                Arc::new(Float32Array::from(
                    (0..ROWS * dimension)
                        .map(|_| rng.random::<f32>())
                        .collect::<Vec<_>>(),
                ))
            } else {
                Arc::new(arrow_array::UInt32Array::from(
                    (0..ROWS * dimension)
                        .map(|i| {
                            if kind == "inner_nulls" && i % 7 == 0 {
                                None
                            } else {
                                Some(if kind == "repeated" {
                                    (i % dimension) as u32
                                } else {
                                    rng.random()
                                })
                            }
                        })
                        .collect::<Vec<_>>(),
                ))
            };
            Arc::new(FixedSizeListArray::new(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    values.data_type().clone(),
                    true,
                )),
                dimension as i32,
                values,
                None,
            ))
        }
    };
    let mut metadata = HashMap::new();
    if compression != "unset" {
        metadata.insert(COMPRESSION_META_KEY.into(), compression.into());
    }
    if kind == "invalid_extension" {
        metadata.insert(
            "ARROW:extension:name".into(),
            "arrow.fixed_shape_tensor".into(),
        );
        metadata.insert("ARROW:extension:metadata".into(), "invalid-json".into());
    }
    let field = arrow_schema::Field::new("value", array.data_type().clone(), false)
        .with_metadata(metadata.clone());
    let automatic = encode_first_page(field, array.clone(), version).await;
    if matches!(
        kind,
        "float" | "noise" | "inner_nulls" | "small" | "invalid_extension"
    ) {
        // Auto must preserve raw storage when compression has no sampled benefit
        // or the tensor is unsupported. Stable writers already use raw storage.
        metadata.remove(COMPRESSION_META_KEY);
    }
    metadata.insert("lance-encoding:axis-delta".into(), "off".into());
    let field =
        arrow_schema::Field::new("value", array.data_type().clone(), false).with_metadata(metadata);
    let disabled = encode_first_page(field, array, version).await;
    assert_eq!(
        format!("{:?}", automatic.description),
        format!("{:?}", disabled.description)
    );
    assert_eq!(automatic.data.len(), disabled.data.len());
    for (automatic, disabled) in automatic.data.iter().zip(&disabled.data) {
        assert_eq!(automatic.as_ref(), disabled.as_ref());
    }
}

fn int8_dictionary_with_all_values() -> ArrayRef {
    let values: Vec<u16> = (0..127u16).collect();
    let keys: Vec<i8> = (0..127)
        .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 127) as i8)
        .collect();
    Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(keys),
            Arc::new(UInt16Array::from(values)),
        )
        .unwrap(),
    )
}

#[rstest::rstest]
// Both chunks land in one page, which the writer merges.
#[case::one_page(4096, vec![TestEncoding::StructuralU32])]
// Each chunk is its own page, so the reader merges the two decoded dictionaries
// for a batch that spans both pages.
#[case::page_per_chunk(
    64,
    vec![TestEncoding::Array, TestEncoding::StructuralU16, TestEncoding::StructuralU32]
)]
#[tokio::test]
async fn test_dictionary_merge_keeps_declared_key_type_round_trip(
    #[case] page_size: u64,
    #[case] encodings: Vec<TestEncoding>,
) {
    // Two Dictionary(Int8, UInt16) chunks, each holding all 127 values an Int8
    // key can address.  Arrow's lossy concat merge over-counts the union (hash
    // collisions in its interner) and overflows Int8, although the exact union
    // fits.  The column must round-trip with its declared key type.
    let a = int8_dictionary_with_all_values();
    let b = int8_dictionary_with_all_values();
    // Build the expected array directly: `concat` hits the same Arrow
    // interner overflow, so the test harness cannot derive it.
    let values: Vec<u16> = (0..127u16).collect();
    let permutation: Vec<i8> = (0..127)
        .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 127) as i8)
        .collect();
    let mut keys = permutation.clone();
    keys.extend_from_slice(&permutation);
    let expected: ArrayRef = Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(keys),
            Arc::new(UInt16Array::from(values)),
        )
        .unwrap(),
    );
    let test_cases = TestCases::default()
        .with_encodings(encodings)
        .with_page_sizes(vec![page_size])
        .with_batch_size(1000);
    check_round_trip_encoding_of_data_with_expected(
        vec![a, b],
        Some(expected),
        &test_cases,
        HashMap::new(),
    )
    .await;
}

fn disjoint_int8_dictionary(offset: u16) -> ArrayRef {
    let values: Vec<u16> = (0..100u16).map(|value| value + offset).collect();
    let keys: Vec<i8> = (0..100).map(|i| i as i8).collect();
    Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(keys),
            Arc::new(UInt16Array::from(values)),
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn test_dictionary_merge_overflow_is_rejected_at_write() {
    // Two chunks with disjoint dictionaries whose exact union (200 values)
    // cannot be addressed by Int8 indices.  The writer must reject the
    // mismatch instead of persisting indices wider than the declared key.
    use crate::encoder::{
        ColumnIndexSequence, EncodingOptions, MIN_PAGE_BUFFER_ALIGNMENT, OutOfLineBuffers,
    };
    use crate::repdef::RepDefBuilder;
    use crate::testing::{create_test_field_encoder, test_encoding_strategy};

    let a = disjoint_int8_dictionary(0);
    let b = disjoint_int8_dictionary(1000);
    let field = ArrowField::new("d", a.data_type().clone(), false);
    let lance_field = lance_core::datatypes::Field::try_from(&field).unwrap();
    let encoding_strategy = test_encoding_strategy(TestEncoding::StructuralU32);
    let mut column_index_seq = ColumnIndexSequence::default();
    let encoding_options = EncodingOptions {
        cache_bytes_per_column: 1024 * 1024,
        max_page_bytes: 32 * 1024 * 1024,
        keep_original_array: true,
        buffer_alignment: MIN_PAGE_BUFFER_ALIGNMENT,
    };
    let mut encoder = create_test_field_encoder(
        encoding_strategy.as_ref(),
        &lance_field,
        &mut column_index_seq,
        &encoding_options,
    )
    .unwrap();

    let mut external_buffers = OutOfLineBuffers::new(0, MIN_PAGE_BUFFER_ALIGNMENT);
    let mut error = None;
    for (index, array) in [a, b].into_iter().enumerate() {
        let row_number = (index * 100) as u64;
        match encoder.maybe_encode(
            array,
            &mut external_buffers,
            RepDefBuilder::default(),
            row_number,
            100,
        ) {
            Ok(tasks) => {
                for task in tasks {
                    if let Err(err) = task.await {
                        error = Some(err);
                        break;
                    }
                }
            }
            Err(err) => error = Some(err),
        }
        if error.is_some() {
            break;
        }
    }
    if error.is_none() {
        for task in encoder.flush(&mut external_buffers).unwrap() {
            if let Err(err) = task.await {
                error = Some(err);
                break;
            }
        }
    }

    let error = error.expect("oversized dictionary merge must be rejected");
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(
        error.to_string().contains("widen the dictionary key type"),
        "unexpected error: {error}"
    );
}
