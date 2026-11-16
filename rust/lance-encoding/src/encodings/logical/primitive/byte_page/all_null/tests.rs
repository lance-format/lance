// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{future::Future, sync::Arc};

use arrow_array::new_null_array;
use arrow_schema::Field;
use lance_core::utils::allocation::{AllocationBudget, ReservedBuffer};
use rstest::rstest;

use super::*;
use crate::{
    compression::DefaultDecompressionStrategy,
    encodings::{logical::primitive::AdmittedBytePageScheduler, physical::range::RangeEncoder},
};

fn sparse_validity(rows: u64, null_positions: bool) -> pb23::SparseValiditySet {
    pb23::SparseValiditySet {
        meaning: if null_positions {
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions as i32
        } else {
            pb23::sparse_validity_set::Meaning::SparseValidityValidPositions as i32
        },
        positions: Some(if null_positions {
            RangeEncoder::new(64, 0, 1).encoding(rows).unwrap()
        } else {
            EncodingUtils::constant(None)
        }),
        num_positions: if null_positions { rows } else { 0 },
    }
}

fn fixture(kind: u8) -> PageInfo {
    let encoding = if kind == 0 {
        PageEncoding::Structural(EncodingUtils::constant_layout(
            &[DefinitionInterpretation::NullableItem],
            None,
        ))
    } else {
        PageEncoding::Planar(pb23::PlanarLayout {
            structure: Some(pb23::StructuralPlane {
                codec: Some(if kind == 3 {
                    pb23::structural_plane::Codec::Dense(pb23::DenseStructure {
                        layers: vec![pb23::RepDefLayer::RepdefNullableItem as i32],
                    })
                } else {
                    pb23::structural_plane::Codec::Sparse(pb23::SparseStructure {
                        layers: vec![pb23::SparseStructuralLayer {
                            layer: Some(pb23::sparse_structural_layer::Layer::Validity(
                                sparse_validity(9, kind == 2),
                            )),
                        }],
                    })
                }),
            }),
            values: Some(pb23::ValuePlane {
                codec: Some(pb23::value_plane::Codec::None(pb23::NoValues {})),
            }),
        })
    };
    PageInfo {
        num_rows: 9,
        priority: 0,
        encoding,
        buffer_offsets_and_sizes: Arc::from([]),
    }
}

fn unexpected_read(
    _: Range<u64>,
    _: ReservedBuffer,
) -> impl Future<Output = Result<ReservedBuffer>> {
    std::future::ready(Err(Error::io("null page must not read payload")))
}

#[rstest]
#[tokio::test]
async fn metadata_nulls_use_exact_output_credit_and_alias_lifetime(
    #[values(0, 1, 2, 3)] kind: u8,
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    data_type: DataType,
) {
    let page = AdmittedBytePageScheduler::try_from_page(
        &fixture(kind),
        &DefaultDecompressionStrategy::default(),
    )
    .unwrap();
    // Five output rows: six offsets, one bitmap byte, no value payload.
    let required = match data_type {
        DataType::Binary | DataType::Utf8 => 25,
        _ => 49,
    };
    let budget = AllocationBudget::new(required);
    let mut short = budget.try_reserve(required - 1).unwrap();
    let input = page
        .prepare_array_range_in(2..7, data_type.clone(), &mut short, unexpected_read)
        .await
        .unwrap()
        .unwrap();
    assert!(input.decode_in(&mut short).unwrap().is_none());
    assert_eq!(short.len(), required - 1);
    drop(short);
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(required).unwrap();
    let input = page
        .prepare_array_range_in(2..7, data_type.clone(), &mut work, unexpected_read)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), required, "preparation is metadata-only");
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.to_data(), new_null_array(&data_type, 5).to_data());
    assert_eq!(work.len(), 0);
    let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
    drop((output, work));
    assert_eq!(
        budget.reserved(),
        required - 1,
        "one byte keeps all offsets charged"
    );
    assert!(budget.try_reserve(required).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(required).is_some());
}

#[rstest]
#[tokio::test]
async fn null_page_validates_ranges_and_shape_before_capacity(#[values(0, 1, 2, 3)] kind: u8) {
    let page = AdmittedBytePageScheduler::try_from_page(
        &fixture(kind),
        &DefaultDecompressionStrategy::default(),
    )
    .unwrap();
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    for range in [0..0, 0..10, Range { start: 2, end: 1 }] {
        let error = page
            .prepare_array_range_in(range, DataType::Binary, &mut work, unexpected_read)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("all-null range"));
    }
    for data_type in [
        DataType::Int32,
        DataType::List(Arc::new(Field::new("item", DataType::Binary, true))),
    ] {
        let error = page
            .prepare_array_range_in(0..1, data_type, &mut work, unexpected_read)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("flat byte/string"));
    }
    let input = page
        .prepare_array_range_in(0..1, DataType::Binary, &mut work, unexpected_read)
        .await
        .unwrap()
        .unwrap();
    drop(input); // cancellation before CPU dispatch retains no physical allocation
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::missing_descriptor(0)]
#[case::invalid_meaning(1)]
#[case::nonempty_valid_set(2)]
#[case::null_set_outside_domain(3)]
#[case::payload_for_empty_set(4)]
#[case::unexpected_buffers(5)]
fn rejects_unproven_sparse_nulls(#[case] kind: u8) {
    let mut page = fixture(1);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    let Some(pb23::sparse_structural_layer::Layer::Validity(validity)) =
        structure.layers[0].layer.as_mut()
    else {
        unreachable!()
    };
    match kind {
        0 => validity.positions = None,
        1 => validity.meaning = 999,
        2 => {
            *validity = sparse_validity(9, true);
            validity.meaning =
                pb23::sparse_validity_set::Meaning::SparseValidityValidPositions as i32;
        }
        3 => {
            *validity = sparse_validity(8, true);
        }
        4 => validity.positions = Some(EncodingUtils::flat(64, None)),
        _ => page.buffer_offsets_and_sizes = Arc::from([(0, 1)]),
    }
    let error =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::CorruptFile { .. }
    ));
    assert!(error.to_string().contains("null") || error.to_string().contains("empty unsigned"));
}

#[test]
fn no_values_is_not_evidence_that_nested_parents_are_null() {
    let mut page = fixture(1);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    layout.structure = Some(pb23::StructuralPlane {
        codec: Some(pb23::structural_plane::Codec::Sparse(
            pb23::SparseStructure {
                layers: vec![pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::List(
                        pb23::SparseListLayer {
                            num_child_slots: 0,
                            num_non_empty_positions: 0,
                            non_empty_positions: Some(EncodingUtils::constant(None)),
                            counts: Some(EncodingUtils::constant(None)),
                            validity: Some(pb23::SparseValiditySet {
                                meaning:
                                    pb23::sparse_validity_set::Meaning::SparseValidityNullPositions
                                        as i32,
                                num_positions: 0,
                                positions: Some(EncodingUtils::constant(None)),
                            }),
                        },
                    )),
                }],
            },
        )),
    });
    // This describes valid empty lists, not null lists. No all-null shortcut.
    let error =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("dense FullZip page"));
}

#[test]
fn payload_validity_remains_explicitly_unsupported() {
    let mut page = fixture(2);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    let Some(pb23::sparse_structural_layer::Layer::Validity(validity)) =
        structure.layers[0].layer.as_mut()
    else {
        unreachable!()
    };
    validity.positions = Some(EncodingUtils::flat(64, None));
    page.buffer_offsets_and_sizes = Arc::from([(0, 9 * 8)]);
    let error =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("metadata-only positions"));
}

#[test]
fn null_page_does_not_ignore_dense_level_metadata() {
    let mut page = fixture(0);
    let PageEncoding::Structural(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(Layout::ConstantLayout(layout)) = &mut layout.layout else {
        unreachable!()
    };
    layout.num_def_values = 9;
    let error =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("level streams"));
}
