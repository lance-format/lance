// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Translate planar wire metadata to and from shared engine descriptions.
//! These conversions do not use a stable-format protobuf.

use lance_core::{Error, Result};

use super::{metadata, pb23};

fn missing(label: &str) -> Error {
    Error::invalid_input(format!("Planar page is missing {label}"))
}

impl pb23::PlanarLayout {
    /// Return the sparse structure, if this page uses slot-domain mappings.
    pub fn sparse(&self) -> Option<&pb23::SparseStructure> {
        match self.structure.as_ref()?.codec.as_ref()? {
            pb23::structural_plane::Codec::Sparse(sparse) => Some(sparse),
            _ => None,
        }
    }

    /// Build the execution description for dense mini-block, zipped, or constant values.
    /// Sparse structure is scheduled independently of its value engine.
    pub fn dense_execution_layout(&self) -> Result<metadata::PageLayout> {
        use metadata::page_layout::Layout;
        use pb23::{structural_plane::Codec as Structure, value_plane::Codec as Values};
        let structure = self
            .structure
            .as_ref()
            .and_then(|s| s.codec.as_ref())
            .ok_or_else(|| missing("structure"))?;
        let Structure::Dense(dense) = structure else {
            return Err(Error::invalid_input("Expected dense planar structure"));
        };
        if dense.layers.is_empty()
            || dense
                .layers
                .iter()
                .any(|&layer| metadata::RepDefLayer::try_from(layer).is_err() || layer == 0)
        {
            return Err(Error::invalid_input(
                "Planar dense layers are empty or invalid",
            ));
        }
        let values = self
            .values
            .as_ref()
            .and_then(|v| v.codec.as_ref())
            .ok_or_else(|| missing("values"))?;
        let layout = match values {
            Values::Chunked(values) => {
                let levels = values
                    .levels
                    .as_ref()
                    .ok_or_else(|| missing("dense chunk levels"))?;
                Layout::MiniBlockLayout(metadata::MiniBlockLayout {
                    rep_compression: levels.repetition.clone(),
                    def_compression: levels.definition.clone(),
                    repetition_index_depth: levels.repetition_index_depth,
                    value_compression: values.compression.clone(),
                    dictionary: values.dictionary.clone(),
                    num_dictionary_items: values.num_dictionary_items,
                    num_buffers: values.num_buffers,
                    num_items: values.num_items,
                    layers: dense.layers.clone(),
                    has_large_chunk: true,
                })
            }
            Values::Zipped(values) => {
                let levels = values
                    .levels
                    .as_ref()
                    .ok_or_else(|| missing("dense zip levels"))?;
                Layout::FullZipLayout(values.full_zip(dense.layers.clone(), Some(levels))?)
            }
            Values::Constant(value) => {
                if value.scalar.is_empty() || value.scalar.len() > 32 {
                    return Err(Error::invalid_input(
                        "Planar constant scalar must contain 1..=32 bytes",
                    ));
                }
                if dense
                    .layers
                    .iter()
                    .any(|&layer| layer != metadata::RepDefLayer::RepdefAllValidItem as i32)
                {
                    return Err(Error::invalid_input(
                        "Dense constant requires trivial all-valid structure",
                    ));
                }
                Layout::ConstantLayout(metadata::ConstantLayout {
                    inline_value: Some(value.scalar.clone()),
                    layers: dense.layers.clone(),
                    ..Default::default()
                })
            }
            Values::Indexed(_) => {
                return Err(Error::invalid_input(
                    "Indexed values require sparse structure",
                ));
            }
            Values::None(_) => {
                if dense.layers.as_slice() != [metadata::RepDefLayer::RepdefNullableItem as i32] {
                    return Err(Error::invalid_input(
                        "Dense no-values requires a single null leaf layer",
                    ));
                }
                Layout::ConstantLayout(metadata::ConstantLayout {
                    layers: dense.layers.clone(),
                    ..Default::default()
                })
            }
        };
        Ok(metadata::PageLayout {
            layout: Some(layout),
        })
    }
}

impl pb23::ZippedValues {
    /// Build a zipped execution description, with optional dense controls.
    pub(crate) fn full_zip(
        &self,
        layers: Vec<i32>,
        levels: Option<&pb23::DenseZipLevels>,
    ) -> Result<metadata::FullZipLayout> {
        let details = match self
            .addressing
            .as_ref()
            .ok_or_else(|| missing("zipped addressing"))?
        {
            pb23::zipped_values::Addressing::BitsPerValue(bits) => {
                metadata::full_zip_layout::Details::BitsPerValue(*bits)
            }
            pb23::zipped_values::Addressing::BitsPerOffset(bits) => {
                metadata::full_zip_layout::Details::BitsPerOffset(*bits)
            }
        };
        Ok(metadata::FullZipLayout {
            bits_rep: levels.map_or(0, |l| l.bits_repetition),
            bits_def: levels.map_or(0, |l| l.bits_definition),
            details: Some(details),
            num_items: self
                .num_items
                .try_into()
                .map_err(|_| Error::invalid_input("Zipped item count exceeds u32"))?,
            num_visible_items: self
                .num_visible_items
                .try_into()
                .map_err(|_| Error::invalid_input("Zipped leaf count exceeds u32"))?,
            value_compression: self.compression.clone(),
            layers,
        })
    }
}

/// Translate a newly encoded dense page without changing its physical buffers.
pub(crate) fn from_dense_execution(layout: metadata::PageLayout) -> Result<pb23::PlanarLayout> {
    use metadata::page_layout::Layout;
    use pb23::value_plane::Codec;
    let (layers, values) = match layout.layout.ok_or_else(|| missing("dense layout"))? {
        Layout::MiniBlockLayout(m) => (
            m.layers,
            Codec::Chunked(pb23::ChunkedValues {
                compression: m.value_compression,
                levels: Some(pb23::DenseChunkLevels {
                    repetition: m.rep_compression,
                    definition: m.def_compression,
                    repetition_index_depth: m.repetition_index_depth,
                }),
                dictionary: m.dictionary,
                num_dictionary_items: m.num_dictionary_items,
                num_buffers: m.num_buffers,
                num_items: m.num_items,
            }),
        ),
        Layout::FullZipLayout(m) => (
            m.layers,
            Codec::Zipped(pb23::ZippedValues {
                compression: m.value_compression,
                levels: Some(pb23::DenseZipLevels {
                    bits_repetition: m.bits_rep,
                    bits_definition: m.bits_def,
                }),
                addressing: m.details.map(|d| match d {
                    metadata::full_zip_layout::Details::BitsPerValue(v) => {
                        pb23::zipped_values::Addressing::BitsPerValue(v)
                    }
                    metadata::full_zip_layout::Details::BitsPerOffset(v) => {
                        pb23::zipped_values::Addressing::BitsPerOffset(v)
                    }
                }),
                num_items: u64::from(m.num_items),
                num_visible_items: u64::from(m.num_visible_items),
            }),
        ),
        Layout::ConstantLayout(m) => {
            let value = match m.inline_value {
                Some(scalar) => Codec::Constant(pb23::ConstantValue { scalar }),
                None => Codec::None(pb23::NoValues {}),
            };
            (m.layers, value)
        }
        _ => {
            return Err(Error::invalid_input(
                "Only dense value engines can produce dense planar metadata",
            ));
        }
    };
    Ok(pb23::PlanarLayout {
        structure: Some(pb23::StructuralPlane {
            codec: Some(pb23::structural_plane::Codec::Dense(pb23::DenseStructure {
                layers,
            })),
        }),
        values: Some(pb23::ValuePlane {
            codec: Some(values),
        }),
    })
}
