// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution metadata for shared structural engines. Page descriptors are not
//! wire messages. Stable protobufs are converted at the file boundary; current
//! compression descriptors are shared with the independent 2.3 grammar.

use prost::Message;

pub use super::pb23::{
    BufferCompression, ByteStreamSplit, CompressionScheme, CompressiveEncoding, Constant, Delta,
    Dictionary, FixedSizeList, Flat, Fsst, General, InlineBitpacking, OutOfLineBitpacking,
    PackedStruct, Range, RepDefLayer, Rle, Variable, VariablePackedStruct, compressive_encoding,
    variable_packed_struct,
};

/// A structural engine's execution description, independent of its wire envelope.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PageLayout {
    /// The selected engine. Missing layouts are rejected before execution.
    pub layout: Option<page_layout::Layout>,
}

/// Structural engine choices.
pub mod page_layout {
    use super::*;
    /// Physical engines shared by stable and planar pages.
    #[derive(Clone, Debug, PartialEq)]
    pub enum Layout {
        /// Independently compressed chunks, with optional levels and dictionary.
        MiniBlockLayout(MiniBlockLayout),
        /// Individually addressable values interleaved with optional control words.
        FullZipLayout(FullZipLayout),
        /// An inline scalar or no materialized values, with optional level buffers.
        ConstantLayout(ConstantLayout),
        /// Stable Blob v1 descriptors and their null semantics.
        BlobLayout(Box<BlobLayout>),
    }
}

/// Buffer and row accounting needed by the mini-block engine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MiniBlockLayout {
    /// Repetition stream encoding; absent when no repetition buffer exists.
    pub rep_compression: Option<CompressiveEncoding>,
    /// Definition stream encoding; absent when no definition buffer exists.
    pub def_compression: Option<CompressiveEncoding>,
    /// Value encoding, required for engines that materialize values.
    pub value_compression: Option<CompressiveEncoding>,
    /// Page dictionary encoding; absent when chunks do not share a dictionary.
    pub dictionary: Option<CompressiveEncoding>,
    /// Number of entries in the page dictionary.
    pub num_dictionary_items: u64,
    /// Inner-most to outer-most repetition/definition interpretations.
    pub layers: Vec<i32>,
    /// Value buffers per chunk, excluding repetition and definition streams.
    pub num_buffers: u64,
    /// Number of list layers represented by the repetition index.
    pub repetition_index_depth: u32,
    /// Number of structural events, including invisible entries.
    pub num_items: u64,
    /// Selects u32 chunk metadata; false selects the stable u16 representation.
    pub has_large_chunk: bool,
}

/// Addressing and level information needed by the zipped engine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FullZipLayout {
    /// Repetition bits in each control word.
    pub bits_rep: u32,
    /// Definition bits in each control word.
    pub bits_def: u32,
    /// Fixed value width or variable length-prefix width; required for zipped values.
    pub details: Option<full_zip_layout::Details>,
    /// Number of structural events, including invisible entries.
    pub num_items: u32,
    /// Number of leaf slots that are visible to the parent.
    pub num_visible_items: u32,
    /// Value encoding, required for engines that materialize values.
    pub value_compression: Option<CompressiveEncoding>,
    /// Inner-most to outer-most repetition/definition interpretations.
    pub layers: Vec<i32>,
}

/// Zipped value addressing.
pub mod full_zip_layout {
    /// How to locate values within the zipped buffer.
    #[derive(Clone, Debug, PartialEq)]
    pub enum Details {
        /// Fixed width of each encoded value, in bits.
        BitsPerValue(u32),
        /// Width of each variable-value length prefix, in bits.
        BitsPerOffset(u32),
    }
}

/// Scalar and level information needed by the constant engine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstantLayout {
    /// Inner-most to outer-most repetition/definition interpretations.
    pub layers: Vec<i32>,
    /// Encoded scalar; absent for pages without materialized values.
    pub inline_value: Option<bytes::Bytes>,
    /// Repetition stream encoding; absent when no repetition buffer exists.
    pub rep_compression: Option<CompressiveEncoding>,
    /// Definition stream encoding; absent when no definition buffer exists.
    pub def_compression: Option<CompressiveEncoding>,
    /// Number of entries in the repetition stream.
    pub num_rep_values: u64,
    /// Number of entries in the definition stream.
    pub num_def_values: u64,
}

/// Frozen Blob v1 execution description used by stable readers and writers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlobLayout {
    /// Descriptor layout, required by the stable Blob v1 engine.
    pub inner_layout: Option<Box<PageLayout>>,
    /// Inner-most to outer-most repetition/definition interpretations.
    pub layers: Vec<i32>,
}

impl PageLayout {
    /// Estimate retained metadata size using the same conservative compression
    /// tree multiplier as the wire-backed page metadata.
    pub(crate) fn estimated_size(&self) -> usize {
        let compression =
            |value: &Option<CompressiveEncoding>| value.as_ref().map_or(0, |v| v.encoded_len() * 4);
        std::mem::size_of::<Self>()
            + match &self.layout {
                Some(page_layout::Layout::MiniBlockLayout(v)) => {
                    v.layers.capacity() * 4
                        + compression(&v.rep_compression)
                        + compression(&v.def_compression)
                        + compression(&v.value_compression)
                        + compression(&v.dictionary)
                }
                Some(page_layout::Layout::FullZipLayout(v)) => {
                    v.layers.capacity() * 4 + compression(&v.value_compression)
                }
                Some(page_layout::Layout::ConstantLayout(v)) => {
                    v.layers.capacity() * 4
                        + v.inline_value.as_ref().map_or(0, |v| v.len())
                        + compression(&v.rep_compression)
                        + compression(&v.def_compression)
                }
                Some(page_layout::Layout::BlobLayout(v)) => {
                    std::mem::size_of::<BlobLayout>()
                        + v.layers.capacity() * 4
                        + v.inner_layout.as_ref().map_or(0, |v| v.estimated_size())
                }
                None => 0,
            }
    }
}
