// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit stable wire boundaries. No serialization round trip is used to
//! translate metadata, and a current-only codec cannot be emitted as stable.

use super::{metadata, pb21};
use lance_core::{Error, Result};

macro_rules! message {
    ($old:path, $new:path, direct: [$($direct:ident),*], optional: [$($optional:ident),*], boxed: [$($boxed:ident),*], repeated: [$($repeated:ident),*]) => {
        impl TryFrom<$old> for $new {
            type Error = Error;
            fn try_from(value: $old) -> Result<Self> {
                Ok(Self {
                    $($direct: value.$direct,)*
                    $($optional: value.$optional.map(TryInto::try_into).transpose()?,)*
                    $($boxed: value.$boxed.map(|v| (*v).try_into().map(Box::new)).transpose()?,)*
                    $($repeated: value.$repeated.into_iter().map(TryInto::try_into).collect::<Result<_>>()?,)*
                })
            }
        }
        impl TryFrom<$new> for $old {
            type Error = Error;
            fn try_from(value: $new) -> Result<Self> {
                Ok(Self {
                    $($direct: value.$direct,)*
                    $($optional: value.$optional.map(TryInto::try_into).transpose()?,)*
                    $($boxed: value.$boxed.map(|v| (*v).try_into().map(Box::new)).transpose()?,)*
                    $($repeated: value.$repeated.into_iter().map(TryInto::try_into).collect::<Result<_>>()?,)*
                })
            }
        }
    };
}

macro_rules! scalar_variants {
    ($old:path, $new:path, [$($variant:ident),*]) => {
        impl TryFrom<$old> for $new {
            type Error = Error;
            fn try_from(value: $old) -> Result<Self> {
                use $old as Old;
                Ok(match value { $(Old::$variant(v) => Self::$variant(v),)* })
            }
        }
        impl TryFrom<$new> for $old {
            type Error = Error;
            fn try_from(value: $new) -> Result<Self> {
                use $new as New;
                Ok(match value { $(New::$variant(v) => Self::$variant(v),)* })
            }
        }
    };
}

macro_rules! variants {
    ($old:path, $new:path, [$($variant:ident),*] $(, unsupported: [$($unsupported:ident),*])? $(, unsupported_old: [$($unsupported_old:ident),*])?) => {
        impl TryFrom<$old> for $new {
            type Error = Error;
            fn try_from(value: $old) -> Result<Self> {
                use $old as Old;
                match value {
                    $(Old::$variant(v) => Ok(Self::$variant(v.try_into()?)),)*
                    $($(Old::$unsupported_old(_) => Err(Error::invalid_input(concat!(stringify!($unsupported_old), " is not part of the stable encoding grammar"))),)*)?
                }
            }
        }
        impl TryFrom<$new> for $old {
            type Error = Error;
            fn try_from(value: $new) -> Result<Self> {
                use $new as New;
                match value {
                    $(New::$variant(v) => Ok(Self::$variant(v.try_into()?)),)*
                    $($(New::$unsupported(_) => Err(Error::invalid_input(concat!(stringify!($unsupported), " is not part of the stable encoding grammar"))),)*)?
                }
            }
        }
    };
}

// Recursive protobuf variants carry boxes; convert their contents explicitly.
macro_rules! boxed {
    ($old:path, $new:path) => {
        impl TryFrom<Box<$old>> for Box<$new> {
            type Error = Error;
            fn try_from(value: Box<$old>) -> Result<Self> {
                (*value).try_into().map(Box::new)
            }
        }
        impl TryFrom<Box<$new>> for Box<$old> {
            type Error = Error;
            fn try_from(value: Box<$new>) -> Result<Self> {
                (*value).try_into().map(Box::new)
            }
        }
    };
}
message!(
    pb21::BufferCompression,
    metadata::BufferCompression,
    direct: [scheme, level],
    optional: [],
    boxed: [],
    repeated: []
);
message!(
    pb21::Flat,
    metadata::Flat,
    direct: [bits_per_value],
    optional: [data],
    boxed: [],
    repeated: []
);
message!(
    pb21::Variable,
    metadata::Variable,
    direct: [],
    optional: [values],
    boxed: [offsets],
    repeated: []
);
message!(
    pb21::Constant,
    metadata::Constant,
    direct: [value],
    optional: [],
    boxed: [],
    repeated: []
);
message!(
    pb21::OutOfLineBitpacking,
    metadata::OutOfLineBitpacking,
    direct: [uncompressed_bits_per_value],
    optional: [],
    boxed: [values],
    repeated: []
);
message!(
    pb21::InlineBitpacking,
    metadata::InlineBitpacking,
    direct: [uncompressed_bits_per_value],
    optional: [values],
    boxed: [],
    repeated: []
);
message!(
    pb21::Fsst,
    metadata::Fsst,
    direct: [symbol_table],
    optional: [],
    boxed: [values],
    repeated: []
);
message!(
    pb21::Dictionary,
    metadata::Dictionary,
    direct: [num_dictionary_items],
    optional: [],
    boxed: [indices, items],
    repeated: []
);
message!(
    pb21::Rle,
    metadata::Rle,
    direct: [],
    optional: [],
    boxed: [values, run_lengths],
    repeated: []
);
message!(
    pb21::FixedSizeList,
    metadata::FixedSizeList,
    direct: [items_per_value, has_validity],
    optional: [],
    boxed: [values],
    repeated: []
);
message!(
    pb21::PackedStruct,
    metadata::PackedStruct,
    direct: [bits_per_value],
    optional: [],
    boxed: [values],
    repeated: []
);
message!(
    pb21::VariablePackedStruct,
    metadata::VariablePackedStruct,
    direct: [],
    optional: [],
    boxed: [],
    repeated: [fields]
);
message!(
    pb21::General,
    metadata::General,
    direct: [],
    optional: [compression],
    boxed: [values],
    repeated: []
);
message!(
    pb21::ByteStreamSplit,
    metadata::ByteStreamSplit,
    direct: [],
    optional: [],
    boxed: [values],
    repeated: []
);
message!(
    pb21::CompressiveEncoding,
    metadata::CompressiveEncoding,
    direct: [],
    optional: [compression],
    boxed: [],
    repeated: []
);
message!(
    pb21::MiniBlockLayout,
    metadata::MiniBlockLayout,
    direct: [num_dictionary_items, layers, num_buffers, repetition_index_depth, num_items, has_large_chunk],
    optional: [rep_compression, def_compression, value_compression, dictionary],
    boxed: [],
    repeated: []
);
message!(
    pb21::FullZipLayout,
    metadata::FullZipLayout,
    direct: [bits_rep, bits_def, num_items, num_visible_items, layers],
    optional: [details, value_compression],
    boxed: [],
    repeated: []
);
message!(
    pb21::ConstantLayout,
    metadata::ConstantLayout,
    direct: [layers, inline_value, num_rep_values, num_def_values],
    optional: [rep_compression, def_compression],
    boxed: [],
    repeated: []
);
message!(
    pb21::BlobLayout,
    metadata::BlobLayout,
    direct: [layers],
    optional: [],
    boxed: [inner_layout],
    repeated: []
);
message!(
    pb21::PageLayout,
    metadata::PageLayout,
    direct: [],
    optional: [layout],
    boxed: [],
    repeated: []
);
message!(
    pb21::variable_packed_struct::FieldEncoding,
    metadata::variable_packed_struct::FieldEncoding,
    direct: [],
    optional: [value, layout],
    boxed: [],
    repeated: []
);
boxed!(pb21::Variable, metadata::Variable);
boxed!(pb21::OutOfLineBitpacking, metadata::OutOfLineBitpacking);
boxed!(pb21::Fsst, metadata::Fsst);
boxed!(pb21::Dictionary, metadata::Dictionary);
boxed!(pb21::Rle, metadata::Rle);
boxed!(pb21::FixedSizeList, metadata::FixedSizeList);
boxed!(pb21::PackedStruct, metadata::PackedStruct);
boxed!(pb21::General, metadata::General);
boxed!(pb21::ByteStreamSplit, metadata::ByteStreamSplit);
boxed!(pb21::BlobLayout, metadata::BlobLayout);
variants!(
    pb21::compressive_encoding::Compression,
    metadata::compressive_encoding::Compression,
    [Flat, Variable, Constant, OutOfLineBitpacking, InlineBitpacking, Fsst, Dictionary,
     Rle, ByteStreamSplit, General, FixedSizeList, PackedStruct, VariablePackedStruct],
    unsupported: [Range, Delta], unsupported_old: [Range, Delta]
);
variants!(
    pb21::page_layout::Layout,
    metadata::page_layout::Layout,
    [MiniBlockLayout, FullZipLayout, ConstantLayout, BlobLayout], unsupported_old: [SparseLayout]
);
scalar_variants!(
    pb21::full_zip_layout::Details,
    metadata::full_zip_layout::Details,
    [BitsPerValue, BitsPerOffset]
);
scalar_variants!(
    pb21::variable_packed_struct::field_encoding::Layout,
    metadata::variable_packed_struct::field_encoding::Layout,
    [BitsPerValue, BitsPerLength]
);
