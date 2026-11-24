// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use core::ops::Range;

use byteorder::BigEndian;
use byteorder::WriteBytesExt;

use super::constants::*;
use super::jentry::JEntry;
use crate::OwnedJsonb;
use crate::RawJsonb;
use crate::core::ExtensionItem;
use crate::core::JsonbItem;
use crate::core::NumberItem;
use crate::error::Result;

pub struct ArrayBuilder<'a> {
    items: Vec<JsonbItem<'a>>,
}

impl<'a> ArrayBuilder<'a> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push_jsonb_item(&mut self, item: JsonbItem<'a>) {
        self.items.push(item);
    }

    pub(crate) fn build(self) -> Result<OwnedJsonb> {
        let mut buf = Vec::new();
        let header = ARRAY_CONTAINER_TAG | self.items.len() as u32;
        buf.write_u32::<BigEndian>(header)?;

        let mut jentry_index = reserve_jentries(&mut buf, self.items.len() * 4);
        for item in self.items.into_iter() {
            append_jsonb_item(&mut buf, &mut jentry_index, item)?;
        }
        Ok(OwnedJsonb::new(buf))
    }
}

fn append_jsonb_item(buf: &mut Vec<u8>, jentry_index: &mut usize, item: JsonbItem) -> Result<()> {
    match item {
        JsonbItem::Null => {
            let jentry = JEntry::make_null_jentry();
            replace_jentry(buf, jentry, jentry_index);
        }
        JsonbItem::Boolean(v) => {
            let jentry = if v {
                JEntry::make_true_jentry()
            } else {
                JEntry::make_false_jentry()
            };
            replace_jentry(buf, jentry, jentry_index);
        }
        JsonbItem::Number(num) => match num {
            NumberItem::Raw(data) => {
                let jentry = JEntry::make_number_jentry(data.len());
                replace_jentry(buf, jentry, jentry_index);
                buf.extend_from_slice(data);
            }
            NumberItem::Number(num) => {
                let len = num.compact_encode(&mut *buf)?;
                let jentry = JEntry::make_number_jentry(len);
                replace_jentry(buf, jentry, jentry_index);
            }
        },
        JsonbItem::String(data) => {
            let jentry = JEntry::make_string_jentry(data.len());
            replace_jentry(buf, jentry, jentry_index);
            buf.extend_from_slice(data.as_bytes());
        }
        JsonbItem::Extension(ext) => match ext {
            ExtensionItem::Raw(data) => {
                let jentry = JEntry::make_extension_jentry(data.len());
                replace_jentry(buf, jentry, jentry_index);
                buf.extend_from_slice(data);
            }
            ExtensionItem::Extension(ext) => {
                let len = ext.compact_encode(&mut *buf)?;
                let jentry = JEntry::make_extension_jentry(len);
                replace_jentry(buf, jentry, jentry_index);
            }
        },
        JsonbItem::Raw(raw_jsonb) => {
            append_raw_jsonb_data(buf, jentry_index, raw_jsonb)?;
        }
        JsonbItem::Owned(owned_jsonb) => {
            let raw_jsonb = owned_jsonb.as_raw();
            append_raw_jsonb_data(buf, jentry_index, raw_jsonb)?;
        }
    }
    Ok(())
}

fn append_raw_jsonb_data(
    buf: &mut Vec<u8>,
    jentry_index: &mut usize,
    raw_jsonb: RawJsonb,
) -> Result<()> {
    let (header_type, _) = raw_jsonb.read_header(0)?;
    if header_type == SCALAR_CONTAINER_TAG {
        let scalar_jentry = raw_jsonb.read_jentry(4)?;
        let range = Range {
            start: 8,
            end: raw_jsonb.len(),
        };
        let data = raw_jsonb.slice(range)?;
        replace_jentry(buf, scalar_jentry, jentry_index);
        buf.extend_from_slice(data);
    } else {
        let jentry = JEntry::make_container_jentry(raw_jsonb.len());
        replace_jentry(buf, jentry, jentry_index);
        buf.extend_from_slice(raw_jsonb.data);
    }
    Ok(())
}

fn reserve_jentries(buf: &mut Vec<u8>, len: usize) -> usize {
    let old_len = buf.len();
    let new_len = old_len + len;
    buf.resize(new_len, 0);
    old_len
}

fn replace_jentry(buf: &mut [u8], jentry: JEntry, jentry_index: &mut usize) {
    let jentry_bytes = jentry.encoded().to_be_bytes();
    for (i, b) in jentry_bytes.iter().enumerate() {
        buf[*jentry_index + i] = *b;
    }
    *jentry_index += 4;
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::ArrayBuilder;
    use crate::OwnedJsonb;
    use crate::core::JsonbItem;

    #[test]
    fn test_array_builder_matches_encoder() {
        let scalar: OwnedJsonb = "\"s\"".parse().unwrap();
        let nested: OwnedJsonb = r#"{"k":[1,2]}"#.parse().unwrap();

        let mut builder = ArrayBuilder::with_capacity(5);
        builder.push_jsonb_item(JsonbItem::Null);
        builder.push_jsonb_item(JsonbItem::Boolean(false));
        builder.push_jsonb_item(JsonbItem::String(Cow::Borrowed("x")));
        builder.push_jsonb_item(JsonbItem::Owned(scalar));
        builder.push_jsonb_item(JsonbItem::Raw(nested.as_raw()));
        let from_builder = builder.build().unwrap().to_vec();

        let from_encoder: OwnedJsonb = r#"[null,false,"x","s",{"k":[1,2]}]"#.parse().unwrap();
        assert_eq!(from_builder, from_encoder.to_vec());
    }
}
