// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::ops::Range;

use super::constants::*;
use crate::RawJsonb;
use crate::core::JsonbItem;
use crate::core::util::jentry_to_jsonb_item;
use crate::error::Result;

pub struct ArrayIterator<'a> {
    raw_jsonb: RawJsonb<'a>,
    jentry_offset: usize,
    item_offset: usize,
    length: usize,
    index: usize,
}

impl<'a> ArrayIterator<'a> {
    pub(crate) fn new(raw_jsonb: RawJsonb<'a>) -> Result<Option<Self>> {
        let (header_type, header_len) = raw_jsonb.read_header(0)?;
        if header_type == ARRAY_CONTAINER_TAG {
            Ok(Some(Self::new_with_len(raw_jsonb, header_len)))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn new_with_len(raw_jsonb: RawJsonb<'a>, length: usize) -> Self {
        Self {
            raw_jsonb,
            jentry_offset: 4,
            item_offset: 4 + 4 * length,
            length,
            index: 0,
        }
    }

    fn remaining(&self) -> usize {
        self.length - self.index
    }
}

impl<'a> Iterator for ArrayIterator<'a> {
    type Item = Result<JsonbItem<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            return None;
        }
        let jentry = match self.raw_jsonb.read_jentry(self.jentry_offset) {
            Ok(jentry) => jentry,
            Err(err) => return Some(Err(err)),
        };

        let item_length = jentry.length as usize;
        let item_range = Range {
            start: self.item_offset,
            end: self.item_offset + item_length,
        };
        let data = match self.raw_jsonb.slice(item_range) {
            Ok(data) => data,
            Err(err) => return Some(Err(err)),
        };
        let item = jentry_to_jsonb_item(jentry, data);

        self.index += 1;
        self.jentry_offset += 4;
        self.item_offset += item_length;

        Some(Ok(item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ArrayIterator<'_> {}

pub struct ObjectValueIterator<'a> {
    raw_jsonb: RawJsonb<'a>,
    jentry_offset: usize,
    item_offset: usize,
    length: usize,
    index: usize,
}

impl<'a> ObjectValueIterator<'a> {
    pub(crate) fn new(raw_jsonb: RawJsonb<'a>) -> Result<Option<Self>> {
        let (header_type, header_len) = raw_jsonb.read_header(0)?;
        if header_type == OBJECT_CONTAINER_TAG {
            let mut jentry_offset = 4;
            let mut item_offset = 4 + 8 * header_len;
            for _ in 0..header_len {
                let key_jentry = raw_jsonb.read_jentry(jentry_offset)?;
                jentry_offset += 4;
                item_offset += key_jentry.length as usize;
            }

            Ok(Some(Self {
                raw_jsonb,
                jentry_offset,
                item_offset,
                length: header_len,
                index: 0,
            }))
        } else {
            Ok(None)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.length
    }

    fn remaining(&self) -> usize {
        self.length - self.index
    }
}

impl<'a> Iterator for ObjectValueIterator<'a> {
    type Item = Result<JsonbItem<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            return None;
        }
        let jentry = match self.raw_jsonb.read_jentry(self.jentry_offset) {
            Ok(jentry) => jentry,
            Err(err) => return Some(Err(err)),
        };

        let val_length = jentry.length as usize;
        let val_range = Range {
            start: self.item_offset,
            end: self.item_offset + val_length,
        };
        let data = match self.raw_jsonb.slice(val_range) {
            Ok(data) => data,
            Err(err) => return Some(Err(err)),
        };
        let val_item = jentry_to_jsonb_item(jentry, data);

        self.index += 1;
        self.jentry_offset += 4;
        self.item_offset += val_length;

        Some(Ok(val_item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ObjectValueIterator<'_> {}

pub struct ObjectIterator<'a> {
    raw_jsonb: RawJsonb<'a>,
    key_jentry_offset: usize,
    val_jentry_offset: usize,
    key_offset: usize,
    val_offset: usize,
    length: usize,
    index: usize,
}

impl<'a> ObjectIterator<'a> {
    pub(crate) fn new(raw_jsonb: RawJsonb<'a>) -> Result<Option<Self>> {
        let (header_type, header_len) = raw_jsonb.read_header(0)?;
        if header_type == OBJECT_CONTAINER_TAG {
            Ok(Some(Self::new_with_len(raw_jsonb, header_len)?))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn new_with_len(raw_jsonb: RawJsonb<'a>, length: usize) -> Result<Self> {
        let mut key_jentry_offset = 4;
        let mut key_length = 0;
        for _ in 0..length {
            let key_jentry = raw_jsonb.read_jentry(key_jentry_offset)?;
            key_jentry_offset += 4;
            key_length += key_jentry.length as usize;
        }

        let key_offset = 4 + 8 * length;
        Ok(Self {
            raw_jsonb,
            key_jentry_offset: 4,
            val_jentry_offset: 4 + 4 * length,
            key_offset,
            val_offset: key_offset + key_length,
            length,
            index: 0,
        })
    }

    fn remaining(&self) -> usize {
        self.length - self.index
    }
}

impl<'a> Iterator for ObjectIterator<'a> {
    type Item = Result<(&'a str, JsonbItem<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            return None;
        }

        let key_jentry = match self.raw_jsonb.read_jentry(self.key_jentry_offset) {
            Ok(jentry) => jentry,
            Err(err) => return Some(Err(err)),
        };
        let val_jentry = match self.raw_jsonb.read_jentry(self.val_jentry_offset) {
            Ok(jentry) => jentry,
            Err(err) => return Some(Err(err)),
        };
        let key_length = key_jentry.length as usize;
        let val_length = val_jentry.length as usize;

        let key_range = Range {
            start: self.key_offset,
            end: self.key_offset + key_length,
        };
        let key_data = match self.raw_jsonb.slice(key_range) {
            Ok(data) => data,
            Err(err) => return Some(Err(err)),
        };
        let key = unsafe { std::str::from_utf8_unchecked(key_data) };

        let val_range = Range {
            start: self.val_offset,
            end: self.val_offset + val_length,
        };
        let val_data = match self.raw_jsonb.slice(val_range) {
            Ok(data) => data,
            Err(err) => return Some(Err(err)),
        };
        let val_item = jentry_to_jsonb_item(val_jentry, val_data);

        self.index += 1;
        self.key_jentry_offset += 4;
        self.val_jentry_offset += 4;
        self.key_offset += key_length;
        self.val_offset += val_length;

        Some(Ok((key, val_item)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ObjectIterator<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OwnedJsonb;

    fn owned(json: &str) -> OwnedJsonb {
        json.parse().unwrap()
    }

    fn number_string(item: JsonbItem<'_>) -> String {
        match item {
            JsonbItem::Number(value) => value.as_number().unwrap().to_string(),
            _ => panic!("expected number item"),
        }
    }

    fn string_value(item: JsonbItem<'_>) -> String {
        match item {
            JsonbItem::String(value) => value.into_owned(),
            _ => panic!("expected string item"),
        }
    }

    #[test]
    fn array_iterator_reports_exact_remaining_items() {
        let jsonb = owned(r#"[1,"two",true,null]"#);
        let mut iter = ArrayIterator::new(jsonb.as_raw()).unwrap().unwrap();

        assert_eq!(iter.len(), 4);
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(number_string(iter.next().unwrap().unwrap()), "1");
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(string_value(iter.next().unwrap().unwrap()), "two");
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.next().unwrap().unwrap(), JsonbItem::Boolean(true));
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.next().unwrap().unwrap(), JsonbItem::Null);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn empty_array_iterator_reports_zero_remaining_items() {
        let jsonb = owned("[]");
        let mut iter = ArrayIterator::new(jsonb.as_raw()).unwrap().unwrap();

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn object_iterator_preserves_key_value_pairs_and_size_hint() {
        let jsonb = owned(r#"{"first":1,"second":"two","third":true}"#);
        let mut iter = ObjectIterator::new(jsonb.as_raw()).unwrap().unwrap();

        assert_eq!(iter.len(), 3);
        assert_eq!(iter.size_hint(), (3, Some(3)));

        let (key, value) = iter.next().unwrap().unwrap();
        assert_eq!(key, "first");
        assert_eq!(number_string(value), "1");
        assert_eq!(iter.size_hint(), (2, Some(2)));

        let (key, value) = iter.next().unwrap().unwrap();
        assert_eq!(key, "second");
        assert_eq!(string_value(value), "two");
        assert_eq!(iter.size_hint(), (1, Some(1)));

        let (key, value) = iter.next().unwrap().unwrap();
        assert_eq!(key, "third");
        assert_eq!(value, JsonbItem::Boolean(true));
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn empty_object_iterator_reports_zero_remaining_items() {
        let jsonb = owned("{}");
        let mut iter = ObjectIterator::new(jsonb.as_raw()).unwrap().unwrap();

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn object_value_iterator_reports_exact_remaining_items() {
        let jsonb = owned(r#"{"first":1,"second":"two","third":true}"#);
        let mut values = ObjectValueIterator::new(jsonb.as_raw()).unwrap().unwrap();

        assert_eq!(values.len(), 3);
        assert_eq!(values.size_hint(), (3, Some(3)));
        assert_eq!(number_string(values.next().unwrap().unwrap()), "1");
        assert_eq!(values.size_hint(), (2, Some(2)));
        assert_eq!(string_value(values.next().unwrap().unwrap()), "two");
        assert_eq!(values.size_hint(), (1, Some(1)));
        assert_eq!(values.next().unwrap().unwrap(), JsonbItem::Boolean(true));
        assert_eq!(values.size_hint(), (0, Some(0)));
        assert!(values.next().is_none());
    }
}
