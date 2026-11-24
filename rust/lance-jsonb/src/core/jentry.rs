// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use super::constants::*;

#[derive(Clone, Debug, PartialOrd, PartialEq, Eq, Ord)]
pub(super) struct JEntry {
    pub(super) type_code: u32,
    pub(super) length: u32,
}

impl JEntry {
    pub(super) fn decode_jentry(encoded: u32) -> Self {
        let type_code = encoded & JENTRY_TYPE_MASK;
        let length = encoded & JENTRY_OFF_LEN_MASK;
        Self { type_code, length }
    }

    pub(super) fn make_null_jentry() -> Self {
        Self {
            type_code: NULL_TAG,
            length: 0,
        }
    }

    pub(super) fn make_true_jentry() -> Self {
        Self {
            type_code: TRUE_TAG,
            length: 0,
        }
    }

    pub(super) fn make_false_jentry() -> Self {
        Self {
            type_code: FALSE_TAG,
            length: 0,
        }
    }

    pub(super) fn make_string_jentry(length: usize) -> Self {
        Self {
            type_code: STRING_TAG,
            length: length as u32,
        }
    }

    pub(super) fn make_number_jentry(length: usize) -> Self {
        Self {
            type_code: NUMBER_TAG,
            length: length as u32,
        }
    }

    pub(super) fn make_container_jentry(length: usize) -> Self {
        Self {
            type_code: CONTAINER_TAG,
            length: length as u32,
        }
    }

    pub(super) fn make_extension_jentry(length: usize) -> Self {
        Self {
            type_code: EXTENSION_TAG,
            length: length as u32,
        }
    }

    pub(super) fn encoded(&self) -> u32 {
        self.type_code | self.length
    }
}
