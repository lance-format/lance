// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use lance_arrow::{
    BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
    BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};

pub const INLINE_MAX: usize = 64 * 1024; // 64KB inline cutoff
pub const DEDICATED_THRESHOLD: usize = 4 * 1024 * 1024; // 4MB dedicated cutoff
const PACK_FILE_MAX_SIZE: usize = 1024 * 1024 * 1024; // 1GiB per .pack sidecar

pub fn blob_inline_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
) -> Result<usize> {
    blob_threshold_from_metadata(
        metadata,
        field_name,
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
        INLINE_MAX,
        true,
    )
}

pub fn blob_dedicated_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
) -> Result<usize> {
    blob_threshold_from_metadata(
        metadata,
        field_name,
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
        DEDICATED_THRESHOLD,
        false,
    )
}

pub fn blob_pack_file_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
) -> Result<usize> {
    blob_threshold_from_metadata(
        metadata,
        field_name,
        BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
        PACK_FILE_MAX_SIZE,
        false,
    )
}

fn blob_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
    key: &str,
    default_value: usize,
    allow_zero: bool,
) -> Result<usize> {
    let Some(value) = metadata.get(key) else {
        return Ok(default_value);
    };
    let threshold = value.parse::<usize>().map_err(|_| {
        Error::invalid_input(format!(
            "Invalid blob threshold metadata {key}={value:?} for field '{field_name}'; \
             expected a non-negative integer that fits in usize"
        ))
    })?;
    if !allow_zero && threshold == 0 {
        return Err(Error::invalid_input(format!(
            "Invalid blob threshold metadata {key}={value:?} for field '{field_name}'; \
             expected a positive integer"
        )));
    }
    Ok(threshold)
}

pub fn validate_blob_threshold_metadata(schema: &Schema) -> Result<()> {
    for field in schema.fields_pre_order().filter(|field| field.is_blob_v2()) {
        blob_inline_threshold_from_metadata(&field.metadata, &field.name)?;
        blob_dedicated_threshold_from_metadata(&field.metadata, &field.name)?;
        blob_pack_file_threshold_from_metadata(&field.metadata, &field.name)?;
    }
    Ok(())
}
