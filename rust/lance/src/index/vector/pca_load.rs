// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Load the PCA model stored in `index.idx`.

use std::collections::HashMap;

use lance_index::VECTOR_PCA_INDEX_VERSION;
use lance_index::vector::pca::PCA_METADATA_KEY;

use crate::{Error, Result};

/// Checks the version-4 contract against schema metadata.
///
/// Version 4 requires `lance:pca` and physical type `IVF_RQ`. Every other
/// index metadata version must not carry that key. The buffer itself is
/// decoded separately.
pub fn validate_pca_metadata(
    metadata: &HashMap<String, String>,
    index_version: i32,
    index_type: &str,
) -> Result<()> {
    let has_model = metadata.contains_key(PCA_METADATA_KEY);
    let pca_version = index_version == VECTOR_PCA_INDEX_VERSION as i32;
    if pca_version {
        if index_type != "IVF_RQ" {
            return Err(Error::corrupt_file_named(
                "index.idx",
                format!(
                    "index metadata version {index_version} requires physical type IVF_RQ, got {index_type}"
                ),
            ));
        }
        if !has_model {
            return Err(Error::corrupt_file_named(
                "index.idx",
                format!(
                    "index metadata version {index_version} requires {PCA_METADATA_KEY} in index.idx"
                ),
            ));
        }
        return Ok(());
    }
    if has_model {
        return Err(Error::corrupt_file_named(
            "index.idx",
            format!(
                "{PCA_METADATA_KEY} is only valid for index metadata version {VECTOR_PCA_INDEX_VERSION}, got {index_version}"
            ),
        ));
    }
    Ok(())
}

pub fn pca_buffer_index(metadata: &HashMap<String, String>) -> Result<u32> {
    let raw = metadata.get(PCA_METADATA_KEY).ok_or_else(|| {
        Error::corrupt_file_named(
            "index.idx",
            format!(
                "index metadata version {VECTOR_PCA_INDEX_VERSION} is missing {PCA_METADATA_KEY}"
            ),
        )
    })?;
    raw.parse::<u32>().map_err(|_| {
        Error::corrupt_file_named(
            "index.idx",
            format!("{PCA_METADATA_KEY} buffer index {raw:?} is not a u32"),
        )
    })
}
