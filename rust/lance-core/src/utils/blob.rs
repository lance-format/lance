// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path;

use crate::{Error, Result};

/// Validate a Managed descriptor's object-relative path and known byte range.
///
/// No base ID is reserved. Resolving the base belongs to the snapshot holding
/// the descriptor, and must fail if that snapshot has no matching binding.
pub fn validate_managed_reference(uri: &str, position: u64, size: u64) -> Result<Path> {
    if uri.is_empty()
        || uri.starts_with('/')
        || uri.ends_with('/')
        || uri.contains("://")
        || uri.contains('\\')
        || uri
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(Error::invalid_input(format!(
            "Managed blob_uri must be a non-empty canonical relative object path, got {uri:?}"
        )));
    }
    position.checked_add(size).ok_or_else(|| {
        Error::invalid_input(format!(
            "Managed blob range overflows u64: position={position}, size={size}"
        ))
    })?;
    Path::parse(uri)
        .map_err(|error| Error::invalid_input(format!("Invalid Managed blob_uri {uri:?}: {error}")))
}

/// Format a blob sidecar path for a data file.
///
/// Layout: `<base>/<data_file_key>/<obfuscated_blob_id>.blob`
/// - `base` is typically the dataset's data directory.
/// - `data_file_key` is the stem of the data file (without extension).
/// - `blob_id` is transformed via `reverse_bits()` before binary formatting.
pub fn blob_path(base: &Path, data_file_key: &str, blob_id: u32) -> Path {
    let file_name = format!("{:032b}.blob", blob_id.reverse_bits());
    base.clone().join(data_file_key).join(file_name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("")]
    #[case("/data/a.blob")]
    #[case("data/../a.blob")]
    #[case("data/./a.blob")]
    #[case("data//a.blob")]
    #[case("data/a.blob/")]
    #[case("s3://bucket/a.blob")]
    #[case("data\\a.blob")]
    fn managed_paths_reject_noncanonical_references(#[case] uri: &str) {
        let error = validate_managed_reference(uri, 0, 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("Managed blob_uri"));
    }

    #[test]
    fn managed_ranges_preserve_empty_values_and_reject_overflow() {
        assert!(validate_managed_reference("data/a.blob", u64::MAX, 0).is_ok());
        let error = validate_managed_reference("data/a.blob", u64::MAX, 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("position=18446744073709551615, size=1")
        );
    }

    #[test]
    fn test_blob_path_formatting() {
        let base = Path::from("base");
        let path = blob_path(&base, "deadbeef", 2);
        assert_eq!(
            path.to_string(),
            "base/deadbeef/01000000000000000000000000000000.blob"
        );
    }

    #[test]
    fn test_blob_path_scattered_prefixes_for_sequential_ids() {
        let base = Path::from("base");
        let p1 = blob_path(&base, "deadbeef", 1);
        let p2 = blob_path(&base, "deadbeef", 2);
        assert_ne!(p1.to_string(), p2.to_string());
        assert_eq!(
            p1.to_string(),
            "base/deadbeef/10000000000000000000000000000000.blob"
        );
        assert_eq!(
            p2.to_string(),
            "base/deadbeef/01000000000000000000000000000000.blob"
        );
    }
}
