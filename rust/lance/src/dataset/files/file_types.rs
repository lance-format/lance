// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Discriminants are the dictionary keys used in the `tracked_files` output
// schema; they must stay in sync with `FILE_TYPE_DICT_ARRAY` in `arrow.rs`.
#[repr(i8)]
#[derive(Debug, Clone, Copy)]
pub enum FileType {
    Manifest = 0,
    DataFile = 1,
    DeletionFile = 2,
    TransactionFile = 3,
    IndexFile = 4,
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Manifest => "manifest",
            Self::DataFile => "data file",
            Self::DeletionFile => "deletion file",
            Self::TransactionFile => "transaction file",
            Self::IndexFile => "index file",
        };
        write!(f, "{s}")
    }
}

impl From<FileType> for i8 {
    fn from(file_type: FileType) -> Self {
        file_type as Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Array, StringArray, cast::AsArray};

    use crate::dataset::files::arrow::FILE_TYPE_DICT_ARRAY;

    const ALL: [FileType; 5] = [
        FileType::Manifest,
        FileType::DataFile,
        FileType::DeletionFile,
        FileType::TransactionFile,
        FileType::IndexFile,
    ];

    /// The discriminants double as dictionary keys for the `tracked_files`
    /// output, so reordering either list would silently mislabel every row.
    #[test]
    fn test_discriminants_index_into_the_dictionary() {
        let dict: &StringArray = FILE_TYPE_DICT_ARRAY.as_string();
        assert_eq!(
            dict.len(),
            ALL.len(),
            "every variant needs a dictionary slot"
        );

        for file_type in ALL {
            let key = i8::from(file_type);
            assert_eq!(
                dict.value(key as usize),
                file_type.to_string(),
                "{file_type:?} has key {key}"
            );
        }
    }

    #[test]
    fn test_discriminants_are_contiguous_from_zero() {
        let keys: Vec<i8> = ALL.iter().copied().map(i8::from).collect();
        assert_eq!(keys, (0..ALL.len() as i8).collect::<Vec<_>>());
    }
}
