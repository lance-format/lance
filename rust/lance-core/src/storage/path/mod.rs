// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Storage object keys, independent of backend implementations.
//!
//! Adapted from Apache Arrow object_store 0.14.2 to preserve existing Lance key encoding.

use percent_encoding::percent_decode;
use std::fmt::Formatter;
#[cfg(not(target_arch = "wasm32"))]
use url::Url;

/// The delimiter to separate object namespaces, creating a directory structure.
pub const DELIMITER: &str = "/";

/// The path delimiter as a single byte
pub const DELIMITER_BYTE: u8 = DELIMITER.as_bytes()[0];

/// The path delimiter as a single char
pub const DELIMITER_CHAR: char = DELIMITER_BYTE as char;

mod parts;

pub use parts::{InvalidPart, PathPart, PathParts};

/// Error returned by [`Path::parse`]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Error when there's an empty segment between two slashes `/` in the path
    #[error("Path \"{}\" contained empty path segment", path)]
    EmptySegment {
        /// The source path
        path: String,
    },

    /// Error when an invalid segment is encountered in the given path
    #[error("Error parsing Path \"{}\": {}", path, source)]
    BadSegment {
        /// The source path
        path: String,
        /// The part containing the error
        source: InvalidPart,
    },

    /// Error when path cannot be canonicalized
    #[error("Failed to canonicalize path \"{}\": {}", path.display(), source)]
    Canonicalize {
        /// The source path
        path: std::path::PathBuf,
        /// The underlying error
        source: std::io::Error,
    },

    /// Error when the path is not a valid URL
    #[error("Unable to convert path \"{}\" to URL", path.display())]
    InvalidPath {
        /// The source path
        path: std::path::PathBuf,
    },

    /// Error when a path contains non-unicode characters
    #[error("Path \"{}\" contained non-unicode characters: {}", path, source)]
    NonUnicode {
        /// The source path
        path: String,
        /// The underlying `UTF8Error`
        source: std::str::Utf8Error,
    },
}

/// A parsed path representation that can be safely written to object storage
///
/// A [`Path`] maintains the following invariants:
///
/// * Paths are delimited by `/`
/// * Paths do not contain leading or trailing `/`
/// * Paths do not contain relative path segments, i.e. `.` or `..`
/// * Paths do not contain empty path segments
/// * Paths do not contain any ASCII control characters
///
/// There are no enforced restrictions on path length, however, it should be noted that most
/// object stores do not permit paths longer than 1024 bytes, and many filesystems do not
/// support path segments longer than 255 bytes.
///
/// # Encode
///
/// In theory object stores support any UTF-8 character sequence, however, certain character
/// sequences cause compatibility problems with some applications and protocols. Additionally
/// some filesystems may impose character restrictions, see [`LocalFileSystem`]. As such the
/// naming guidelines for [S3], [GCS] and [Azure Blob Storage] all recommend sticking to a
/// limited character subset.
///
/// [S3]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
/// [GCS]: https://cloud.google.com/storage/docs/naming-objects
/// [Azure Blob Storage]: https://docs.microsoft.com/en-us/rest/api/storageservices/Naming-and-Referencing-Containers--Blobs--and-Metadata#blob-names
///
/// A string containing potentially problematic path segments can therefore be encoded to a [`Path`]
/// using [`Path::from`] or [`Path::from_iter`]. This will percent encode any problematic
/// segments according to [RFC 1738].
///
/// ```
/// # use lance_core::storage::path::Path;
/// assert_eq!(Path::from("foo/bar").as_ref(), "foo/bar");
/// assert_eq!(Path::from("foo//bar").as_ref(), "foo/bar");
/// assert_eq!(Path::from("foo/../bar").as_ref(), "foo/%2E%2E/bar");
/// assert_eq!(Path::from("/").as_ref(), "");
/// assert_eq!(Path::from_iter(["foo", "foo/bar"]).as_ref(), "foo/foo%2Fbar");
/// ```
///
/// Note: if provided with an already percent encoded string, this will encode it again
///
/// ```
/// # use lance_core::storage::path::Path;
/// assert_eq!(Path::from("foo/foo%2Fbar").as_ref(), "foo/foo%252Fbar");
/// ```
///
/// # Parse
///
/// Alternatively a [`Path`] can be parsed from an existing string, returning an
/// error if it is invalid. Unlike the encoding methods above, this will permit
/// arbitrary unicode, including percent encoded sequences.
///
/// ```
/// # use lance_core::storage::path::Path;
/// assert_eq!(Path::parse("/foo/foo%2Fbar").unwrap().as_ref(), "foo/foo%2Fbar");
/// Path::parse("..").unwrap_err(); // Relative path segments are disallowed
/// Path::parse("/foo//").unwrap_err(); // Empty path segments are disallowed
/// Path::parse("\x00").unwrap_err(); // ASCII control characters are disallowed
/// ```
///
/// [RFC 1738]: https://www.ietf.org/rfc/rfc1738.txt
/// [`LocalFileSystem`]: https://docs.rs/object_store/0.14.2/object_store/local/struct.LocalFileSystem.html
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Path {
    /// The raw path with no leading or trailing delimiters
    raw: String,
}

impl Path {
    /// An empty [`Path`] that points to the root of the store, equivalent to `Path::from("/")`.
    ///
    /// See also [`Path::is_root`].
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_core::storage::path::Path;
    /// assert_eq!(Path::ROOT, Path::from("/"));
    /// ```
    pub const ROOT: Self = Self { raw: String::new() };

    /// Parse a string as a [`Path`], returning a [`Error`] if invalid,
    /// as defined on the docstring for [`Path`]
    ///
    /// Note: this will strip any leading `/` or trailing `/`
    pub fn parse(path: impl AsRef<str>) -> Result<Self, Error> {
        let path = path.as_ref();

        let stripped = path.strip_prefix(DELIMITER).unwrap_or(path);
        if stripped.is_empty() {
            return Ok(Default::default());
        }

        let stripped = stripped.strip_suffix(DELIMITER).unwrap_or(stripped);

        for segment in stripped.split(DELIMITER) {
            if segment.is_empty() {
                return Err(Error::EmptySegment { path: path.into() });
            }

            PathPart::parse(segment).map_err(|source| {
                let path = path.into();
                Error::BadSegment { source, path }
            })?;
        }

        Ok(Self {
            raw: stripped.to_string(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Convert a filesystem path to a [`Path`] relative to the filesystem root
    ///
    /// This will return an error if the path contains illegal character sequences
    /// as defined on the docstring for [`Path`] or does not exist
    ///
    /// Note: this will canonicalize the provided path, resolving any symlinks
    pub fn from_filesystem_path(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let absolute = std::fs::canonicalize(&path).map_err(|source| {
            let path = path.as_ref().into();
            Error::Canonicalize { source, path }
        })?;

        Self::from_absolute_path(absolute)
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Convert an absolute filesystem path to a [`Path`] relative to the filesystem root
    ///
    /// This will return an error if the path contains illegal character sequences,
    /// as defined on the docstring for [`Path`], or `path` is not an absolute path
    pub fn from_absolute_path(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let url = absolute_path_to_url(path)?;

        // Reverse any percent encoding performed by conversion to URL
        Self::from_url_path(url.path())
    }

    /// Parse a url encoded string as a [`Path`], returning a [`Error`] if invalid
    ///
    /// This will return an error if the path contains illegal character sequences
    /// as defined on the docstring for [`Path`]
    pub fn from_url_path(path: impl AsRef<str>) -> Result<Self, Error> {
        let path = path.as_ref();
        let decoded = percent_decode(path.as_bytes())
            .decode_utf8()
            .map_err(|source| {
                let path = path.into();
                Error::NonUnicode { source, path }
            })?;

        Self::parse(decoded)
    }

    /// Returns the number of [`PathPart`]s in this [`Path`]
    ///
    /// This is equivalent to calling `.parts().count()` manually.
    ///
    /// # Performance
    ///
    /// This operation is `O(n)`.
    #[doc(alias = "len")]
    pub fn parts_count(&self) -> usize {
        self.raw.split_terminator(DELIMITER).count()
    }

    /// True if this [`Path`] points to the root of the store, equivalent to `Path::from("/")`.
    ///
    /// See also [`Path::ROOT`].
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_core::storage::path::Path;
    /// assert!(Path::from("/").is_root());
    /// assert!(Path::parse("").unwrap().is_root());
    /// ```
    pub fn is_root(&self) -> bool {
        self.raw.is_empty()
    }

    /// Returns the [`PathPart`]s of this [`Path`]
    ///
    /// Equivalent to calling `.into_iter()` on a `&Path`.
    pub fn parts(&self) -> PathParts<'_> {
        PathParts::new(&self.raw)
    }

    /// Returns a copy of this [`Path`] with the last path segment removed
    ///
    /// Returns `None` if this path has zero segments.
    pub fn parent(&self) -> Option<Self> {
        if self.raw.is_empty() {
            return None;
        }

        let Some((prefix, _filename)) = self.raw.rsplit_once(DELIMITER) else {
            return Some(Self::ROOT);
        };

        Some(Self {
            raw: prefix.to_string(),
        })
    }

    /// Returns the last path segment containing the filename stored in this [`Path`]
    ///
    /// Returns `None` only if this path is the root path.
    pub fn filename(&self) -> Option<&str> {
        match self.raw.is_empty() {
            true => None,
            false => self.raw.rsplit(DELIMITER).next(),
        }
    }

    /// Returns the extension of the file stored in this [`Path`], if any
    pub fn extension(&self) -> Option<&str> {
        self.filename()
            .and_then(|f| f.rsplit_once('.'))
            .and_then(|(_, extension)| {
                if extension.is_empty() {
                    None
                } else {
                    Some(extension)
                }
            })
    }

    /// Returns an iterator of the [`PathPart`] of this [`Path`] after `prefix`
    ///
    /// Returns `None` if the prefix does not match.
    pub fn prefix_match(&self, prefix: &Self) -> Option<impl Iterator<Item = PathPart<'_>> + '_> {
        let mut stripped = self.raw.strip_prefix(&prefix.raw)?;
        if !stripped.is_empty() && !prefix.raw.is_empty() {
            stripped = stripped.strip_prefix(DELIMITER)?;
        }
        Some(PathParts::new(stripped))
    }

    /// Returns true if this [`Path`] starts with `prefix`
    pub fn prefix_matches(&self, prefix: &Self) -> bool {
        self.prefix_match(prefix).is_some()
    }

    /// Creates a new child of this [`Path`]
    pub fn child<'a>(&self, child: impl Into<PathPart<'a>>) -> Self {
        self.clone().join(child)
    }

    /// Appends a single path segment to this [`Path`]
    pub fn join<'a>(self, child: impl Into<PathPart<'a>>) -> Self {
        let child_cow_str = child.into().raw;

        let raw = if self.raw.is_empty() {
            child_cow_str.to_string()
        } else {
            let mut raw = self.raw;
            raw.push_str(DELIMITER);
            raw.push_str(&child_cow_str);
            raw
        };

        Self { raw }
    }
}

impl AsRef<str> for Path {
    fn as_ref(&self) -> &str {
        &self.raw
    }
}

impl From<&str> for Path {
    fn from(path: &str) -> Self {
        Self::from_iter(path.split(DELIMITER))
    }
}

impl From<String> for Path {
    fn from(path: String) -> Self {
        Self::from_iter(path.split(DELIMITER))
    }
}

impl From<Path> for String {
    fn from(path: Path) -> Self {
        path.raw
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(f)
    }
}

impl<'a, I> FromIterator<I> for Path
where
    I: Into<PathPart<'a>>,
{
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let mut this = Self::ROOT;
        this.extend(iter);
        this
    }
}

/// See also [`Path::parts`]
impl<'a> IntoIterator for &'a Path {
    type Item = PathPart<'a>;
    type IntoIter = PathParts<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PathParts::new(&self.raw)
    }
}

/// [`Path`] supports appending [`PathPart`]s of one `Path` to another `Path`.
///
/// # Examples
///
/// Suppose Alice is copying Bob's file to her own user directory.
/// We could choose the full path of the new file by taking the original
/// absolute path, making it relative to Bob's home
///
/// ```rust
/// # use lance_core::storage::path::Path;
/// let alice_home = Path::from("Users/alice");
/// let bob_home = Path::from("Users/bob");
/// let bob_file = Path::from("Users/bob/documents/file.txt");
///
/// let mut alice_file = alice_home;
/// alice_file.extend(bob_file.prefix_match(&bob_home).unwrap());
///
/// assert_eq!(alice_file, Path::from("Users/alice/documents/file.txt"));
/// ```
impl<'a, I: Into<PathPart<'a>>> Extend<I> for Path {
    fn extend<T: IntoIterator<Item = I>>(&mut self, iter: T) {
        for s in iter {
            let s = s.into();
            if !s.raw.is_empty() {
                if !self.raw.is_empty() {
                    self.raw.push(DELIMITER_CHAR);
                }
                self.raw.push_str(&s.raw);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Given an absolute filesystem path convert it to a URL representation without canonicalization
pub(crate) fn absolute_path_to_url(path: impl AsRef<std::path::Path>) -> Result<Url, Error> {
    Url::from_file_path(&path).map_err(|_| Error::InvalidPath {
        path: path.as_ref().into(),
    })
}

#[cfg(test)]
mod tests {
    use super::Path;
    use rstest::rstest;

    #[rstest]
    #[case("a//b", "a/b")]
    #[case("a/../b", "a/%2E%2E/b")]
    #[case("a/%2Fb", "a/%252Fb")]
    #[case("/", "")]
    fn encoding_preserves_existing_key_conventions(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(Path::from(input).as_ref(), expected);
    }

    #[rstest]
    #[case("a//b")]
    #[case("a/../b")]
    #[case("a/./b")]
    #[case("a/\0b")]
    fn parsing_rejects_invalid_segments(#[case] input: &str) {
        assert!(Path::parse(input).is_err());
    }

    #[test]
    fn parsing_and_url_decoding_have_distinct_meanings() {
        assert_eq!(Path::parse("a%2Fb").unwrap().as_ref(), "a%2Fb");
        assert_eq!(Path::from_url_path("a%2Fb").unwrap().as_ref(), "a/b");
        assert_eq!(Path::from_url_path("a%252Fb").unwrap().as_ref(), "a%2Fb");
        assert_eq!(Path::from("a").join("b/c").as_ref(), "a/b%2Fc");
        assert!(!Path::from("database/a").prefix_matches(&Path::from("data")));
        assert!(Path::from("data/a").prefix_matches(&Path::from("data")));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn local_paths_do_not_decode_literal_percent_sequences_twice() {
        let base = std::env::temp_dir();
        let file = base.join("a %2F#é");
        let path = Path::from_absolute_path(&file).unwrap();
        assert_eq!(path.filename(), Some("a %2F#é"));
    }
}
