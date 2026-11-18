// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::storage::*;
use ::object_store as native;
use futures::{StreamExt, TryStreamExt};

pub fn to_path(path: &Path) -> Result<native::path::Path> {
    native::path::Path::parse(path.as_ref()).map_err(|source| Error::InvalidPath {
        source: Box::new(source),
    })
}

/// Preserve native error categories at the provider boundary.
/// Unimplemented operations become NotSupported so callers have one fallback rule.
pub fn from_error(error: native::Error) -> Error {
    match error {
        native::Error::Generic { store, source } => match source.downcast::<Error>() {
            Ok(error) => *error,
            Err(source) => Error::Generic { store, source },
        },
        native::Error::InvalidPath { source } => Error::InvalidPath {
            source: Box::new(source),
        },
        error @ native::Error::NotImplemented { .. } => Error::NotSupported {
            source: Box::new(error),
        },
        native::Error::NotSupported { source } => Error::NotSupported { source },
        error @ native::Error::UnknownConfigurationKey { .. } => Error::InvalidInput {
            source: Box::new(error),
        },
        native::Error::NotFound { path, source } => Error::NotFound { path, source },
        native::Error::AlreadyExists { path, source } => Error::AlreadyExists { path, source },
        native::Error::Precondition { path, source } => Error::Precondition { path, source },
        native::Error::NotModified { path, source } => Error::NotModified { path, source },
        native::Error::PermissionDenied { path, source } => {
            Error::PermissionDenied { path, source }
        }
        native::Error::Unauthenticated { path, source } => Error::Unauthenticated { path, source },
        other => Error::Generic {
            store: "object_store",
            source: Box::new(other),
        },
    }
}

pub fn to_error(source: Error) -> native::Error {
    native::Error::Generic {
        store: "lance",
        source: Box::new(source),
    }
}

impl From<PutPayload> for native::PutPayload {
    fn from(payload: PutPayload) -> Self {
        payload.iter().cloned().collect()
    }
}

impl From<Attribute> for native::Attribute {
    fn from(value: Attribute) -> Self {
        match value {
            Attribute::ContentDisposition => Self::ContentDisposition,
            Attribute::ContentEncoding => Self::ContentEncoding,
            Attribute::ContentLanguage => Self::ContentLanguage,
            Attribute::ContentType => Self::ContentType,
            Attribute::CacheControl => Self::CacheControl,
            Attribute::StorageClass => Self::StorageClass,
            Attribute::Metadata(key) => Self::Metadata(key),
        }
    }
}

fn to_attributes(values: Attributes) -> native::Attributes {
    values
        .into_iter()
        .map(|(key, value)| {
            (
                native::Attribute::from(key),
                match value {
                    std::borrow::Cow::Borrowed(value) => native::AttributeValue::from(value),
                    std::borrow::Cow::Owned(value) => native::AttributeValue::from(value),
                },
            )
        })
        .collect()
}

fn from_attributes(values: native::Attributes) -> Result<Attributes> {
    values
        .iter()
        .map(|(key, value)| {
            let key = match key {
                native::Attribute::ContentDisposition => Attribute::ContentDisposition,
                native::Attribute::ContentEncoding => Attribute::ContentEncoding,
                native::Attribute::ContentLanguage => Attribute::ContentLanguage,
                native::Attribute::ContentType => Attribute::ContentType,
                native::Attribute::CacheControl => Attribute::CacheControl,
                native::Attribute::StorageClass => Attribute::StorageClass,
                native::Attribute::Metadata(key) => Attribute::Metadata(key.clone()),
                other => {
                    return Err(Error::NotSupported {
                        source: format!("Unsupported object attribute {other:?}").into(),
                    });
                }
            };
            Ok((key, std::borrow::Cow::Owned(value.as_ref().to_owned())))
        })
        .collect()
}

impl From<PutMode> for native::PutMode {
    fn from(mode: PutMode) -> Self {
        match mode {
            PutMode::Overwrite => Self::Overwrite,
            PutMode::Create => Self::Create,
            PutMode::Update(version) => Self::Update(native::UpdateVersion {
                e_tag: version.e_tag,
                version: version.version,
            }),
        }
    }
}
impl From<CopyMode> for native::CopyMode {
    fn from(mode: CopyMode) -> Self {
        match mode {
            CopyMode::Overwrite => Self::Overwrite,
            CopyMode::Create => Self::Create,
        }
    }
}
impl From<RenameTargetMode> for native::RenameTargetMode {
    fn from(mode: RenameTargetMode) -> Self {
        match mode {
            RenameTargetMode::Overwrite => Self::Overwrite,
            RenameTargetMode::Create => Self::Create,
        }
    }
}
impl From<GetOptions> for native::GetOptions {
    fn from(value: GetOptions) -> Self {
        Self {
            if_match: value.if_match,
            if_none_match: value.if_none_match,
            range: value.range.map(native::GetRange::Bounded),
            head: value.head,
            extensions: value.extensions,
            ..Default::default()
        }
    }
}
impl From<PutOptions> for native::PutOptions {
    fn from(value: PutOptions) -> Self {
        Self {
            mode: value.mode.into(),
            attributes: to_attributes(value.attributes),
            extensions: value.extensions,
            ..Default::default()
        }
    }
}
impl From<PutMultipartOptions> for native::PutMultipartOptions {
    fn from(value: PutMultipartOptions) -> Self {
        Self {
            attributes: to_attributes(value.attributes),
            extensions: value.extensions,
            ..Default::default()
        }
    }
}
impl From<CopyOptions> for native::CopyOptions {
    fn from(value: CopyOptions) -> Self {
        Self {
            mode: value.mode.into(),
            extensions: value.extensions,
        }
    }
}
impl From<RenameOptions> for native::RenameOptions {
    fn from(value: RenameOptions) -> Self {
        Self {
            target_mode: value.target_mode.into(),
            extensions: value.extensions,
        }
    }
}
impl From<PaginatedListOptions> for native::list::PaginatedListOptions {
    fn from(value: PaginatedListOptions) -> Self {
        Self {
            offset: value.offset,
            delimiter: value.delimiter,
            max_keys: value.max_keys,
            page_token: value.page_token,
            extensions: value.extensions,
        }
    }
}
impl From<native::PutResult> for PutResult {
    fn from(value: native::PutResult) -> Self {
        Self {
            e_tag: value.e_tag,
            version: value.version,
            extensions: value.extensions,
        }
    }
}

impl TryFrom<native::ObjectMeta> for ObjectMeta {
    type Error = Error;
    fn try_from(value: native::ObjectMeta) -> Result<Self> {
        Ok(Self {
            location: Path::parse(value.location.as_ref())?,
            last_modified: value.last_modified,
            size: value.size,
            e_tag: value.e_tag,
            version: value.version,
        })
    }
}

impl TryFrom<native::ListResult> for ListResult {
    type Error = Error;
    fn try_from(value: native::ListResult) -> Result<Self> {
        Ok(Self {
            common_prefixes: value
                .common_prefixes
                .into_iter()
                .map(|p| Path::parse(p.as_ref()).map_err(Error::from))
                .collect::<Result<_>>()?,
            objects: value
                .objects
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
            extensions: value.extensions,
        })
    }
}

impl TryFrom<native::GetResult> for GetResult {
    type Error = Error;
    fn try_from(value: native::GetResult) -> Result<Self> {
        Ok(Self {
            payload: match value.payload {
                native::GetResultPayload::File(file, path) => GetResultPayload::File(file, path),
                native::GetResultPayload::Stream(stream) => {
                    GetResultPayload::Stream(stream.map_err(from_error).boxed())
                }
            },
            meta: value.meta.try_into()?,
            range: value.range,
            attributes: from_attributes(value.attributes)?,
            extensions: value.extensions,
        })
    }
}
