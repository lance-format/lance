// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use bytes::Bytes;
use futures::{StreamExt, stream};
use lance_core::Error;
use object_store::{GetResult, ObjectMeta, Result};

use super::GetRequest;
use crate::traits::ByteStream;

// One budget bounds the entire download, including requests to reopen a failed body.
async fn request_with_retry(request: &GetRequest, retries: &mut usize) -> Result<GetResult> {
    loop {
        match request
            .object_store
            .get_opts(&request.path, request.options.clone())
            .await
        {
            Ok(result) => return Ok(result),
            Err(error) => {
                if *retries == 0 || !matches!(error, object_store::Error::Generic { .. }) {
                    return Err(error);
                }
                *retries -= 1;
            }
        }
    }
}

pub(super) async fn get_stream(request: GetRequest, mut retries: usize) -> Result<ByteStream> {
    let result = request_with_retry(&request, &mut retries).await?;
    let expected_range = match &request.options.range {
        Some(range) => range.as_range(result.meta.size).map_err(|error| {
            Error::invalid_input(format!(
                "invalid range {range:?} for '{}' with {} bytes: {error}",
                request.path, result.meta.size
            ))
        })?,
        None => 0..result.meta.size,
    };
    if result.range != expected_range {
        return Err(Error::io(format!(
            "object at '{}' returned range {:?}, expected {:?}",
            request.path, result.range, expected_range
        ))
        .into());
    }
    let download = StreamingDownload {
        request,
        remaining: expected_range,
        meta: result.meta.clone(),
        body: result.into_stream(),
        retries,
    };
    Ok(stream::try_unfold(download, |download| download.next()).boxed())
}

struct StreamingDownload {
    request: GetRequest,
    remaining: Range<u64>,
    meta: ObjectMeta,
    body: ByteStream,
    retries: usize,
}

impl StreamingDownload {
    async fn next(mut self) -> Result<Option<(Bytes, Self)>> {
        loop {
            let error = match self.body.next().await {
                Some(Ok(bytes)) => {
                    if bytes.len() as u64 > self.remaining.end - self.remaining.start {
                        let error = Error::io(format!(
                            "object body at '{}' exceeds remaining range {:?}: chunk length {}",
                            self.request.path,
                            self.remaining,
                            bytes.len()
                        ));
                        return Err(error.into());
                    }
                    // The length check above bounds this addition by the range end.
                    self.remaining.start += bytes.len() as u64;
                    return Ok(Some((bytes, self)));
                }
                None if self.remaining.is_empty() => return Ok(None),
                None => Error::io(format!(
                    "object body at '{}' ended before remaining range {:?}",
                    self.request.path, self.remaining
                ))
                .into(),
                Some(Err(error)) => error,
            };
            if self.remaining.is_empty() {
                return Ok(None);
            }
            if self.retries == 0 || !matches!(error, object_store::Error::Generic { .. }) {
                return Err(error);
            }
            // Without either identifier, a resumed body could concatenate different objects.
            if self.meta.e_tag.is_none() && self.meta.version.is_none() {
                return Err(Error::io(format!(
                    "cannot resume object at '{}' without an ETag or version after: {error}",
                    self.request.path
                ))
                .into());
            }
            self.retries -= 1;
            self.request.options.range = Some(self.remaining.clone().into());
            self.request.options.if_match = self.meta.e_tag.clone();
            self.request.options.version = self.meta.version.clone();
            // Release the failed response before opening another connection.
            self.body = stream::empty().boxed();
            let result = request_with_retry(&self.request, &mut self.retries).await?;
            // Also verify metadata: custom stores may ignore conditional GET options.
            if result.meta.e_tag != self.meta.e_tag
                || result.meta.version != self.meta.version
                || result.meta.size != self.meta.size
            {
                return Err(object_store::Error::Precondition {
                    path: self.request.path.to_string(),
                    source: format!(
                        "object changed while resuming a streaming download: \
                         ETag {:?} -> {:?}, version {:?} -> {:?}, size {} -> {}",
                        self.meta.e_tag,
                        result.meta.e_tag,
                        self.meta.version,
                        result.meta.version,
                        self.meta.size,
                        result.meta.size
                    )
                    .into(),
                });
            }
            if result.range != self.remaining {
                return Err(Error::io(format!(
                    "object at '{}' returned range {:?}, expected remaining range {:?}",
                    self.request.path, result.range, self.remaining
                ))
                .into());
            }
            self.body = result.into_stream();
        }
    }
}
