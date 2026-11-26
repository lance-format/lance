// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::future::BoxFuture;
use object_store::{PutMode, PutOptions};
use tokio::task::JoinHandle;

/// Marks a snapshot replacement whose acknowledgement cannot safely be replayed.
#[derive(Clone)]
pub(in crate::object_store) enum ConditionalUpdate {
    Single,
    Multipart(MultipartUpdate),
}

/// Retry individual upload parts without replaying publication. The service
/// writer must stay available for abort when a caller cancels a pending part.
pub(in crate::object_store) fn conditional_operator(
    operator: &opendal::Operator,
    part_retries: usize,
) -> opendal::Operator {
    let context = operator.base_context();
    let transport = ConditionalTransport {
        inner: context.http_transport().clone(),
        part_retries,
    };
    operator
        .clone()
        .with_context(context.with_http_transport(opendal::HttpTransporter::new(transport)))
}

#[derive(Debug)]
struct ConditionalTransport {
    inner: opendal::HttpTransporter,
    part_retries: usize,
}

impl ConditionalTransport {
    async fn part_attempt(
        &self,
        req: http::Request<opendal::Buffer>,
    ) -> opendal::Result<http::Response<opendal::HttpBody>> {
        let response = self.inner.fetch(req).await?;
        if response.status().is_success() {
            return Ok(response);
        }
        let (parts, mut body) = response.into_parts();
        let body = body.to_buffer().await?;
        let code = crate::object_store::throttle::error_document_code(&String::from_utf8_lossy(
            &body.to_bytes(),
        ));
        // S3 reports RequestTimeout as 400 and OperationAborted as 409.
        // Keep permanent S3 errors intact for the service's normal error mapping.
        let retry = match code.as_deref() {
            Some(
                "RequestTimeout"
                | "OperationAborted"
                | "InternalError"
                | "SlowDown"
                | "ServiceUnavailable"
                | "TooManyRequests"
                | "ExceedAccountQPSLimit"
                | "ExceedAccountRateLimit"
                | "ExceedBucketQPSLimit"
                | "ExceedBucketRateLimit",
            ) => true,
            Some("NoSuchBucket" | "NoSuchKey" | "InvalidRange" | "ConditionalRequestConflict") => {
                false
            }
            _ => matches!(
                parts.status.as_u16(),
                408 | 429 | 499 | 500 | 502 | 503 | 504
            ),
        };
        if retry {
            return Err(opendal::Error::new(
                opendal::ErrorKind::Unexpected,
                format!(
                    "S3 UploadPart returned {} ({})",
                    parts.status,
                    code.as_deref().unwrap_or("unknown error")
                ),
            )
            .set_temporary());
        }
        let length = body.len() as u64;
        Ok(http::Response::from_parts(
            parts,
            opendal::HttpBody::new(futures::stream::iter([Ok(body)]), Some(length)),
        ))
    }
}

impl opendal::HttpTransport for ConditionalTransport {
    async fn fetch(
        &self,
        req: http::Request<opendal::Buffer>,
    ) -> opendal::Result<http::Response<opendal::HttpBody>> {
        use opendal::{Error, ErrorKind};
        if req.method() == http::Method::PUT
            && req.uri().query().is_some_and(|q| {
                url::form_urlencoded::parse(q.as_bytes()).any(|(key, _)| key == "partNumber")
            })
        {
            // Reusing the upload ID, part number and bytes replaces the same
            // staged part. Dropping this future leaves the writer free to abort.
            let mut backoff = lance_core::utils::backoff::Backoff::default()
                .with_unit(100)
                .with_max(15_000);
            for _ in 0..self.part_retries {
                match self.part_attempt(req.clone()).await {
                    Err(error) if error.is_temporary() => {
                        tokio::time::sleep(backoff.next_backoff()).await
                    }
                    result => return result,
                }
            }
            return self.inner.fetch(req).await;
        }
        let completion = req.method() == http::Method::POST;
        if !req.headers().contains_key(http::header::IF_MATCH)
            || !(completion || req.method() == http::Method::PUT)
        {
            return self.inner.fetch(req).await;
        }
        let mut response = self.inner.fetch(req).await.map_err(Error::set_permanent)?;
        let status = response.status();
        // The service adapter loses the S3 error code on conditional multipart
        // 404s. A missing upload is distinct from a deleted destination.
        if (completion && status == http::StatusCode::NOT_FOUND)
            || status == http::StatusCode::CONFLICT
        {
            let body = response
                .body_mut()
                .to_buffer()
                .await
                .map_err(Error::set_permanent)?
                .to_bytes();
            let code =
                crate::object_store::throttle::error_document_code(&String::from_utf8_lossy(&body));
            let kind = match (status, code.as_deref()) {
                (http::StatusCode::NOT_FOUND, Some("NoSuchKey"))
                | (
                    http::StatusCode::CONFLICT,
                    Some("ConditionalRequestConflict" | "OperationAborted"),
                ) => ErrorKind::ConditionNotMatch,
                _ => ErrorKind::Unexpected,
            };
            return Err(Error::new(
                kind,
                format!(
                    "S3 conditional write returned {status} ({})",
                    code.as_deref().unwrap_or("unknown error")
                ),
            ));
        }
        Ok(response)
    }
}

#[derive(Clone)]
pub(in crate::object_store) struct MultipartUpdate {
    pub(super) etag: String,
    pub(super) part_size: usize,
    pub(super) concurrency: usize,
}

impl MultipartUpdate {
    pub(in crate::object_store) fn for_payload(
        options: &PutOptions,
        size: usize,
    ) -> object_store::Result<Option<Self>> {
        let PutMode::Update(version) = &options.mode else {
            return Ok(None);
        };
        let initial = crate::object_writer::initial_upload_size();
        let Some(etag) = version.e_tag.as_deref().filter(|_| size > initial) else {
            return Ok(None);
        };
        let concurrency = crate::object_writer::max_upload_parallelism();
        if concurrency == 0 {
            return Err(object_store::Error::Generic {
                store: "S3 conditional update",
                source: "LANCE_UPLOAD_CONCURRENCY must be greater than zero".into(),
            });
        }
        Ok(Some(Self {
            etag: etag.to_owned(),
            part_size: initial.max(size.div_ceil(10_000)),
            concurrency,
        }))
    }

    pub(in crate::object_store) async fn write_opendal(
        &self,
        operator: &opendal::Operator,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
    ) -> object_store::Result<object_store::PutResult> {
        use opendal::ErrorKind;
        let map_error = |error: opendal::Error| match error.kind() {
            ErrorKind::ConditionNotMatch => object_store::Error::Precondition {
                path: location.to_string(),
                source: Box::new(error),
            },
            ErrorKind::AlreadyExists => object_store::Error::AlreadyExists {
                path: location.to_string(),
                source: Box::new(error),
            },
            _ => object_store::Error::Generic {
                store: "S3",
                source: Box::new(error),
            },
        };
        let path = opendal::raw::percent_decode_path(location.as_ref());
        // The bridge sends one write. Exact chunks carry If-Match through
        // OpenDAL's multipart writer to the final completion request.
        let writer = operator
            .writer_with(&path)
            .if_match(&self.etag)
            .chunk(self.part_size)
            .concurrent(self.concurrency)
            .await
            .map_err(map_error)?;
        let mut pending = PendingUpload::new(writer, |mut writer| {
            Box::pin(async move {
                if let Err(error) = writer.abort().await {
                    log::warn!("Failed to abort conditional S3 upload: {error}");
                }
            })
        });
        let writer = pending.resource.as_mut().expect("upload is active");
        let result = async {
            writer.write(opendal::Buffer::from_iter(payload)).await?;
            writer.close().await
        }
        .await;
        let meta = pending.finish(result).await.map_err(map_error)?;
        Ok(object_store::PutResult {
            e_tag: meta.etag().map(str::to_owned),
            version: meta.version().map(str::to_owned),
            extensions: Default::default(),
        })
    }
}

// A cancelled caller must release staged parts even when no result is returned.
pub(super) struct PendingUpload<T: Send + 'static> {
    resource: Option<T>,
    abort: fn(T) -> BoxFuture<'static, ()>,
    runtime: Option<tokio::runtime::Handle>,
}

impl<T: Send + 'static> PendingUpload<T> {
    pub(super) fn new(resource: T, abort: fn(T) -> BoxFuture<'static, ()>) -> Self {
        Self {
            resource: Some(resource),
            abort,
            runtime: tokio::runtime::Handle::try_current().ok(),
        }
    }

    fn abort_task(&mut self) -> Option<JoinHandle<()>> {
        let resource = self.resource.take()?;
        Some(self.runtime.as_ref()?.spawn((self.abort)(resource)))
    }

    pub(super) async fn finish<R, E>(mut self, result: Result<R, E>) -> Result<R, E> {
        if result.is_ok() {
            self.resource.take();
        } else if let Some(abort) = self.abort_task() {
            let _ = abort.await;
        }
        result
    }
}

impl<T: Send + 'static> Drop for PendingUpload<T> {
    fn drop(&mut self) {
        self.abort_task();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_outside_runtime_still_aborts_upload() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (sent, aborted) = tokio::sync::oneshot::channel();
        let pending = runtime.block_on(async {
            PendingUpload::new(sent, |sent| Box::pin(async move { sent.send(()).unwrap() }))
        });
        drop(pending);
        runtime.block_on(aborted).unwrap();
    }

    #[test]
    fn multipart_update_fits_large_objects_within_s3_part_limit() {
        let options = PutMode::Update(object_store::UpdateVersion {
            e_tag: Some("\"original\"".into()),
            version: None,
        })
        .into();
        let initial = crate::object_writer::initial_upload_size();
        assert!(
            MultipartUpdate::for_payload(&options, initial)
                .unwrap()
                .is_none()
        );
        for bytes in [6_usize << 30, 5_usize << 40] {
            let update = MultipartUpdate::for_payload(&options, bytes)
                .unwrap()
                .unwrap();
            assert!(bytes.div_ceil(update.part_size) <= 10_000);
            assert!((initial..=(5_usize << 30)).contains(&update.part_size));
        }
        assert!(
            MultipartUpdate::for_payload(&PutMode::Create.into(), 6_usize << 30)
                .unwrap()
                .is_none()
        );
    }
}
