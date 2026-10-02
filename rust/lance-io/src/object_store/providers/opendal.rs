// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;

use opendal::Operator;
use opendal::layers::RetryLayer;

pub(in crate::object_store) fn finish_opendal_operator(
    operator: Operator,
    max_retries: usize,
) -> Operator {
    if max_retries == 0 {
        return operator;
    }

    let retry_layer = RetryLayer::new()
        .with_max_times(max_retries)
        .with_min_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(15))
        .with_factor(2.0)
        .with_jitter();

    operator.layer(retry_layer)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use opendal::raw::oio;
    use opendal::raw::{
        OpCopy, OpCreateDir, OpList, OpPresign, OpRead, OpRename, OpStat, OpWrite, RpCreateDir,
        RpPresign, RpRename, RpStat, Service, ServiceInfo,
    };
    use opendal::{
        Buffer, Capability, Error, ErrorKind, Metadata, MetadataBuilder, OperationContext, Result,
    };
    use rstest::rstest;

    use super::*;

    #[derive(Debug, Clone)]
    struct RetryTestService {
        stat_attempts: Arc<AtomicUsize>,
        stat_failures: usize,
        is_stat_error_temporary: bool,
        write_attempts: Arc<AtomicUsize>,
        write_failures: usize,
        write_payloads: Arc<Mutex<Vec<Bytes>>>,
        close_attempts: Arc<AtomicUsize>,
        close_failures: usize,
    }

    impl RetryTestService {
        fn new(stat_failures: usize, is_stat_error_temporary: bool) -> Self {
            Self {
                stat_attempts: Arc::new(AtomicUsize::new(0)),
                stat_failures,
                is_stat_error_temporary,
                write_attempts: Arc::new(AtomicUsize::new(0)),
                write_failures: 0,
                write_payloads: Arc::new(Mutex::new(Vec::new())),
                close_attempts: Arc::new(AtomicUsize::new(0)),
                close_failures: 0,
            }
        }

        fn with_write_failures(mut self, write_failures: usize) -> Self {
            self.write_failures = write_failures;
            self
        }

        fn with_close_failures(mut self, close_failures: usize) -> Self {
            self.close_failures = close_failures;
            self
        }

        fn stat_attempts(&self) -> usize {
            self.stat_attempts.load(Ordering::Relaxed)
        }

        fn write_attempts(&self) -> usize {
            self.write_attempts.load(Ordering::Relaxed)
        }

        fn write_payloads(&self) -> Vec<Bytes> {
            self.write_payloads.lock().unwrap().clone()
        }

        fn close_attempts(&self) -> usize {
            self.close_attempts.load(Ordering::Relaxed)
        }
    }

    impl Service for RetryTestService {
        type Reader = ();
        type Writer = RetryTestWriter;
        type Lister = ();
        type Deleter = ();
        type Copier = ();
        type Composer = ();

        fn info(&self) -> ServiceInfo {
            ServiceInfo::with_scheme("retry_test")
        }

        fn capability(&self) -> Capability {
            Capability {
                stat: true,
                write: true,
                write_can_multi: true,
                ..Default::default()
            }
        }

        async fn create_dir(
            &self,
            _: &OperationContext,
            _: &str,
            _: OpCreateDir,
        ) -> Result<RpCreateDir> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        async fn stat(&self, _: &OperationContext, _: &str, _: OpStat) -> Result<RpStat> {
            let attempt = self.stat_attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt <= self.stat_failures {
                let error = Error::new(ErrorKind::Unexpected, "injected stat failure");
                return if self.is_stat_error_temporary {
                    Err(error.set_temporary())
                } else {
                    Err(error)
                };
            }

            Ok(RpStat::new(MetadataBuilder::file(0).build()))
        }

        fn read(&self, _: &OperationContext, _: &str, _: OpRead) -> Result<Self::Reader> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        fn write(&self, _: &OperationContext, _: &str, _: OpWrite) -> Result<Self::Writer> {
            Ok(RetryTestWriter {
                write_attempts: self.write_attempts.clone(),
                write_failures: self.write_failures,
                write_payloads: self.write_payloads.clone(),
                close_attempts: self.close_attempts.clone(),
                close_failures: self.close_failures,
                content_length: 0,
            })
        }

        fn delete(&self, _: &OperationContext) -> Result<Self::Deleter> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        fn list(&self, _: &OperationContext, _: &str, _: OpList) -> Result<Self::Lister> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        fn copy(&self, _: &OperationContext, _: &str, _: &str, _: OpCopy) -> Result<Self::Copier> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        async fn rename(
            &self,
            _: &OperationContext,
            _: &str,
            _: &str,
            _: OpRename,
        ) -> Result<RpRename> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        async fn presign(&self, _: &OperationContext, _: &str, _: OpPresign) -> Result<RpPresign> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }
    }

    fn build_test_operator(service: RetryTestService, max_retries: usize) -> Operator {
        let operator = Operator::from_parts(OperationContext::default(), Arc::new(service));
        finish_opendal_operator(operator, max_retries)
    }

    #[derive(Debug)]
    struct RetryTestWriter {
        write_attempts: Arc<AtomicUsize>,
        write_failures: usize,
        write_payloads: Arc<Mutex<Vec<Bytes>>>,
        close_attempts: Arc<AtomicUsize>,
        close_failures: usize,
        content_length: usize,
    }

    impl oio::Write for RetryTestWriter {
        async fn write(&mut self, buffer: Buffer) -> Result<()> {
            let attempt = self.write_attempts.fetch_add(1, Ordering::Relaxed) + 1;
            self.write_payloads.lock().unwrap().push(buffer.to_bytes());
            if attempt <= self.write_failures {
                return Err(
                    Error::new(ErrorKind::Unexpected, "injected write failure").set_temporary()
                );
            }

            self.content_length += buffer.len();
            Ok(())
        }

        async fn close(&mut self) -> Result<Metadata> {
            let attempt = self.close_attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt <= self.close_failures {
                return Err(
                    Error::new(ErrorKind::Unexpected, "injected close failure").set_temporary()
                );
            }

            Ok(MetadataBuilder::file(self.content_length as u64).build())
        }

        async fn abort(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[rstest]
    #[case::recovers(2, true, 3, 3, None)]
    #[case::persistent(1, false, 3, 1, Some(false))]
    #[case::disabled(1, true, 0, 1, Some(true))]
    #[case::exhausted(usize::MAX, true, 3, 4, Some(false))]
    #[tokio::test(start_paused = true)]
    async fn test_stat_retry(
        #[case] failures: usize,
        #[case] is_temporary: bool,
        #[case] max_retries: usize,
        #[case] expected_attempts: usize,
        #[case] expected_temporary: Option<bool>,
    ) {
        let service = RetryTestService::new(failures, is_temporary);
        let operator = build_test_operator(service.clone(), max_retries);

        let result = operator.stat("file").await;
        assert_eq!(service.stat_attempts(), expected_attempts);
        match expected_temporary {
            None => {
                result.expect("stat should succeed");
            }
            Some(expected) => {
                let error = result.expect_err("stat should fail");
                assert_eq!(error.is_temporary(), expected);
                assert_eq!(error.is_persistent(), !expected);
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_write_retry() {
        let service = RetryTestService::new(0, true).with_write_failures(1);
        let operator = build_test_operator(service.clone(), 3);

        operator
            .write("file", "data")
            .await
            .expect("write should succeed");
        assert_eq!(service.write_attempts(), 2);
        assert_eq!(
            service.write_payloads(),
            vec![Bytes::from_static(b"data"), Bytes::from_static(b"data")]
        );
        assert_eq!(service.close_attempts(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn test_close_retry() {
        let service = RetryTestService::new(0, true).with_close_failures(1);
        let operator = build_test_operator(service.clone(), 3);

        operator
            .write("file", "data")
            .await
            .expect("write should succeed");
        assert_eq!(service.close_attempts(), 2);
    }
}
