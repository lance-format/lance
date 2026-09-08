// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::str::Utf8Error;

use arrow_schema::ArrowError;
use jni::{JNIEnv, errors::Error as JniError};
use lance::Error as LanceError;
use lance_namespace::error::NamespaceError;
use serde_json::Error as JsonError;

#[derive(Debug, PartialEq, Eq)]
pub enum JavaExceptionClass {
    IllegalArgumentException,
    IOException,
    RuntimeException,
    UnsupportedOperationException,
    AlreadyInException,
    LanceNamespaceException,
    LanceTimeoutException,
}

impl JavaExceptionClass {
    pub fn as_str(&self) -> &str {
        match self {
            Self::IllegalArgumentException => "java/lang/IllegalArgumentException",
            Self::IOException => "java/io/IOException",
            Self::RuntimeException => "java/lang/RuntimeException",
            Self::UnsupportedOperationException => "java/lang/UnsupportedOperationException",
            // Included for display purposes.  This is not a real exception.
            Self::AlreadyInException => "AlreadyInException",
            Self::LanceNamespaceException => "org/lance/namespace/errors/LanceNamespaceException",
            Self::LanceTimeoutException => "org/lance/LanceTimeoutException",
        }
    }
}

#[derive(Debug)]
pub struct Error {
    message: String,
    java_class: JavaExceptionClass,
    namespace_error_code: Option<u32>,
}

impl Error {
    pub fn new(message: String, java_class: JavaExceptionClass) -> Self {
        Self {
            message,
            java_class,
            namespace_error_code: None,
        }
    }

    pub fn runtime_error(message: String) -> Self {
        Self {
            message,
            java_class: JavaExceptionClass::RuntimeException,
            namespace_error_code: None,
        }
    }

    pub fn io_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::IOException)
    }

    pub fn input_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::IllegalArgumentException)
    }

    pub fn unsupported_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::UnsupportedOperationException)
    }

    pub fn timeout_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::LanceTimeoutException)
    }

    pub fn namespace_error(code: u32, message: String) -> Self {
        Self {
            message,
            java_class: JavaExceptionClass::LanceNamespaceException,
            namespace_error_code: Some(code),
        }
    }

    pub fn in_exception() -> Self {
        Self {
            message: String::default(),
            java_class: JavaExceptionClass::AlreadyInException,
            namespace_error_code: None,
        }
    }

    pub fn throw(&self, env: &mut JNIEnv) {
        if self.java_class == JavaExceptionClass::AlreadyInException {
            // An exception is already in progress, so we don't need to throw another one.
            return;
        }

        // For namespace errors, throw the specific LanceNamespaceException
        if self.java_class == JavaExceptionClass::LanceNamespaceException
            && let Some(code) = self.namespace_error_code
        {
            // Call LanceNamespaceException.fromCode static method
            if self.throw_namespace_exception(env, code).is_err() {
                // lance-namespace is bundled as a dependency, so the exception classes
                // should always be available. Panic if they're not.
                panic!(
                    "Failed to throw LanceNamespaceException (code={}). \
                        org.lance.namespace.errors.LanceNamespaceException and ErrorCode classes \
                        must be available in the classpath.",
                    code
                );
            }
            return;
        }

        if let Err(e) = env.throw_new(self.java_class.as_str(), &self.message) {
            eprintln!("Error when throwing Java exception: {:?}", e.to_string());
            panic!("Error when throwing Java exception: {:?}", e);
        }
    }

    fn throw_namespace_exception(
        &self,
        env: &mut JNIEnv,
        code: u32,
    ) -> std::result::Result<(), ()> {
        // Use ErrorFactory.fromErrorCode(code, message) to get the specific exception subclass
        // (e.g., TableNotFoundException, NamespaceNotFoundException, etc.)
        let factory_class = "org/lance/namespace/errors/ErrorFactory";

        let factory_cls = env.find_class(factory_class).map_err(|_| ())?;
        let from_error_code_method = env
            .get_static_method_id(
                &factory_cls,
                "fromErrorCode",
                "(ILjava/lang/String;)Lorg/lance/namespace/errors/LanceNamespaceException;",
            )
            .map_err(|_| ())?;

        let message_str = env.new_string(&self.message).map_err(|_| ())?;

        let exception_obj = unsafe {
            env.call_static_method_unchecked(
                &factory_cls,
                from_error_code_method,
                jni::signature::ReturnType::Object,
                &[
                    jni::sys::jvalue {
                        i: code as jni::sys::jint,
                    },
                    jni::sys::jvalue {
                        l: message_str.as_raw(),
                    },
                ],
            )
        }
        .map_err(|_| ())?;

        let exception = match exception_obj {
            jni::objects::JValueGen::Object(obj) => obj,
            _ => return Err(()),
        };

        // Throw the exception
        env.throw(jni::objects::JThrowable::from(exception))
            .map_err(|_| ())?;

        Ok(())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.java_class.as_str(), self.message)
    }
}

impl From<LanceError> for Error {
    fn from(err: LanceError) -> Self {
        let backtrace_suffix = err
            .backtrace()
            .map(|bt| format!("\n\nRust backtrace:\n{}", bt))
            .unwrap_or_default();
        let message = format!("{}{}", err, backtrace_suffix);

        match &err {
            LanceError::DatasetNotFound { .. }
            | LanceError::DatasetAlreadyExists { .. }
            | LanceError::CommitConflict { .. }
            | LanceError::InvalidInput { .. } => Self::input_error(message),
            LanceError::IO { .. } => Self::io_error(message),
            LanceError::Timeout { .. } => Self::timeout_error(message),
            LanceError::NotSupported { .. } => Self::unsupported_error(message),
            LanceError::NotFound { .. } => Self::io_error(message),
            LanceError::Namespace { source, .. } => {
                // Try to downcast to NamespaceError and get the error code
                if let Some(ns_err) = source.downcast_ref::<NamespaceError>() {
                    let ns_message = format!("{}{}", ns_err, backtrace_suffix);
                    Self::namespace_error(ns_err.code().as_u32(), ns_message)
                } else {
                    log::warn!(
                        "Failed to downcast NamespaceError source, falling back to runtime error. \
                         This may indicate a version mismatch. Source type: {:?}",
                        source
                    );
                    Self::runtime_error(message)
                }
            }
            _ => Self::runtime_error(message),
        }
    }
}

impl From<ArrowError> for Error {
    fn from(err: ArrowError) -> Self {
        match err {
            ArrowError::InvalidArgumentError { .. } => Self::input_error(err.to_string()),
            ArrowError::IoError { .. } => Self::io_error(err.to_string()),
            ArrowError::NotYetImplemented(_) => Self::unsupported_error(err.to_string()),
            _ => Self::runtime_error(err.to_string()),
        }
    }
}

impl From<JsonError> for Error {
    fn from(err: JsonError) -> Self {
        Self::io_error(err.to_string())
    }
}

impl From<JniError> for Error {
    fn from(err: JniError) -> Self {
        match err {
            // If we get this then it means that an exception was already in progress.  We can't
            // throw another one so we just return an error indicating that.
            JniError::JavaException => Self::in_exception(),
            _ => Self::runtime_error(err.to_string()),
        }
    }
}

impl From<Utf8Error> for Error {
    fn from(err: Utf8Error) -> Self {
        Self::input_error(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: extract the java_class from an Error via Display output
    fn java_class(err: &Error) -> &JavaExceptionClass {
        &err.java_class
    }

    #[test]
    fn test_invalid_input_maps_to_illegal_argument() {
        let lance_err = LanceError::invalid_input("bad input");
        let jni_err: Error = lance_err.into();
        assert_eq!(
            *java_class(&jni_err),
            JavaExceptionClass::IllegalArgumentException
        );
        assert!(jni_err.message.contains("bad input"));
    }

    #[test]
    fn test_dataset_not_found_maps_to_illegal_argument() {
        let lance_err = LanceError::dataset_not_found("my_dataset", "not found".to_string().into());
        let jni_err: Error = lance_err.into();
        assert_eq!(
            *java_class(&jni_err),
            JavaExceptionClass::IllegalArgumentException
        );
        assert!(jni_err.message.contains("my_dataset"));
    }

    #[test]
    fn test_dataset_already_exists_maps_to_illegal_argument() {
        let lance_err = LanceError::dataset_already_exists("my_dataset");
        let jni_err: Error = lance_err.into();
        assert_eq!(
            *java_class(&jni_err),
            JavaExceptionClass::IllegalArgumentException
        );
        assert!(jni_err.message.contains("my_dataset"));
    }

    #[test]
    fn test_commit_conflict_maps_to_illegal_argument() {
        let lance_err = LanceError::commit_conflict_source(42, "conflict".to_string().into());
        let jni_err: Error = lance_err.into();
        assert_eq!(
            *java_class(&jni_err),
            JavaExceptionClass::IllegalArgumentException
        );
    }

    #[test]
    fn test_io_maps_to_ioexception() {
        let lance_err = LanceError::io("disk failure");
        let jni_err: Error = lance_err.into();
        assert_eq!(*java_class(&jni_err), JavaExceptionClass::IOException);
        assert!(jni_err.message.contains("disk failure"));
    }

    #[test]
    fn test_not_supported_maps_to_unsupported() {
        let lance_err = LanceError::not_supported("nope");
        let jni_err: Error = lance_err.into();
        assert_eq!(
            *java_class(&jni_err),
            JavaExceptionClass::UnsupportedOperationException
        );
        assert!(jni_err.message.contains("nope"));
    }

    #[test]
    fn test_not_found_maps_to_ioexception() {
        let lance_err = LanceError::not_found("missing_uri");
        let jni_err: Error = lance_err.into();
        assert_eq!(*java_class(&jni_err), JavaExceptionClass::IOException);
        assert!(jni_err.message.contains("missing_uri"));
    }

    #[test]
    fn test_fallthrough_maps_to_runtime() {
        let lance_err = LanceError::internal("internal oops");
        let jni_err: Error = lance_err.into();
        assert_eq!(*java_class(&jni_err), JavaExceptionClass::RuntimeException);
        assert!(jni_err.message.contains("internal oops"));
    }

    #[test]
    fn test_no_backtrace_suffix_when_backtrace_is_none() {
        // Without the backtrace feature enabled in lance-core default tests,
        // backtrace() returns None, so no suffix should be appended.
        let lance_err = LanceError::io("clean message");
        let jni_err: Error = lance_err.into();
        assert!(
            !jni_err.message.contains("Rust backtrace:"),
            "Expected no backtrace suffix, got: {}",
            jni_err.message
        );
    }
}
