// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::sync::Arc;

use arrow_schema::ArrowError;
use snafu::{IntoError as _, Location, Snafu};

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[cfg(feature = "backtrace")]
mod backtrace_support {
    use std::backtrace::Backtrace;

    use snafu::{AsBacktrace, GenerateImplicitData};

    #[derive(Debug)]
    pub struct MaybeBacktrace(pub Option<Backtrace>);

    impl GenerateImplicitData for MaybeBacktrace {
        fn generate() -> Self {
            Self(<Option<Backtrace>>::generate())
        }
    }

    impl AsBacktrace for MaybeBacktrace {
        fn as_backtrace(&self) -> Option<&Backtrace> {
            self.0.as_ref()
        }
    }
}

#[cfg(not(feature = "backtrace"))]
mod backtrace_support {
    use std::backtrace::Backtrace;

    use snafu::{AsBacktrace, GenerateImplicitData};

    #[derive(Debug)]
    pub struct MaybeBacktrace;

    impl GenerateImplicitData for MaybeBacktrace {
        fn generate() -> Self {
            Self
        }
    }

    impl AsBacktrace for MaybeBacktrace {
        fn as_backtrace(&self) -> Option<&Backtrace> {
            None
        }
    }
}

use backtrace_support::MaybeBacktrace;

/// Error for when a requested field is not found in a schema.
///
/// This error computes suggestions lazily (only when displayed) to avoid
/// computing Levenshtein distance when the error is created but never shown.
#[derive(Debug)]
pub struct FieldNotFoundError {
    pub field_name: String,
    pub candidates: Vec<String>,
}

impl fmt::Display for FieldNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Field '{}' not found.", self.field_name)?;
        let suggestion =
            crate::levenshtein::find_best_suggestion(&self.field_name, &self.candidates);
        if let Some(suggestion) = suggestion {
            write!(f, " Did you mean '{}'?", suggestion)?;
        }
        write!(f, "\nAvailable fields: [")?;
        for (i, candidate) in self.candidates.iter().take(10).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "'{}'", candidate)?;
        }
        if self.candidates.len() > 10 {
            let remaining = self.candidates.len() - 10;
            write!(f, ", ... and {} more]", remaining)?;
        } else {
            write!(f, "]")?;
        }
        Ok(())
    }
}

impl std::error::Error for FieldNotFoundError {}

/// A manifest commit returned an error and its final outcome could not be
/// determined safely.
///
/// This is wrapped in [`Error::Wrapped`] so Lance can expose a structured
/// source without adding a variant to the exhaustive public [`Error`] enum.
#[derive(Debug)]
pub struct CommitStatusUnknownError {
    version: u64,
    source: BoxedError,
}

impl CommitStatusUnknownError {
    /// Return the manifest version whose commit outcome is unknown.
    pub fn version(&self) -> u64 {
        self.version
    }
}

impl std::fmt::Display for CommitStatusUnknownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Commit result for version {} is unknown: the commit may or may not have been \
             applied; check the table state before retrying: {}",
            self.version, self.source
        )
    }
}

impl std::error::Error for CommitStatusUnknownError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Allocates error on the heap and then places `e` into it.
#[inline]
pub fn box_error(e: impl std::error::Error + Send + Sync + 'static) -> BoxedError {
    Box::new(e)
}

/// Why a writer is fenced. Both reasons are terminal, but callers must tell them
/// apart (a peer takeover vs. our own failure) rather than parse the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FenceReason {
    /// A successor writer claimed a higher epoch; this writer lost ownership.
    PeerClaimedEpoch,
    /// Our own WAL persistence failed, so in-memory state may have diverged from
    /// the durable WAL. The writer must be reopened to replay.
    PersistenceFailure,
}

impl std::fmt::Display for FenceReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Stable strings — surfaced in error messages.
        let s = match self {
            Self::PeerClaimedEpoch => "peer claimed epoch",
            Self::PersistenceFailure => "persistence failure",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("Invalid user input: {source}, {location}"))]
    InvalidInput {
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Dataset already exists: {uri}, {location}"))]
    DatasetAlreadyExists {
        uri: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Append with different schema: {difference}, location: {location}"))]
    SchemaMismatch {
        difference: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Dataset at path {path} was not found: {source}, {location}"))]
    DatasetNotFound {
        path: String,
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Encountered corrupt file {path}: {source}, {location}"))]
    CorruptFile {
        path: object_store::path::Path,
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Not supported: {source}, {location}"))]
    NotSupported {
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Commit conflict for version {version}: {source}, {location}"))]
    CommitConflict {
        version: u64,
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Incompatible transaction: {source}, {location}"))]
    IncompatibleTransaction {
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Retryable commit conflict for version {version}: {source}, {location}"))]
    RetryableCommitConflict {
        version: u64,
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Too many concurrent writers. {message}, {location}"))]
    TooMuchWriteContention {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Operation timed out: {message}, {location}"))]
    Timeout {
        message: String,
        #[snafu(implicit)]
        location: Location,
    },
    #[snafu(display(
        "Encountered internal error. Please file a bug report at https://github.com/lance-format/lance/issues. {message}, {location}"
    ))]
    Internal {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("A prerequisite task failed: {message}, {location}"))]
    PrerequisiteFailed {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Unprocessable: {message}, {location}"))]
    Unprocessable {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("LanceError(Arrow): {message}, {location}"))]
    Arrow {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("LanceError(Schema): {message}, {location}"))]
    Schema {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Not found: {uri}, {location}"))]
    NotFound {
        uri: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("LanceError(IO): {source}, {location}"))]
    IO {
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("LanceError(Index): {message}, {location}"))]
    Index {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Lance index not found: {identity}, {location}"))]
    IndexNotFound {
        identity: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Cannot infer storage location from: {message}"))]
    InvalidTableLocation { message: String },
    /// Stream early stop
    Stop,
    #[snafu(display("Wrapped error: {error}, {location}"))]
    Wrapped {
        #[snafu(source)]
        error: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Cloned error: {message}, {location}"))]
    Cloned {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Query Execution error: {message}, {location}"))]
    Execution {
        message: String,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Ref is invalid: {message}"))]
    InvalidRef { message: String },
    #[snafu(display("Ref conflict error: {message}"))]
    RefConflict { message: String },
    #[snafu(display("Ref not found error: {message}"))]
    RefNotFound { message: String },
    #[snafu(display("Cleanup error: {message}"))]
    Cleanup { message: String },
    #[snafu(display("Version not found error: {message}"))]
    VersionNotFound { message: String },
    #[snafu(display("Version conflict error: {message}"))]
    VersionConflict {
        message: String,
        major_version: u16,
        minor_version: u16,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    #[snafu(display("Namespace error: {source}, {location}"))]
    Namespace {
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
        #[snafu(implicit)]
        backtrace: MaybeBacktrace,
    },
    /// External error passed through from user code.
    ///
    /// This variant preserves errors that users pass into Lance APIs (e.g., via streams
    /// with custom error types). The original error can be recovered using [`Error::into_external`]
    /// or inspected using [`Error::external_source`].
    #[snafu(transparent)]
    External { source: BoxedError },

    /// A requested field was not found in a schema.
    #[snafu(transparent)]
    FieldNotFound { source: FieldNotFoundError },

    #[snafu(display(
        "Spill disk cap of {cap_bytes} bytes exceeded; currently using {used_bytes} bytes, {location}"
    ))]
    DiskCapExceeded {
        cap_bytes: u64,
        used_bytes: u64,
        #[snafu(implicit)]
        location: Location,
    },
    /// A writer has been fenced and must stop (see [`FenceReason`]). The message
    /// keeps the `Writer fenced` prefix for legacy string consumers; new code
    /// should match on [`Error::fence_reason`].
    #[snafu(display("Writer fenced ({reason}): {message}, {location}"))]
    Fenced {
        reason: FenceReason,
        message: String,
        #[snafu(implicit)]
        location: Location,
    },
    /// A write was refused to keep the writer inside its memory budget.
    ///
    /// Unlike every other write error this one is *expected* under load and
    /// carries no data loss: the write was never accepted, so a caller that
    /// retries once the flush pipeline drains loses nothing. Callers should
    /// surface it as a retryable "busy" signal (HTTP 503), not a failure.
    /// Match via [`Error::is_backpressure`] rather than on the message.
    #[snafu(display("Write rejected by backpressure: {message}, {location}"))]
    Backpressure {
        message: String,
        #[snafu(implicit)]
        location: Location,
    },
    /// An error shared behind an [`Arc`] so it can be cloned cheaply without
    /// deep-copying its source (see [`CloneableError`]). Accessors that read a
    /// variant or typed source rather than the message — [`Error::external_source`],
    /// [`Error::fence_reason`], [`Error::is_backpressure`], [`Error::is_not_found`],
    /// [`Error::is_commit_status_unknown`], [`Error::backtrace`] — see through this
    /// wrapper, so a shared error answers them exactly as the error it wraps.
    #[snafu(transparent)]
    Shared { source: Arc<Self> },
}

impl Error {
    /// Returns the captured Rust backtrace, if available.
    ///
    /// Requires the `backtrace` feature to be enabled at compile time
    /// and `RUST_BACKTRACE=1` at runtime.
    #[cfg(feature = "backtrace")]
    pub fn backtrace(&self) -> Option<&std::backtrace::Backtrace> {
        match self {
            Self::InvalidInput { backtrace, .. }
            | Self::DatasetAlreadyExists { backtrace, .. }
            | Self::SchemaMismatch { backtrace, .. }
            | Self::DatasetNotFound { backtrace, .. }
            | Self::CorruptFile { backtrace, .. }
            | Self::NotSupported { backtrace, .. }
            | Self::CommitConflict { backtrace, .. }
            | Self::IncompatibleTransaction { backtrace, .. }
            | Self::RetryableCommitConflict { backtrace, .. }
            | Self::TooMuchWriteContention { backtrace, .. }
            | Self::Internal { backtrace, .. }
            | Self::PrerequisiteFailed { backtrace, .. }
            | Self::Unprocessable { backtrace, .. }
            | Self::Arrow { backtrace, .. }
            | Self::Schema { backtrace, .. }
            | Self::NotFound { backtrace, .. }
            | Self::IO { backtrace, .. }
            | Self::Index { backtrace, .. }
            | Self::IndexNotFound { backtrace, .. }
            | Self::Wrapped { backtrace, .. }
            | Self::Cloned { backtrace, .. }
            | Self::Execution { backtrace, .. }
            | Self::VersionConflict { backtrace, .. }
            | Self::Namespace { backtrace, .. } => {
                use snafu::AsBacktrace;
                backtrace.as_backtrace()
            }
            // Variants without a backtrace field — listed explicitly so that
            // adding a new variant with a backtrace field triggers a compiler error.
            Self::InvalidTableLocation { .. }
            | Self::Stop
            | Self::InvalidRef { .. }
            | Self::RefConflict { .. }
            | Self::RefNotFound { .. }
            | Self::Cleanup { .. }
            | Self::VersionNotFound { .. }
            | Self::External { .. }
            | Self::FieldNotFound { .. }
            | Self::Timeout { .. }
            | Self::DiskCapExceeded { .. }
            | Self::Fenced { .. }
            | Self::Backpressure { .. } => None,
            // A shared error delegates to the error it wraps.
            Self::Shared { source } => source.backtrace(),
        }
    }

    /// Returns the captured Rust backtrace, if available.
    ///
    /// Always returns `None` when the `backtrace` feature is not enabled.
    #[cfg(not(feature = "backtrace"))]
    pub fn backtrace(&self) -> Option<&std::backtrace::Backtrace> {
        None
    }

    #[track_caller]
    pub fn corrupt_file(path: object_store::path::Path, message: impl Into<String>) -> Self {
        CorruptFileSnafu { path }.into_error(message.into().into())
    }

    /// Reports a corrupt file when the caller only has a logical/section name
    /// rather than the real file path (for example, a decoder that validates an
    /// in-memory buffer and does not know where it came from).
    ///
    /// `name` is carried in the `path` field of the resulting [`Error::CorruptFile`]
    /// variant and is NOT a filesystem path; callers that have the real path should
    /// use [`Self::corrupt_file`] instead.
    #[track_caller]
    pub fn corrupt_file_named(name: &str, message: impl Into<String>) -> Self {
        Self::corrupt_file(object_store::path::Path::from(name), message)
    }

    #[track_caller]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        InvalidInputSnafu.into_error(message.into().into())
    }

    #[track_caller]
    pub fn invalid_input_source(source: BoxedError) -> Self {
        InvalidInputSnafu.into_error(source)
    }

    #[track_caller]
    pub fn io(message: impl Into<String>) -> Self {
        IOSnafu.into_error(message.into().into())
    }

    /// A successor writer claimed a higher epoch; this writer lost ownership.
    #[track_caller]
    pub fn fenced_by_peer(message: impl Into<String>) -> Self {
        FencedSnafu {
            reason: FenceReason::PeerClaimedEpoch,
            message: message.into(),
        }
        .build()
    }

    /// Our WAL persistence failed; in-memory state may have diverged from the
    /// durable WAL, so the writer must be reopened to replay.
    #[track_caller]
    pub fn writer_poisoned(message: impl Into<String>) -> Self {
        FencedSnafu {
            reason: FenceReason::PersistenceFailure,
            message: message.into(),
        }
        .build()
    }

    /// Peel every [`Error::Shared`] wrapper and return the underlying error.
    ///
    /// Accessors route through this, so a shared error answers the same as the
    /// error it wraps. Callers that need to match a concrete variant without an
    /// accessor (such as [`Error::Timeout`]) should match on `find_root()`, so
    /// the match still holds after the error has been shared through a
    /// [`CloneableError`].
    pub fn find_root(&self) -> &Self {
        let mut current = self;
        while let Self::Shared { source } = current {
            current = source.as_ref();
        }
        current
    }

    /// The [`FenceReason`] if this is [`Error::Fenced`], else `None`. Prefer this
    /// over matching the error message to decide how to react to a fence.
    pub fn fence_reason(&self) -> Option<FenceReason> {
        match self.find_root() {
            Self::Fenced { reason, .. } => Some(*reason),
            _ => None,
        }
    }

    /// A write was refused because the writer is at its memory ceiling; the
    /// data was never accepted. See [`Error::Backpressure`].
    #[track_caller]
    pub fn backpressure(message: impl Into<String>) -> Self {
        BackpressureSnafu {
            message: message.into(),
        }
        .build()
    }

    /// Whether this is [`Error::Backpressure`] — i.e. a retryable "writer is
    /// full" signal rather than a real failure. Prefer this over matching the
    /// error message.
    pub fn is_backpressure(&self) -> bool {
        matches!(self.find_root(), Self::Backpressure { .. })
    }

    #[track_caller]
    pub fn io_source(source: BoxedError) -> Self {
        IOSnafu.into_error(source)
    }

    #[track_caller]
    pub fn dataset_already_exists(uri: impl Into<String>) -> Self {
        DatasetAlreadyExistsSnafu { uri: uri.into() }.build()
    }

    #[track_caller]
    pub fn dataset_not_found(path: impl Into<String>, source: BoxedError) -> Self {
        DatasetNotFoundSnafu { path: path.into() }.into_error(source)
    }

    #[track_caller]
    pub fn version_conflict(
        message: impl Into<String>,
        major_version: u16,
        minor_version: u16,
    ) -> Self {
        VersionConflictSnafu {
            message: message.into(),
            major_version,
            minor_version,
        }
        .build()
    }

    #[track_caller]
    pub fn not_found(uri: impl Into<String>) -> Self {
        NotFoundSnafu { uri: uri.into() }.build()
    }

    /// Return whether this error or one of its typed sources is a missing object.
    pub fn is_not_found(&self) -> bool {
        match self.find_root() {
            Self::NotFound { .. } => true,
            Self::Wrapped { error, .. }
                if error.downcast_ref::<CommitStatusUnknownError>().is_some() =>
            {
                false
            }
            Self::IO { source, .. } | Self::Wrapped { error: source, .. } => {
                error_source_is_not_found(source.as_ref())
            }
            _ => false,
        }
    }

    #[track_caller]
    pub fn wrapped(error: BoxedError) -> Self {
        WrappedSnafu.into_error(error)
    }

    #[track_caller]
    pub fn schema(message: impl Into<String>) -> Self {
        SchemaSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn not_supported(message: impl Into<String>) -> Self {
        NotSupportedSnafu.into_error(message.into().into())
    }

    #[track_caller]
    pub fn not_supported_source(source: BoxedError) -> Self {
        NotSupportedSnafu.into_error(source)
    }

    #[track_caller]
    pub fn internal(message: impl Into<String>) -> Self {
        InternalSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn timeout(message: impl Into<String>) -> Self {
        TimeoutSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn namespace(message: impl Into<String>) -> Self {
        NamespaceSnafu.into_error(message.into().into())
    }

    #[track_caller]
    pub fn namespace_source(source: Box<dyn std::error::Error + Send + Sync + 'static>) -> Self {
        NamespaceSnafu.into_error(source)
    }

    #[track_caller]
    pub fn arrow(message: impl Into<String>) -> Self {
        ArrowSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn execution(message: impl Into<String>) -> Self {
        ExecutionSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn cloned(message: impl Into<String>) -> Self {
        ClonedSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn schema_mismatch(difference: impl Into<String>) -> Self {
        SchemaMismatchSnafu {
            difference: difference.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn unprocessable(message: impl Into<String>) -> Self {
        UnprocessableSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn too_much_write_contention(message: impl Into<String>) -> Self {
        TooMuchWriteContentionSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn prerequisite_failed(message: impl Into<String>) -> Self {
        PrerequisiteFailedSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn index(message: impl Into<String>) -> Self {
        IndexSnafu {
            message: message.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn index_not_found(identity: impl Into<String>) -> Self {
        IndexNotFoundSnafu {
            identity: identity.into(),
        }
        .build()
    }

    #[track_caller]
    pub fn commit_conflict_source(version: u64, source: BoxedError) -> Self {
        CommitConflictSnafu { version }.into_error(source)
    }

    #[track_caller]
    pub fn retryable_commit_conflict_source(version: u64, source: BoxedError) -> Self {
        RetryableCommitConflictSnafu { version }.into_error(source)
    }

    #[track_caller]
    pub fn commit_status_unknown_source(version: u64, source: BoxedError) -> Self {
        Self::wrapped(box_error(CommitStatusUnknownError { version, source }))
    }

    /// Return whether this error represents a commit whose final outcome could
    /// not be determined safely.
    pub fn is_commit_status_unknown(&self) -> bool {
        matches!(
            self.find_root(),
            Self::Wrapped { error, .. }
                if error.downcast_ref::<CommitStatusUnknownError>().is_some()
        )
    }

    #[track_caller]
    pub fn incompatible_transaction_source(source: BoxedError) -> Self {
        IncompatibleTransactionSnafu.into_error(source)
    }

    #[track_caller]
    pub fn disk_cap_exceeded(cap_bytes: u64, used_bytes: u64) -> Self {
        DiskCapExceededSnafu {
            cap_bytes,
            used_bytes,
        }
        .build()
    }

    /// Create an External error from a boxed error source.
    pub fn external(source: BoxedError) -> Self {
        Self::External { source }
    }

    /// Create a FieldNotFound error with the given field name and available candidates.
    pub fn field_not_found(field_name: impl Into<String>, candidates: Vec<String>) -> Self {
        Self::FieldNotFound {
            source: FieldNotFoundError {
                field_name: field_name.into(),
                candidates,
            },
        }
    }

    /// Returns a reference to the external error source if this is an `External`
    /// variant, seeing through any `Shared` wrapper.
    ///
    /// This allows downcasting to recover the original error type.
    pub fn external_source(&self) -> Option<&BoxedError> {
        match self.find_root() {
            Self::External { source } => Some(source),
            _ => None,
        }
    }

    /// Consumes the error and returns the external source if this is an `External`
    /// variant, seeing through a sole-owner `Shared` wrapper.
    ///
    /// Otherwise returns `Err` with an equivalent error for chained handling: a
    /// sole-owner `Shared` is unwrapped to its inner error, a still-shared one
    /// stays wrapped, and any other error is returned as-is.
    pub fn into_external(self) -> std::result::Result<BoxedError, Self> {
        match self {
            Self::External { source } => Ok(source),
            // A sole-owner Shared can surrender its inner error; a still-shared
            // one cannot, so it stays wrapped (`external_source` reads it by ref).
            Self::Shared { source } => match Arc::try_unwrap(source) {
                Ok(inner) => inner.into_external(),
                Err(shared) => Err(Self::Shared { source: shared }),
            },
            other => Err(other),
        }
    }
}

fn error_source_is_not_found(source: &(dyn std::error::Error + 'static)) -> bool {
    if let Some(error) = source.downcast_ref::<Error>() {
        return error.is_not_found();
    }
    if let Some(error) = source.downcast_ref::<object_store::Error>() {
        return matches!(error, object_store::Error::NotFound { .. })
            || std::error::Error::source(error).is_some_and(error_source_is_not_found);
    }
    source.source().is_some_and(error_source_is_not_found)
}

pub trait LanceOptionExt<T> {
    /// Unwraps an option, returning an internal error if the option is None.
    ///
    /// Can be used when an option is expected to have a value.
    fn expect_ok(self) -> Result<T>;
}

impl<T> LanceOptionExt<T> for Option<T> {
    #[track_caller]
    fn expect_ok(self) -> Result<T> {
        self.ok_or_else(|| Error::internal("Expected option to have value"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
pub type ArrowResult<T> = std::result::Result<T, ArrowError>;
#[cfg(feature = "datafusion")]
pub type DataFusionResult<T> = std::result::Result<T, datafusion_common::DataFusionError>;

impl From<ArrowError> for Error {
    #[track_caller]
    fn from(e: ArrowError) -> Self {
        match e {
            ArrowError::ExternalError(source) => {
                // Try to downcast to lance_core::Error first to recover the original
                match source.downcast::<Self>() {
                    Ok(lance_err) => *lance_err,
                    Err(source) => Self::External { source },
                }
            }
            other => Self::arrow(other.to_string()),
        }
    }
}

impl From<&ArrowError> for Error {
    #[track_caller]
    fn from(e: &ArrowError) -> Self {
        Self::arrow(e.to_string())
    }
}

impl From<std::io::Error> for Error {
    #[track_caller]
    fn from(e: std::io::Error) -> Self {
        // A lance `Error` may have been wrapped in an `io::Error` (e.g. via
        // `io::Error::other(Error::...)`) to cross an `AsyncWrite`/`AsyncRead`
        // boundary. Recover it so typed errors such as `DiskCapExceeded`
        // survive the round-trip instead of collapsing into an opaque `IO`.
        if e.get_ref().is_some_and(|inner| inner.is::<Self>()) {
            return *e
                .into_inner()
                .expect("checked Some above")
                .downcast::<Self>()
                .expect("checked type above");
        }
        Self::io_source(box_error(e))
    }
}

impl From<object_store::Error> for Error {
    #[track_caller]
    fn from(e: object_store::Error) -> Self {
        match e {
            // source intentionally dropped; Error::NotFound carries only the path
            object_store::Error::NotFound { path, .. } => Self::not_found(path),
            other => Self::io_source(box_error(other)),
        }
    }
}

impl From<prost::DecodeError> for Error {
    #[track_caller]
    fn from(e: prost::DecodeError) -> Self {
        Self::io_source(box_error(e))
    }
}

impl From<prost::EncodeError> for Error {
    #[track_caller]
    fn from(e: prost::EncodeError) -> Self {
        Self::io_source(box_error(e))
    }
}

impl From<prost::UnknownEnumValue> for Error {
    #[track_caller]
    fn from(e: prost::UnknownEnumValue) -> Self {
        Self::io_source(box_error(e))
    }
}

impl From<tokio::task::JoinError> for Error {
    #[track_caller]
    fn from(e: tokio::task::JoinError) -> Self {
        Self::io_source(box_error(e))
    }
}

impl From<object_store::path::Error> for Error {
    #[track_caller]
    fn from(e: object_store::path::Error) -> Self {
        Self::io_source(box_error(e))
    }
}

impl From<url::ParseError> for Error {
    #[track_caller]
    fn from(e: url::ParseError) -> Self {
        Self::io_source(box_error(e))
    }
}

impl From<serde_json::Error> for Error {
    #[track_caller]
    fn from(e: serde_json::Error) -> Self {
        Self::arrow(e.to_string())
    }
}

impl From<Error> for ArrowError {
    fn from(value: Error) -> Self {
        match value {
            // Pass through external errors directly
            Error::External { source } => Self::ExternalError(source),
            // Preserve schema errors with their specific type
            Error::Schema { message, .. } => Self::SchemaError(message),
            // Wrap all other lance errors so they can be recovered
            e => Self::ExternalError(Box::new(e)),
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::parser::ParserError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::parser::ParserError) -> Self {
        Self::io_source(box_error(e))
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::tokenizer::TokenizerError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::tokenizer::TokenizerError) -> Self {
        Self::io_source(box_error(e))
    }
}

#[cfg(feature = "datafusion")]
impl From<Error> for datafusion_common::DataFusionError {
    #[track_caller]
    fn from(e: Error) -> Self {
        Self::External(Box::new(e))
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_common::DataFusionError> for Error {
    #[track_caller]
    fn from(e: datafusion_common::DataFusionError) -> Self {
        match e {
            // DataFusion wraps an error to attach end-user context and source
            // spans (`Diagnostic`), a description of what was running
            // (`Context`), or to report several failures at once
            // (`Collection`). All three are display-transparent, so the
            // category has to come from the error underneath; classifying the
            // wrapper itself reports a malformed query as an internal failure.
            datafusion_common::DataFusionError::Diagnostic(_, inner)
            | datafusion_common::DataFusionError::Context(_, inner) => Self::from(*inner),
            datafusion_common::DataFusionError::Collection(errors) => {
                match errors.into_iter().next() {
                    // `Collection` reports the first error's message, so take
                    // its category too.
                    Some(first) => Self::from(first),
                    None => Self::execution("DataFusion returned an empty error collection"),
                }
            }
            datafusion_common::DataFusionError::SQL(..)
            | datafusion_common::DataFusionError::Plan(..)
            | datafusion_common::DataFusionError::Configuration(..)
            | datafusion_common::DataFusionError::SchemaError(..) => {
                Self::invalid_input_source(box_error(e))
            }
            datafusion_common::DataFusionError::ArrowError(arrow_err, _) => Self::from(*arrow_err),
            datafusion_common::DataFusionError::NotImplemented(..) => {
                Self::not_supported_source(box_error(e))
            }
            datafusion_common::DataFusionError::Execution(..) => Self::execution(e.to_string()),
            datafusion_common::DataFusionError::Shared(shared) => {
                // DataFusion shares an error across consumers (e.g. a join's
                // build-side error fanned out to every probe partition) behind an
                // `Arc`. If we are the sole owner we can recurse for full fidelity;
                // otherwise re-wrap in `Shared` so the concrete error type is still
                // reachable via `Error::source` / `downcast_ref`.
                match std::sync::Arc::try_unwrap(shared) {
                    Ok(inner) => Self::from(inner),
                    Err(shared) => {
                        let rewrapped = datafusion_common::DataFusionError::Shared(shared);
                        Self::External {
                            source: box_error(rewrapped),
                        }
                    }
                }
            }
            datafusion_common::DataFusionError::External(source) => {
                // Try to downcast to lance_core::Error first
                match source.downcast::<Self>() {
                    Ok(lance_err) => *lance_err,
                    Err(source) => Self::External { source },
                }
            }
            _ => Self::io_source(box_error(e)),
        }
    }
}

// This is a bit odd but some object_store functions only accept
// Stream<Result<T, ObjectStoreError>> and so we need to convert
// to ObjectStoreError to call the methods.
impl From<Error> for object_store::Error {
    fn from(err: Error) -> Self {
        Self::Generic {
            store: "N/A",
            source: Box::new(err),
        }
    }
}

#[track_caller]
pub fn get_caller_location() -> &'static std::panic::Location<'static> {
    std::panic::Location::caller()
}

/// An [`Error`] that can be cloned, for sharing one fallible source across
/// several consumers (for example a shared future two streams both await).
///
/// The error is held behind an [`Arc`], so a clone shares it rather than
/// rebuilding it. Every accessor — `is_not_found`, `fence_reason`,
/// `is_backpressure`, `is_commit_status_unknown`, `external_source` — therefore
/// answers the same on every clone, including `external_source`, whose downcast
/// a message-based rebuild could not preserve. Use [`CloneableError::inner`] to
/// borrow the error and [`CloneableError::into_inner`] to recover it by value.
pub struct CloneableError(Arc<Error>);

impl CloneableError {
    /// Wrap an [`Error`] so it can be shared and cloned.
    ///
    /// An already-shared error reuses its inner `Arc` rather than nesting
    /// another [`Error::Shared`], so repeatedly wrapping and recovering the
    /// same error keeps the wrapper flat instead of growing a `Shared` chain.
    pub fn new(error: Error) -> Self {
        Self(match error {
            Error::Shared { source } => source,
            other => Arc::new(other),
        })
    }

    /// Borrow the shared error.
    pub fn inner(&self) -> &Error {
        &self.0
    }

    /// Recover the error by value. When other clones are still alive the error
    /// cannot be moved out, so it is returned wrapped in [`Error::Shared`],
    /// which answers every accessor the same as the error it wraps.
    pub fn into_inner(self) -> Error {
        Arc::try_unwrap(self.0).unwrap_or_else(|shared| Error::Shared { source: shared })
    }
}

impl Clone for CloneableError {
    fn clone(&self) -> Self {
        // Cheap and lossless: every clone shares one `Arc<Error>`, so accessors
        // (`external_source`, `fence_reason`, ...) answer identically on every
        // side. Recovering an owned error from a still-shared clone yields
        // `Error::Shared`; see `into_inner`.
        Self(Arc::clone(&self.0))
    }
}

#[derive(Clone)]
pub struct CloneableResult<T: Clone>(pub std::result::Result<T, CloneableError>);

impl<T: Clone> From<Result<T>> for CloneableResult<T> {
    fn from(result: Result<T>) -> Self {
        Self(result.map_err(CloneableError::new))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fmt;

    /// The variants whose meaning lives in an accessor rather than in the
    /// message: collapsing them into a cloned string drops the signal.
    #[test]
    fn cloneable_error_preserves_accessor_contracts() {
        // Both reasons, because carrying `reason` through is the arm's whole job.
        for (original, expected) in [
            (
                Error::fenced_by_peer("peer claimed epoch 7"),
                FenceReason::PeerClaimedEpoch,
            ),
            (
                Error::writer_poisoned("wal flush failed"),
                FenceReason::PersistenceFailure,
            ),
        ] {
            let original = CloneableError::new(original);
            assert_eq!(original.inner().fence_reason(), Some(expected));
            assert_eq!(original.clone().inner().fence_reason(), Some(expected));
        }

        let original = CloneableError::new(Error::backpressure("writer at memory ceiling"));
        let cloned = original.clone();
        assert!(original.inner().is_backpressure());
        assert!(cloned.inner().is_backpressure());

        let original = CloneableError::new(Error::commit_status_unknown_source(
            7,
            box_error(Error::timeout("put timed out")),
        ));
        let cloned = original.clone();
        // Cloning the clone must not degrade it further.
        let twice = cloned.clone();
        assert!(original.inner().is_commit_status_unknown());
        assert!(cloned.inner().is_commit_status_unknown());
        assert!(!cloned.inner().is_not_found());
        assert!(twice.inner().is_commit_status_unknown());
    }

    /// The signal #9541 could not fix by rebuilding: an external source must
    /// still downcast to its original type on the far side of a clone.
    #[test]
    fn cloneable_error_preserves_external_source_across_clone() {
        let original = CloneableError::new(Error::external(Box::new(MyCustomError {
            code: 7,
            message: "external across clone".into(),
        })));
        let cloned = original.clone();
        // Both the original and the clone must still downcast to the source type.
        for err in [&original, &cloned] {
            let source = err
                .inner()
                .external_source()
                .expect("external source must survive a clone");
            assert_eq!(source.downcast_ref::<MyCustomError>().unwrap().code, 7);
        }
    }

    /// `new` must reuse an already-shared error's Arc instead of nesting another
    /// `Shared` layer, so repeated share/recover cycles cannot grow the depth
    /// (and with it the depth-proportional recursion in `into_external`,
    /// `Display`, `backtrace`, and `Drop`).
    #[test]
    fn cloneable_error_new_flattens_shared() {
        let shared = Error::Shared {
            source: Arc::new(Error::io("disk gone")),
        };
        let wrapped = CloneableError::new(shared);
        assert!(matches!(wrapped.inner(), Error::IO { .. }));
        assert!(!matches!(wrapped.inner(), Error::Shared { .. }));
    }

    /// A sole owner recovers the concrete inner error, not a `Shared` wrapper —
    /// the common consumer path (e.g. moka's `unwrap_or_clone(..).into_inner()`).
    #[test]
    fn into_inner_sole_owner_returns_concrete_variant() {
        let recovered = CloneableError::new(Error::io("solo")).into_inner();
        assert!(matches!(recovered, Error::IO { .. }));
    }

    /// A still-shared error cannot surrender ownership of its source, so
    /// `into_external` returns it re-wrapped rather than the boxed source.
    #[test]
    fn into_external_keeps_multi_owner_shared_wrapped() {
        let arc = Arc::new(Error::external(Box::new(MyCustomError {
            code: 5,
            message: "still shared".into(),
        })));
        let _second_owner = arc.clone(); // refcount 2 → try_unwrap must fail
        match (Error::Shared { source: arc }).into_external() {
            Err(Error::Shared { .. }) => {}
            other => panic!("multi-owner Shared must stay wrapped, got {other:?}"),
        }
    }

    /// `backtrace()` delegates through the `Shared` wrapper to the inner error.
    #[test]
    fn backtrace_sees_through_shared() {
        let shared = Error::Shared {
            source: Arc::new(Error::io("io")),
        };
        assert_eq!(
            shared.backtrace().is_some(),
            Error::io("io").backtrace().is_some()
        );
    }

    /// Every accessor that reads a variant or typed source rather than the
    /// message must see through an `Error::Shared` wrapper, including nested
    /// ones, so a shared error answers exactly as the error it wraps.
    #[test]
    fn accessors_see_through_shared_wrapper() {
        let external = Error::Shared {
            source: Arc::new(Error::external(Box::new(MyCustomError {
                code: 7,
                message: "wrapped".into(),
            }))),
        };
        let source = external
            .external_source()
            .expect("external_source must see through Shared");
        assert_eq!(source.downcast_ref::<MyCustomError>().unwrap().code, 7);

        let fenced = Error::Shared {
            source: Arc::new(Error::writer_poisoned("wal flush failed")),
        };
        assert_eq!(fenced.fence_reason(), Some(FenceReason::PersistenceFailure));

        let backpressure = Error::Shared {
            source: Arc::new(Error::backpressure("at ceiling")),
        };
        assert!(backpressure.is_backpressure());

        let commit_unknown = Error::Shared {
            source: Arc::new(Error::commit_status_unknown_source(
                9,
                box_error(Error::timeout("put timed out")),
            )),
        };
        assert!(commit_unknown.is_commit_status_unknown());

        // find_root peels every layer, so nested wrappers still answer.
        let nested = Error::Shared {
            source: Arc::new(Error::Shared {
                source: Arc::new(Error::not_found("deep")),
            }),
        };
        assert!(nested.is_not_found());

        // Sole-owner Shared can still surrender ownership of the external source.
        let owned = Error::Shared {
            source: Arc::new(Error::external(Box::new(MyCustomError {
                code: 11,
                message: "owned".into(),
            }))),
        };
        match owned.into_external() {
            Ok(source) => assert_eq!(source.downcast::<MyCustomError>().unwrap().code, 11),
            Err(e) => panic!("into_external should unwrap a sole-owner Shared, got {e:?}"),
        }
    }

    #[test]
    fn cloneable_error_preserves_not_found_contract() {
        let original = CloneableError::new(Error::not_found("metadata.lance"));
        let cloned = original.clone();
        let cloned_again = cloned.clone();
        // Sharing means every clone is still the real NotFound, not a rebuild.
        assert!(matches!(original.inner(), Error::NotFound { .. }));
        assert!(matches!(cloned.inner(), Error::NotFound { .. }));
        assert!(matches!(cloned_again.inner(), Error::NotFound { .. }));
        assert!(
            cloned
                .inner()
                .to_string()
                .to_lowercase()
                .contains("not found")
        );

        // Taken by value while other clones live, a clone becomes `Error::Shared`;
        // is_not_found must still propagate through a downstream error chain.
        let downstream_error = Error::wrapped(Box::new(Error::io_source(Box::new(
            object_store::Error::Generic {
                store: "N/A",
                source: Box::new(cloned.into_inner()),
            },
        ))));
        assert!(downstream_error.is_not_found());

        let original = CloneableError::new(Error::timeout("metadata read timed out"));
        let cloned = original.clone();
        assert!(matches!(original.inner(), Error::Timeout { .. }));
        assert!(matches!(cloned.inner(), Error::Timeout { .. }));

        let original = CloneableError::new(Error::io("metadata read was denied"));
        let cloned = original.clone();
        assert!(matches!(original.inner(), Error::IO { .. }));
        assert!(matches!(cloned.inner(), Error::IO { .. }));
    }

    #[test]
    fn test_caller_location_capture() {
        let current_fn = get_caller_location();
        // make sure ? captures the correct location
        // .into() WILL NOT capture the correct location
        let f: Box<dyn Fn() -> Result<()>> = Box::new(|| {
            Err(object_store::Error::Generic {
                store: "",
                source: "".into(),
            })?;
            Ok(())
        });
        match f().unwrap_err() {
            Error::IO { location, .. } => {
                // +4 is the beginning of object_store::Error::Generic...
                assert_eq!(location.line(), current_fn.line() + 4, "{}", location)
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected ObjectStore error"),
        }
    }

    #[test]
    fn test_caller_location_capture_not_found() {
        let current_fn = get_caller_location();
        let f: Box<dyn Fn() -> Result<()>> = Box::new(|| {
            Err(object_store::Error::NotFound {
                path: "some/path".to_string(),
                source: "not found".into(),
            })?;
            Ok(())
        });
        match f().unwrap_err() {
            Error::NotFound { location, .. } => {
                // +2 is the beginning of object_store::Error::NotFound...
                assert_eq!(location.line(), current_fn.line() + 2, "{}", location)
            }
            #[allow(unreachable_patterns)]
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_object_store_not_found_converts_to_not_found() {
        let os_err = object_store::Error::NotFound {
            path: "test/path".to_string(),
            source: "no such file".into(),
        };
        let lance_err: Error = os_err.into();
        match lance_err {
            Error::NotFound { uri, .. } => {
                assert_eq!(uri, "test/path");
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[derive(Debug)]
    struct MyCustomError {
        code: i32,
        message: String,
    }

    impl fmt::Display for MyCustomError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "MyCustomError({}): {}", self.code, self.message)
        }
    }

    impl std::error::Error for MyCustomError {}

    #[test]
    fn test_io_error_recovers_wrapped_lance_error() {
        // A lance Error wrapped in io::Error::other should round-trip back to
        // the original variant rather than collapsing into Error::IO.
        let io_err = std::io::Error::other(Error::disk_cap_exceeded(100, 50));
        let recovered: Error = io_err.into();
        match recovered {
            Error::DiskCapExceeded {
                cap_bytes,
                used_bytes,
                ..
            } => {
                assert_eq!(cap_bytes, 100);
                assert_eq!(used_bytes, 50);
            }
            other => panic!("expected DiskCapExceeded, got {other:?}"),
        }
    }

    #[test]
    fn test_io_error_without_lance_error_stays_io() {
        // A plain io::Error (no wrapped lance Error) should become Error::IO.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let converted: Error = io_err.into();
        assert!(matches!(converted, Error::IO { .. }));
    }

    #[test]
    fn test_commit_status_unknown_is_structured_without_masking_as_not_found() {
        let error = Error::commit_status_unknown_source(
            42,
            box_error(Error::not_found("temporarily invisible manifest")),
        );

        assert!(error.is_commit_status_unknown());
        assert!(!error.is_not_found());
        assert!(error.to_string().contains("version 42 is unknown"));
        let Error::Wrapped { error, .. } = error else {
            panic!("commit-status-unknown must use the semver-compatible wrapper")
        };
        let status = error
            .downcast_ref::<CommitStatusUnknownError>()
            .expect("wrapper must retain the typed commit status");
        assert_eq!(status.version(), 42);
    }

    #[test]
    fn test_external_error_creation() {
        let custom_err = MyCustomError {
            code: 42,
            message: "test error".to_string(),
        };
        let err = Error::external(Box::new(custom_err));

        match &err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 42);
                assert_eq!(recovered.message, "test error");
            }
            _ => panic!("Expected External variant"),
        }
    }

    #[test]
    fn test_external_source_method() {
        let custom_err = MyCustomError {
            code: 123,
            message: "source test".to_string(),
        };
        let err = Error::external(Box::new(custom_err));

        let source = err.external_source().expect("should have external source");
        let recovered = source.downcast_ref::<MyCustomError>().unwrap();
        assert_eq!(recovered.code, 123);

        // Test that non-External variants return None
        let io_err = Error::io("test");
        assert!(io_err.external_source().is_none());
    }

    #[test]
    fn test_into_external_method() {
        let custom_err = MyCustomError {
            code: 456,
            message: "into test".to_string(),
        };
        let err = Error::external(Box::new(custom_err));

        match err.into_external() {
            Ok(source) => {
                let recovered = source.downcast::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 456);
            }
            Err(_) => panic!("Expected Ok"),
        }

        // Test that non-External variants return Err(self)
        let io_err = Error::io("test");
        match io_err.into_external() {
            Err(Error::IO { .. }) => {}
            _ => panic!("Expected Err with IO variant"),
        }
    }

    #[test]
    fn test_arrow_external_error_conversion() {
        let custom_err = MyCustomError {
            code: 789,
            message: "arrow test".to_string(),
        };
        let arrow_err = ArrowError::ExternalError(Box::new(custom_err));
        let lance_err: Error = arrow_err.into();

        match lance_err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 789);
            }
            _ => panic!("Expected External variant, got {:?}", lance_err),
        }
    }

    #[test]
    fn test_external_to_arrow_roundtrip() {
        let custom_err = MyCustomError {
            code: 999,
            message: "roundtrip".to_string(),
        };
        let lance_err = Error::external(Box::new(custom_err));
        let arrow_err: ArrowError = lance_err.into();

        match arrow_err {
            ArrowError::ExternalError(source) => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 999);
            }
            _ => panic!("Expected ExternalError variant"),
        }
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_schema_error_is_invalid_input() {
        // Schema errors from DataFusion (e.g., a filter referencing an unknown
        // column) are user-input failures, not internal lance failures. They
        // must surface as `Error::InvalidInput` so downstream FFI/Python
        // bindings can map them to the right user-facing error code.
        use datafusion_common::Column;

        let schema_err = datafusion_common::SchemaError::FieldNotFound {
            field: Box::new(Column::from_name("missing_col")),
            valid_fields: vec![],
        };
        let df_err =
            datafusion_common::DataFusionError::SchemaError(Box::new(schema_err), Box::new(None));
        let lance_err: Error = df_err.into();

        match lance_err {
            Error::InvalidInput { .. } => {
                assert!(
                    lance_err.to_string().contains("missing_col"),
                    "expected the column name to survive in the error message, got: {lance_err}"
                );
            }
            _ => panic!("Expected InvalidInput variant, got {:?}", lance_err),
        }
    }

    /// DataFusion wraps errors to attach end-user context (`Diagnostic`), a
    /// description of what was running (`Context`), or to report several at
    /// once (`Collection`). All three are display-transparent, so a wrapped
    /// user error looks exactly like an unwrapped one but would be classified
    /// as an internal failure if the conversion matched on the wrapper.
    #[cfg(feature = "datafusion")]
    #[rstest::rstest]
    #[case::diagnostic(|inner| datafusion_common::DataFusionError::Diagnostic(
        Box::new(datafusion_common::Diagnostic::new_error("invalid function", None)),
        Box::new(inner),
    ))]
    #[case::context(|inner| datafusion_common::DataFusionError::Context(
        "type_coercion".to_string(),
        Box::new(inner),
    ))]
    #[case::collection(|inner| datafusion_common::DataFusionError::Collection(vec![inner]))]
    #[case::nested(|inner| datafusion_common::DataFusionError::Diagnostic(
        Box::new(datafusion_common::Diagnostic::new_error("invalid function", None)),
        Box::new(datafusion_common::DataFusionError::Context(
            "type_coercion".to_string(),
            Box::new(inner),
        )),
    ))]
    fn test_datafusion_wrapped_plan_error_is_invalid_input(
        #[case] wrap: fn(datafusion_common::DataFusionError) -> datafusion_common::DataFusionError,
    ) {
        let df_err = wrap(datafusion_common::DataFusionError::Plan(
            "Invalid function 'no_such_function'".to_string(),
        ));
        let lance_err = Error::from(df_err);

        assert!(
            matches!(lance_err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {lance_err:?}"
        );
        assert!(
            lance_err.to_string().contains("no_such_function"),
            "expected the function name to survive, got: {lance_err}"
        );
    }

    /// Unwrapping must classify by the inner error rather than assume the
    /// wrapper always hides a user error.
    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_wrapped_internal_error_is_not_invalid_input() {
        let df_err = datafusion_common::DataFusionError::Context(
            "while running".to_string(),
            Box::new(datafusion_common::DataFusionError::Internal(
                "invariant violated".to_string(),
            )),
        );

        assert!(
            matches!(Error::from(df_err), Error::IO { .. }),
            "an internal DataFusion failure must not be reported as user input"
        );
    }

    /// A Lance error that round-trips through DataFusion keeps its own
    /// category even when DataFusion wraps it on the way back.
    #[cfg(feature = "datafusion")]
    #[test]
    fn test_wrapped_external_lance_error_keeps_its_category() {
        let df_err = datafusion_common::DataFusionError::Context(
            "while scanning".to_string(),
            Box::new(datafusion_common::DataFusionError::from(Error::io(
                "object store unavailable",
            ))),
        );

        match Error::from(df_err) {
            Error::IO { source, .. } => assert!(
                source.to_string().contains("object store unavailable"),
                "expected the original message, got: {source}"
            ),
            other => panic!("expected the original IO error, got {other:?}"),
        }
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_external_error_conversion() {
        let custom_err = MyCustomError {
            code: 111,
            message: "datafusion test".to_string(),
        };
        let df_err = datafusion_common::DataFusionError::External(Box::new(custom_err));
        let lance_err: Error = df_err.into();

        match lance_err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 111);
            }
            _ => panic!("Expected External variant"),
        }
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_arrow_external_error_conversion() {
        // Test the nested case: ArrowError::ExternalError inside DataFusionError::ArrowError
        let custom_err = MyCustomError {
            code: 222,
            message: "nested test".to_string(),
        };
        let arrow_err = ArrowError::ExternalError(Box::new(custom_err));
        let df_err = datafusion_common::DataFusionError::ArrowError(Box::new(arrow_err), None);
        let lance_err: Error = df_err.into();

        match lance_err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 222);
            }
            _ => panic!("Expected External variant, got {:?}", lance_err),
        }
    }

    /// Test that lance_core::Error round-trips through ArrowError.
    ///
    /// This simulates the case where a user defines an iterator in terms of
    /// lance_core::Error, and the error goes through Arrow's error type
    /// (e.g., via RecordBatchIterator) before being converted back.
    #[test]
    fn test_lance_error_roundtrip_through_arrow() {
        let original = Error::invalid_input("test validation error");

        // Simulate what happens when using ? in an Arrow context
        let arrow_err: ArrowError = original.into();

        // Convert back to lance error (as happens when Lance consumes the stream)
        let recovered: Error = arrow_err.into();

        // Should get back the original lance error directly (not wrapped in External)
        match recovered {
            Error::InvalidInput { .. } => {
                assert!(recovered.to_string().contains("test validation error"));
            }
            _ => panic!("Expected InvalidInput variant, got {:?}", recovered),
        }
    }

    /// Test that lance_core::Error round-trips through DataFusionError.
    ///
    /// This simulates the case where a user defines a stream in terms of
    /// lance_core::Error, and the error goes through DataFusion's error type
    /// (e.g., via SendableRecordBatchStream) before being converted back.
    #[cfg(feature = "datafusion")]
    #[test]
    fn test_lance_error_roundtrip_through_datafusion() {
        let original = Error::invalid_input("test validation error");

        // Simulate what happens when using ? in a DataFusion context
        let df_err: datafusion_common::DataFusionError = original.into();

        // Convert back to lance error (as happens when Lance consumes the stream)
        let recovered: Error = df_err.into();

        // Should get back the original lance error directly (not wrapped in External)
        match recovered {
            Error::InvalidInput { .. } => {
                assert!(recovered.to_string().contains("test validation error"));
            }
            _ => panic!("Expected InvalidInput variant, got {:?}", recovered),
        }
    }

    /// Test that a typed error survives a multiply-owned `DataFusionError::Shared`.
    ///
    /// When DataFusion fans one error out to multiple consumers via `Arc`, we
    /// cannot move the inner error out.  The typed source must still be
    /// reachable after conversion to `lance_core::Error`.
    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_shared_multi_owner_preserves_type() {
        let custom_err = MyCustomError {
            code: 42,
            message: "shared typed error".to_string(),
        };
        let marker = datafusion_common::DataFusionError::External(Box::new(custom_err));
        // Put it in an Arc and keep a second owner so try_unwrap fails.
        let arc = std::sync::Arc::new(marker);
        let _arc2 = arc.clone();
        let shared = datafusion_common::DataFusionError::Shared(arc);

        let lance_err: Error = shared.into();

        // The concrete error must be discoverable via source chain.
        let mut found = false;
        let mut src: Option<&dyn std::error::Error> = Some(&lance_err);
        while let Some(e) = src {
            if e.downcast_ref::<MyCustomError>().is_some() {
                found = true;
                break;
            }
            src = e.source();
        }
        assert!(
            found,
            "MyCustomError not found in source chain: {lance_err:?}"
        );
    }

    #[test]
    fn test_backtrace_accessor() {
        // Verify that backtrace() returns the expected result based on feature state
        let err = Error::io("test backtrace");
        let bt = err.backtrace();
        #[cfg(feature = "backtrace")]
        {
            // With the backtrace feature enabled, whether a backtrace is captured
            // depends on the RUST_BACKTRACE env var at runtime. We just verify
            // the accessor doesn't panic and returns a valid Option.
            let _ = bt;
        }
        #[cfg(not(feature = "backtrace"))]
        {
            // Without the backtrace feature, this must always be None.
            assert!(bt.is_none());
        }
    }

    #[test]
    fn test_backtrace_captured_when_feature_enabled() {
        // Test that backtrace is actually captured when the feature is on and
        // RUST_BACKTRACE=1 is set in the environment before the process starts.
        //
        // NOTE: std::backtrace::Backtrace caches the RUST_BACKTRACE env check,
        // so set_var at runtime does not reliably enable capture. This test
        // verifies the accessor works correctly in both cases:
        // - If RUST_BACKTRACE=1 was set before the test binary started, we get Some.
        // - If not, we get None (even with the feature on), which is expected.
        #[cfg(feature = "backtrace")]
        {
            let err = Error::io("backtrace capture test");
            if std::env::var("RUST_BACKTRACE").is_ok() {
                assert!(
                    err.backtrace().is_some(),
                    "Expected a backtrace when RUST_BACKTRACE=1 and backtrace feature is enabled"
                );
            }
            // When RUST_BACKTRACE is not set, backtrace() may return None even
            // with the feature enabled — this is correct runtime gating behavior.
        }
        #[cfg(not(feature = "backtrace"))]
        {
            let err = Error::io("backtrace capture test");
            assert!(err.backtrace().is_none());
        }
    }

    #[test]
    fn test_backtrace_returns_none_for_variants_without_location() {
        let err = Error::InvalidTableLocation {
            message: "test".to_string(),
        };
        assert!(err.backtrace().is_none());

        let err = Error::InvalidRef {
            message: "test".to_string(),
        };
        assert!(err.backtrace().is_none());

        let err = Error::Stop;
        assert!(err.backtrace().is_none());
    }
}
