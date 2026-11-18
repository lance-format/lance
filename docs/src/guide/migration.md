# Migration Guides

Lance aims to avoid breaking changes when possible. The Python API is stable, and breaking changes are generally
communicated via deprecation warnings for 1–2 months before removal. The Rust public API is still evolving, and any
breaking change across the Rust crates (`lance*`), Python package (`pylance`), or Java SDK (`org.lance:lance-core`)
increments the shared major version number.

### Release and Format Versioning

- **Shared package versions (`1.0.0` and later)**: Following `0.39`, Lance adopted strict [semantic versioning](../community/release.md#release-versioning) starting at `1.0.0`. All language SDKs share the same release version. Major versions not listed on this page contained breaking changes limited to narrow or internal Rust traits that do not require migration steps for standard SDK callers (see the [GitHub Releases](https://github.com/lance-format/lance/releases) for full changelogs).
- **On-disk format versions (`2.0`, `2.1`, `2.2`, `2.3`)**: Package release versions are independent of the dataset `data_storage_version`. Upgrading the SDK does not rewrite existing tables or force a file format upgrade. See [Choosing a data file version](read_and_write.md#choosing-a-data-file-version) for format compatibility details.

## Unreleased: Rust Storage API

The Rust storage API is now owned by Lance. This is a source-level breaking
change with no compatibility aliases; it does not change the on-disk format.

| Previous API | Replacement |
| --- | --- |
| `lance_io::object_store` | `lance_io::storage` |
| `ObjectStore` | `Storage` |
| `ObjectStoreParams` | `StorageParams` |
| `ObjectStoreProvider` | `StorageProvider` |
| `ObjectStoreRegistry` | `StorageRegistry` |
| `WrappingObjectStore` | `BackendWrapper` |
| `object_store_wrapper` / chained wrappers | `StorageParams::backend_wrappers`, in application order |

### Opening and extending storage

Open a configured handle with `Storage::from_uri` or
`Storage::from_uri_and_params`. `Storage::local()` and `Storage::memory()` remain
available for built-in stores. The old direct-backend constructors, backend
injection through params, and deprecated dataset `with_object_store` entry
points have been removed.

Custom integrations implement `StorageProvider::new_backend`, return a
`BackendConfig`, and register their provider in a `StorageRegistry`. Supply that
registry through the dataset session, or pass it to
`Storage::from_uri_and_params`. The registry owns handle construction and installs
tracing, metrics, caller wrappers, and per-binding IO tracking. Providers should
return an uninstrumented backend, retaining their authorization scope, endpoint,
credential refresh, and cache identity.

`Storage::inner` is private. Replace direct calls with `Storage` methods for
reads, writes, metadata, listing, deletion, and conditional publication. Use
`BackendWrapper` for middleware; migrate its implementation to Lance's
`StorageBackend`. Use the getters for `store_prefix`,
`use_constant_size_upload_parts`, and `list_is_lexically_ordered`; providers set
these policies through their configuration instead of mutating a live handle.

### Backend operations and value types

Import `Path`, request options, metadata, payloads, and storage errors from
`lance_io::storage`, rather than the third-party `object_store` crate.
`StorageBackend` requires eleven operations, including `list_paginated`.
Multipart implementations use Lance's `MultipartUpload`. Atomic conditional
operations must remain atomic or return `NotSupported`.

The separate `PaginatedBackend` and wrapper `wrap_paginated` method are removed.
Implement pagination on the backend so it passes through the same policies and
instrumentation as other IO. Return `NotSupported` if pagination cannot be
provided; `Storage` falls back to the wrapped ordinary listing. Other errors
propagate. Unsupported native operations are normalized to `NotSupported`; do
not match the removed `NotImplemented` variant.

Adapt an existing third-party backend once, at the provider boundary, with
`storage::backend::object_store::from_object_store` (or
`from_object_store_with_pagination`). Convert native errors there with
`from_error`. There is no reverse adapter and no implicit conversion between
`object_store::Error` and `lance_core::Error`.

### Options, HTTP clients, and local access

Use `StorageOptionsAccessor` with static options or a `StorageOptionsProvider`
for renewable options. Credential providers must use distinct IDs for distinct
authorization scopes. `StorageParams` no longer accepts third-party credential
or HTTP client types. Use Lance's `storage::transport::HttpClientWrapper` for HTTP
customization. The public `StorageOptions::client_options()` accessor is removed;
pass timeout, proxy, and TLS settings as storage options instead.

`BackendConfig` declares local directory semantics, remote IO, and single-request
copy limits. A local scheme or `is_local` declaration does not grant host
filesystem access. Native access remains restricted to the built-in file provider.
Backend wrappers intercept object IO, including pagination. They do not disable
provider-owned native directory maintenance: canonicalization, hidden staging-file
cleanup, recursive directory deletion, and empty-directory cleanup retain the
provider's local root mapping. Installing a wrapper still disables direct local
file reads and writes, so those operations go through the wrapped backend.
A backend wrapper alone does not define a local IO sandbox; custom integrations
must establish their storage scope through the provider.

See [Object Store Configuration](object_store.md) and the Rust API examples for
provider registration and storage options.

## 11.0.0

* `document_granularity="list_element"` requires v3 reader capability,
  independently of the posting format.

## 9.0.0

* Unless overridden, newly created FTS indexes use format v2. The code analyzer
  and `block_size=256` require format v3, so readers must support v3 before an
  index using either option is created.

* To keep new indexes readable by nodes that support at most format v1 or v2,
  set `format_version` in the index creation parameters, or set
  `LANCE_FTS_FORMAT_VERSION` for a rollout-wide override. Formats v1 and v2
  require the text analyzer and `block_size=128`. A code index cannot use
  format v1 or v2: when `LANCE_FTS_FORMAT_VERSION` is set to `1` or `2`,
  creating a code index fails unless `format_version=3` is passed explicitly,
  because an explicit `format_version` takes precedence over the environment
  variable.

* Operations that maintain an existing FTS index, including append, incremental
  indexing, optimize, and mem-wal maintained-index flush, preserve its format
  version.

## 7.2.0

* The `IndexSegmentBuilder` API has been removed from Rust, Python, and Java.
  This API was deprecated by the distributed indexing flow based on
  `create_index_uncommitted`, `merge_existing_index_segments`, and
  `commit_existing_index_segments`, but remained in the codebase as a parallel
  way to plan and publish staged index segments.

* Callers should now publish staged segment outputs directly with
  `commit_existing_index_segments(...)`. If multiple staged outputs should be
  combined into a larger physical segment first, callers should explicitly group
  those outputs and call `merge_existing_index_segments(...)` for each group
  before committing the final segment list.

* The old builder's `target_segment_bytes` automatic size-based grouping has no
  direct replacement. Distributed index drivers that used it should choose
  segment groups themselves, then pass each group to
  `merge_existing_index_segments(...)`.

## 5.0.0

* The default data storage version changed from 2.0 to 2.1. This affects the `column_indices`
  field in the `DataFile` protobuf message. In 2.0, every field (including non-leaf fields like
  struct containers and list containers) was assigned a sequential column index. In 2.1, non-leaf
  fields (unpacked structs, list containers) are assigned `-1` instead since their validity
  information is now folded into repetition/definition levels. Only leaf fields and packed structs
  are assigned column indices.

    For example, given the schema:

    ```
    x: i32, y: [f32], z: { a: i32 }
    ```

    The fields (in depth-first order) are:

    | Field ID | Field         |
    |----------|---------------|
    | 0        | `x` (i32)     |
    | 1        | `y` (list)    |
    | 2        | `y.item` (f32)|
    | 3        | `z` (struct)  |
    | 4        | `z.a` (i32)   |

    In **2.0**, `column_indices` = `[0, 1, 2, 3, 4]` — every field gets a column.

    In **2.1**, `column_indices` = `[0, -1, 1, -1, 2]` — non-leaf fields (`y` and `z`) get `-1`.

* This change only affects advanced users who construct `DataFile` messages directly, for example
  when building operations by hand for `Dataset.commit`. Normal read and write paths are
  unaffected.

* To opt back to 2.0 format, set `data_storage_version="2.0"` when creating a dataset.

## 1.0.0

* The `SearchResult` returned by scalar indices must now output information about null values.
  Instead of containing a `RowIdTreeMap`, it now contains a `NullableRowIdSet`. Expressions that
  resolve to null values must be included in search results in the null set. This ensures that
  `NOT` can be applied to index search results correctly.

## 0.39

* The `lance` crate no longer re-exports utilities from `lance-arrow` such as `RecordBatchExt` or `SchemaExt`.  In the
short term, if you are relying on these utilities,  you can add a dependency on the `lance-arrow` crate.  However, we
do not expect `lance-arrow` to ever be stable, and you may want to consider forking these utilities.

* Previously, we exported `Error` and `Result` as both `lance::Error` and `lance::error::Error`.  We have now reduced
this to just `lance::Error`.  We have also removed some internal error utilities (such as `OptionExt`) from the public
API and do not plan on reintroducing these.

* The Python and Rust `dataset::diff_meta` API has been removed in favor of `dataset::delta`, 
which returns a `DatasetDelta` that offers both metadata diff through `list_transactions` and data diff 
through `get_inserted_rows` and `get_updated_rows`.

* Some other minor utilities which had previously been public are now private.  It is unlikely anyone was utilizing
these.  Please open an issue if you were relying on any of these.

* The `lance-namespace` Rust crate now splits into `lance-namespace` that contains the main `LanceNamespace` trait 
and data models, and `lance-namespace-impls` that has different implementations of the namespace. 
The `DirectoryNamespace` and `RestNamespace` interfaces have been refactored to be more user friendly.
The `DirectoryNamespace` also now uses Lance ObjectStore for IO instead of directly depending on Apache OpenDAL.
