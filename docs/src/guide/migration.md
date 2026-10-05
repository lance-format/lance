# Migration Guides

Lance aims to avoid breaking changes when possible.  Currently, we are refining the Rust public API so that we can move
it out of experimental status and make stronger commitments to backwards compatibility.  The python API is considered
stable and breaking changes should generally be communicated (via warnings) for 1-2 months prior to being finalized to
give users a chance to migrate.  This page documents the breaking changes between releases and gives advice on how to
migrate.

## 9.0.0

* Unless overridden, newly created FTS indexes use format v2. The code analyzer
  and `block_size=256` require format v3, so readers must support v3 before an
  index using either option is created. `document_granularity="list_element"`
  also requires v3 reader capability, independently of the posting format.

* To keep new indexes readable by nodes that support at most format v1 or v2,
  set `format_version` in the index creation parameters, or set
  `LANCE_FTS_FORMAT_VERSION` for a rollout-wide override. Formats v1 and v2
  require the text analyzer and `block_size=128`.

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

* Some other minor utilities which had previously been public are now private.  It is unlikely anyone was utilizing'
these.  Please open an issue if you were relying on any of these.

* The `lance-namespace` Rust crate now splits into `lance-namespace` that contains the main `LanceNamespace` trait 
and data models, and `lance-namespace-impls` that has different implementations of the namespace. 
The `DirectoryNamespace` and `RestNamespace` interfaces have been refactored to be more user friendly.
The `DirectoryNamespace` also now uses Lance ObjectStore for IO instead of directly depending on Apache OpenDAL.
