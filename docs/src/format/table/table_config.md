# Table Config

The [`Manifest`](index.md#manifest) protobuf message has a `config` field: a
`map<string, string>` of key-value pairs stored with each version of the
table. Keys prefixed with `lance.` are reserved for the Lance library itself.
Other libraries building on Lance may define their own configuration keys, but
should use their own prefix (e.g. `lancedb.`) to avoid collisions with keys
defined by Lance.

Any library that reads or writes the Lance format is expected to understand
and honor the `lance.` keys listed below. Unrecognized keys, including
unrecognized keys with a `lance.` prefix, should be ignored by readers.

Config is read and modified through `Dataset.update_config()` (or the
equivalent binding in other languages), which applies the change as a normal
transaction. This is distinct from schema metadata (attached to the output
schema of scans) and from table metadata (arbitrary metadata associated with
the table); see the [`Manifest` protobuf message](index.md#manifest) for the
full set of metadata maps.

<style>
.table-config-table th:nth-child(1),
.table-config-table td:nth-child(1) {
  white-space: nowrap;
}
</style>

<div class="table-config-table" markdown="1">

| Key                                            | Type                                                                     | Description                                                                                                                                                                                                                                                                             |
|-------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `lance.auto_cleanup.interval`                   | Non-negative integer                                                     | Number of commits between automatic cleanup runs. A cleanup runs on the commit whose version is a multiple of this interval. If unset, or `0`, automatic cleanup is disabled. See [Automatic Cleanup](../../guide/read_and_write.md#automatic-cleanup).                                |
| `lance.auto_cleanup.older_than`                 | Duration string (e.g. `3600s`, `2h`)                                     | Only versions older than this duration are removed by automatic cleanup. See [Automatic Cleanup](../../guide/read_and_write.md#automatic-cleanup).                                                                                                                                       |
| `lance.auto_cleanup.retain_versions`             | Non-negative integer                                                     | Minimum number of most-recent versions to retain, regardless of age, when automatic cleanup runs.                                                                                                                                                                                        |
| `lance.auto_cleanup.referenced_branch`          | Boolean (`true`/`false`)                                                 | Whether automatic cleanup also removes eligible old versions that are only reachable through a [branch or tag](branch_tag.md) reference. Defaults to `false`.                                                                                                                            |
| `lance.auto_cleanup.delete_rate_limit`          | Non-negative integer                                                     | Maximum number of delete operations per second automatic cleanup is allowed to issue against the underlying object store.                                                                                                                                                                |
| `lance.compaction.target_rows_per_fragment`      | Non-negative integer                                                     | Target number of rows per file used to decide which fragments are candidates for [compaction](../../guide/read_and_write.md#compact-data-files). Fragments with fewer rows than this are compacted. Defaults to 1,048,576 (1 Mi).                                                              |
| `lance.compaction.max_rows_per_group`            | Non-negative integer                                                     | Max number of rows per row group when fragments selected for compaction are rewritten. Defaults to 1024.                                                                                                                                                                                  |
| `lance.compaction.max_bytes_per_file`            | Non-negative integer                                                     | Max number of bytes per data file when fragments selected for compaction are rewritten.                                                                                                                                                                                                   |
| `lance.compaction.materialize_deletions`         | Boolean (`true`/`false`)                                                 | Whether to rewrite fragments that have deletions so the deleted rows are removed, even if they otherwise wouldn't need compaction. Defaults to `true`.                                                                                                                                   |
| `lance.compaction.materialize_deletions_threshold` | Float (0.0–1.0)                                                        | Fraction of rows that must be deleted in a fragment before its deletions are materialized. `0` (or lower) materializes deletions for all fragments with deletions; values above `1.0` never materialize deletions. Defaults to `0.1`.                                                   |
| `lance.compaction.defer_index_remap`             | Boolean (`true`/`false`)                                                 | If `true`, indices are not remapped as part of this compaction. Instead, the fragment reuse index is updated, and remapping is deferred to a later operation.                                                                                                                            |
| `lance.compaction.index_remap_mode`              | `direct` or `merged`                                                     | How the old-to-new row-address mapping used to remap indices is built. Defaults to `direct`.                                                                                                                                                                                              |
| `lance.compaction.batch_size`                    | Non-negative integer                                                     | Batch size to use when scanning the input fragments during compaction.                                                                                                                                                                                                                    |
| `lance.compaction.io_buffer_size`                | Non-negative integer                                                     | Number of bytes to allow to queue up in the I/O buffer when scanning input fragments during compaction.                                                                                                                                                                                   |
| `lance.compaction.compaction_mode`               | `re-encode`, `try-binary-copy`, or `force-binary-copy`                   | The compaction mode to use. `try-binary-copy` copies compatible pages verbatim without re-encoding when possible; `force-binary-copy` requires it. Takes priority over the deprecated `enable_binary_copy`/`enable_binary_copy_force` fields.                                            |
| `lance.compaction.binary_copy_read_batch_bytes`  | Non-negative integer                                                     | Read batch size, in bytes, used when performing binary-copy compaction. Defaults to 16 MiB.                                                                                                                                                                                               |
| `lance.compaction.max_source_fragments`          | Non-negative integer                                                     | Maximum number of source fragments to compact in a single run, enabling incremental compaction. Defaults to unlimited.                                                                                                                                                                    |
| `lance.compaction.max_source_rows`               | Non-negative integer                                                     | Maximum number of live source rows (physical rows minus soft-deleted rows) to compact in a single run. Defaults to unlimited.                                                                                                                                                            |
| `lance.compaction.max_source_bytes`              | Non-negative integer                                                     | Maximum number of source bytes (data files and [data overlay files](data_overlay_file.md), excluding blob v2 payloads) to compact in a single run. Defaults to unlimited.                                                                                                                |
| `lance.compaction.max_overlays_per_fragment`     | Non-negative integer or `none`                                          | Maximum number of [data overlay files](data_overlay_file.md) a fragment may carry before it is fully compacted (overlays and deletions materialized into the base data). Defaults to `10`. Set to `0` to compact every fragment with any overlay, or `none` to disable the trigger.     |

</div>

Unrecognized `lance.compaction.*` keys are ignored with a warning; unrecognized keys under other `lance.*` prefixes should likewise be ignored by conforming readers rather than treated as errors.
