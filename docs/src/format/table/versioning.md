# Format Versioning

## Feature Flags

As the table format evolves, new feature flags are added to the format.
There are two separate fields for checking for feature flags,
depending on whether you are trying to read or write the table.
Readers should check the `reader_feature_flags` to see if there are any flag it is not aware of.
Writers should check `writer_feature_flags`. If either sees a flag they don't know,
they should return an "unsupported" error on any read or write operation.

## Current Feature Flags

<style>
.feature-flags-table th:nth-child(2),
.feature-flags-table td:nth-child(2) {
  white-space: nowrap;
  min-width: 250px;
}
</style>

<div class="feature-flags-table" markdown="1">

| Flag Bit | Flag Name                       | Reader Required | Writer Required | Description                                                                                                 |
|----------|---------------------------------|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------|
| 1        | `FLAG_DELETION_FILES`           | Yes             | Yes             | Fragments may contain deletion files, which record the tombstones of soft-deleted rows.                     |
| 2        | `FLAG_STABLE_ROW_IDS`           | Yes             | Yes             | Row IDs are stable for both moves and updates. Fragments contain an index mapping row IDs to row addresses. |
| 4        | `FLAG_USE_V2_FORMAT_DEPRECATED` | No              | No              | Files are written with the new v2 format. This flag is deprecated and no longer used.                       |
| 8        | `FLAG_TABLE_CONFIG`             | No              | Yes             | Table config is present in the manifest.                                                                    |
| 16       | `FLAG_BASE_PATHS`               | Yes             | Yes             | Dataset uses multiple base paths (for shallow clones or multi-base datasets).                               |
| 32       | `FLAG_DISABLE_TRANSACTION_FILE` | No              | Yes             | Transactions are recorded in the manifest rather than in a separate transaction file.                       |
| 64       | `FLAG_UNSTABLE_DATA_OVERLAY_FILES` | Yes          | Yes             | Fragments may carry data overlay files. Unstable: release builds reject it unless explicitly opted in.      |
| 128      | `FLAG_COVERED_INDEX_METADATA`   | Yes             | Yes             | Some index declares covering columns (`IndexMetadata.covering_fields`). Without `FLAG_INDEPENDENT_COVERING_FIELDS`, `fields` contains keyed columns followed by carried ones. An implementation without this flag may select or maintain an index using the wrong fields. |
| 2048     | `FLAG_INDEPENDENT_COVERING_FIELDS` | Yes          | Yes             | Requires `FLAG_COVERED_INDEX_METADATA`. `IndexMetadata.fields` contains only key fields, while `covering_fields` independently declares carried fields and may overlap `fields`. Implementations that only support the trailing-suffix contract must reject the dataset. |

</div>

Unlisted flags are unknown and cause implementations to reject the dataset with an "unsupported" error.
