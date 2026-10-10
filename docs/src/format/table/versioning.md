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

| Bit Mask | Flag Name                       | Reader Required | Writer Required | Description                                                                                                 |
|----------|---------------------------------|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------|
| `2^0`    | `FLAG_DELETION_FILES`           | Yes             | Yes             | Fragments may contain deletion files, which record the tombstones of soft-deleted rows.                     |
| `2^1`    | `FLAG_STABLE_ROW_IDS`           | Yes             | Yes             | Row IDs are stable for both moves and updates. Fragments contain an index mapping row IDs to row addresses. |
| `2^2`    | `FLAG_USE_V2_FORMAT_DEPRECATED` | No              | No              | Files are written with the new v2 format. This flag is deprecated and no longer used.                       |
| `2^3`    | `FLAG_TABLE_CONFIG`             | No              | Yes             | Table config is present in the manifest.                                                                    |
| `2^4`    | `FLAG_BASE_PATHS`               | Yes             | Yes             | Dataset uses multiple base paths (for shallow clones or multi-base datasets).                               |
| `2^5`    | `FLAG_DISABLE_TRANSACTION_FILE` | No              | Yes             | Transactions are recorded in the manifest rather than in a separate transaction file.                       |
| `2^6`    | `FLAG_UNSTABLE_DATA_OVERLAY_FILES` | Yes          | Yes             | Fragments may carry data overlay files. Unstable: release builds reject it unless explicitly opted in.      |
| `2^7`    | `FLAG_COVERED_INDEX_METADATA`   | Yes             | Yes             | Some index declares covering columns (`IndexMetadata.covering_fields`). Without `FLAG_INDEPENDENT_COVERING_FIELDS`, the carried columns form the trailing suffix of `fields`. An implementation without this flag may select or maintain an index using the wrong fields. |
| `2^8`    | `FLAG_MIXED_DATA_FILE_VERSIONS` | Yes             | Yes             | The snapshot may reference recognized V2 data files with different exact versions. Both bits must be set and remain set on later versions. |
| `2^9`    | `FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS` | Yes       | Yes             | The table uses stable row IDs and carries a [Fragment Reuse Index](../index/system/frag_reuse.md). |
| `2^10`   | `FLAG_FRAGMENT_REUSE_INDEX`     | Yes             | Yes             | The fragment reuse index records tagged transitions (`IndexMetadata.index_version >= 1`). Readers must translate row addresses through them; writers must preserve them. An implementation without this flag would decode the details as the legacy format and silently drop the transitions when it next rewrites the fragment reuse index. See [FRI index versions](../index/system/frag_reuse.md#fri-index-versions). |
| `2^11`   | `FLAG_SPILLED_ROW_LINEAGE` | Yes         | Yes             | Some fragment stores its row ids or row version sequences as hidden columns of a data file rather than inline. A reader without this flag would see the fragment as having no row ids. Readers and writers that do not support this flag must refuse the dataset. Spilling remains opt-in through table configuration. |
| `2^12`   | `FLAG_FRAGMENT_TREE`            | Yes             | Yes             | Fragment records live in a [fragment tree](fragment_metadata.md). `Manifest.fragments` is empty. |
| `2^13`   | `FLAG_INDEPENDENT_COVERING_FIELDS` | Yes          | Yes             | Requires `FLAG_COVERED_INDEX_METADATA` and is retained together with it. `IndexMetadata.fields` contains only key fields, while `covering_fields` independently declares carried fields and may overlap `fields`. Implementations that only support the legacy suffix contract must reject the dataset. |
| `2^14`   | `FLAG_MANAGED_BLOBS`            | Yes             | Yes             | Blob descriptors independently address owned objects. Readers resolve the descriptor's base and path; cleanup preserves objects referenced by protected snapshots, independently of the original data file. Both bits remain set on later versions, including restores. |
| `2^15`   | `FLAG_NON_REUSABLE_FIELD_IDS`    | No              | Yes             | The manifest sets `max_allocated_field_id`, and a writer must assign new field IDs above it. Before activation, all clients that can write to the dataset must enforce writer feature flags. See [Field IDs](schema.md#field-ids). |

</div>

Bits at positions 16 and above (`2^16`, `2^17`, and so on) are unknown; unknown flags cause implementations to reject the dataset with an "unsupported" error. The paired mixed-version and Managed Blob reader and writer bits must each either both be set or both be clear; a half-set manifest is invalid.

Publishing a new data file containing a Blob v2 field activates the Managed Blob capability, including files containing only inline values. Metadata-only updates and deletion vectors on existing files do not activate it. Once activated, removing Blob fields or restoring an older snapshot does not clear it. Before activation, all maintenance clients must support Managed Blobs: the flag cannot revoke old handles or prevent older clients from running cleanup through an unflagged historical snapshot. See [client compatibility](../../guide/blob.md#managed-objects-and-client-compatibility).
