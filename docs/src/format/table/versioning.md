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
| `2^14`   | `FLAG_MANAGED_BLOBS`            | Yes             | Yes             | Blob descriptors independently address owned objects. Readers resolve the descriptor's base and path; cleanup preserves objects referenced by protected snapshots, independently of the original data file. Both bits remain set on later versions, including restores. See [Blob v2 capability and compatibility](blob.md#capability-and-compatibility). |
| `2^15`   | `FLAG_NON_REUSABLE_FIELD_IDS`    | No              | Yes             | The manifest sets `max_allocated_field_id`, and a writer must assign new field IDs above it. Before activation, all clients that can write to the dataset must enforce writer feature flags. See [Field IDs](schema.md#field-ids). |
| `2^16`   | `FLAG_ZONE_WRITE_SEEDS`         | Yes             | Yes             | Referenced data files contain version 1 [zone write seeds](../index/scalar/zone_write_seeds.md). Both bits remain set in derived snapshots, including clone and restore. Reserved and unsupported until implemented. |
| `2^17`   | `FLAG_OBJECT_METADATA`         | No              | Yes             | The snapshot contains [clustering definitions, references, or a nonzero allocation high-water mark](clustering.md), nonempty [fragment metadata](index.md#fragment-metadata), or nonempty [index segment metadata](../index/index.md#index-segment-metadata). Reserved, unsupported until writers implement the maintenance rules. Readers may ignore this metadata. |
| `2^19`   | `FLAG_ONGOING_WORK_PINS` | No | Yes | The current manifest owns the active [ongoing-work pin list](ongoing_work_pins.md). Sticky after first activation, including restore and an empty list. Reserved and unsupported until pin maintenance is implemented. |

</div>

Bits 16, 17, and 19 are reserved but remain unsupported. Bit 18 is not assigned by this specification and must remain unsupported until separately defined and implemented. Bits at positions 20 and above are unknown. Unknown or unsupported flags cause implementations to reject the corresponding read or write operation with an "unsupported" error. The paired mixed-version, Managed Blob, and zone write-seed reader and writer bits must each either both be set or both be clear; a half-set manifest is invalid.

Committing schema metadata `lance.blob.managed = "true"` activates the Managed Blob capability, even without new blob files. Ordinary Blob v2 writes without this opt-in do not activate it. Once activated, removing the opt-in, removing Blob fields, or restoring an older snapshot does not clear it. Before activation, all maintenance clients must support Managed Blobs: the flag cannot revoke old handles or prevent older clients from running cleanup through an unflagged historical snapshot. See [client compatibility](../../guide/blob.md#managed-objects-and-client-compatibility).

## Current state for pin decisions

Pin acquisition, recovery, renewal, expiration, ordinary writes carrying pins,
restore, overwrite, compaction, retention planning, and branch-deletion checks
must resolve the authoritative current attached manifest. A previously opened
Dataset, historical transaction read-back, or hint that stops probing at a gap
is insufficient.

For ordinary file/object-store commit handlers, enumerate attached manifests
using the active naming scheme, exclude detached and staging files, and choose
the highest version. On an unordered store, consume every listing page. A
lexically ordered V2 listing may stop at its first valid attached manifest when
that naming scheme's ordering guarantee applies. Read the selected manifest;
if it disappeared, restart resolution rather than accepting an older cached
result. Propagate listing/read errors. An optimization is allowed only if it
preserves this result in the presence of reclaimed slots and delayed writes.
A hint that moves forward alone is not a correctness proof.

External commit stores must resolve and finalize their selected authoritative
head through their own commit protocol; do not replace their coordination with
an uncoordinated object listing. A backend must provide the visibility guarantees
required by its commit protocol or refuse pin acquisition. These reads do not
lock current state. Apply the transaction's publication and retry rules after
resolution; the [maintenance concurrency limits](ongoing_work_pins.md#maintenance-concurrency-limits)
remain even when these reads succeed.
