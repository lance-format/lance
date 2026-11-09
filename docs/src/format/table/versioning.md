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
| `2^10`   | `FLAG_FRAGMENT_REUSE_INDEX`     | Yes             | Yes             | The fragment reuse index uses version 1 or later (`IndexMetadata.index_version >= 1`). Readers must translate row addresses through its transitions; writers must preserve them. An implementation without this flag would decode the details as the legacy format and silently drop the transitions when it next rewrites the fragment reuse index. See [FRI index versions](../index/system/frag_reuse.md#fri-index-versions). |
| `2^11`   | `FLAG_SPILLED_ROW_LINEAGE` | Yes         | Yes             | Some fragment stores its row ids or row version sequences as hidden columns of a data file rather than inline. A reader without this flag would see the fragment as having no row ids. Readers and writers that do not support this flag must refuse the dataset. Spilling remains opt-in through table configuration. |
| `2^12`   | `FLAG_FRAGMENT_TREE`            | Yes             | Yes             | Fragment records live in a [fragment tree](fragment_metadata.md). `Manifest.fragments` is empty. |
| `2^13`   | `FLAG_INDEPENDENT_COVERING_FIELDS` | Yes          | Yes             | Requires `FLAG_COVERED_INDEX_METADATA` and is retained together with it. `IndexMetadata.fields` contains only key fields, while `covering_fields` independently declares carried fields and may overlap `fields`. Implementations that only support the legacy suffix contract must reject the dataset. |
| `2^14`   | `FLAG_MANAGED_BLOBS`            | Yes             | Yes             | Blob descriptors independently address owned objects. Readers resolve the descriptor's base and path; cleanup preserves objects referenced by protected snapshots, independently of the original data file. Both bits remain set on later versions, including restores. See [Blob v2 capability and compatibility](blob.md#capability-and-compatibility). |
| `2^15`   | `FLAG_NON_REUSABLE_FIELD_IDS`    | No              | Yes             | The manifest sets `max_allocated_field_id`, and a writer must assign new field IDs above it. Before activation, all clients that can write to the dataset must enforce writer feature flags. See [Field IDs](schema.md#field-ids). |
| `2^16`   | `FLAG_ZONE_WRITE_SEEDS`         | Yes             | Yes             | Referenced data files contain version 1 [zone write seeds](../index/scalar/zone_write_seeds.md). Both bits remain set in derived snapshots, including clone and restore. |
| `2^17`   | `FLAG_OBJECT_METADATA`         | No              | Yes             | The snapshot contains [clustering definitions, references, or a nonzero allocation high-water mark](clustering.md), nonempty [fragment metadata](index.md#fragment-metadata), or nonempty [index segment metadata](../index/index.md#index-segment-metadata). Readers may ignore this metadata; writers must preserve it. |
| `2^18`   | `FLAG_ROW_ADDR_INDICES`         | No              | Yes             | A stable-row-ID table contains an index that stores physical row addresses. Older writers could treat its entries as stable row IDs and silently extend coverage without indexing rewritten fragments. |
| `2^19`   | `FLAG_INLINE_DELETIONS`         | Yes             | Yes             | Some fragment stores its deletion vector in the manifest (`DeletionFile.inline`) rather than in a file under `_deletions/`. A reader without this flag would look for a file that does not exist. |
| `2^20`   | `FLAG_RETENTION_LEASES` | No | Yes | The current manifest owns the [retention lease list](retention_leases.md). Sticky after first activation, including restore and an empty list. Reserved and unsupported until lease maintenance is implemented. |

</div>

Bit 20 is reserved but remains unsupported. Bits at positions 21 and above are unknown. Unknown or unsupported flags cause implementations to reject the corresponding read or write operation with an "unsupported" error. The paired mixed-version, Managed Blob, and zone write-seed reader and writer bits must each either both be set or both be clear; a half-set manifest is invalid.

Committing schema metadata `lance.blob.managed = "true"` activates the Managed Blob capability, even without new blob files. Ordinary Blob v2 writes without this opt-in do not activate it. Once activated, removing the opt-in, removing Blob fields, or restoring an older snapshot does not clear it. Before activation, all maintenance clients must support Managed Blobs: the flag cannot revoke old handles or prevent older clients from running cleanup through an unflagged historical snapshot. See [client compatibility](../../guide/blob.md#managed-objects-and-client-compatibility).

## Current state for lease decisions

Lease operations and recovery, commits that release a lease, restore,
overwrite, rewrite/trim checks, and retention planning through
a lease-capable handler **must resolve the authoritative current attached head**.
This includes maintenance triggered through an old or unflagged Dataset handle.
A cached version or a hint that stops at a reclaimed gap is insufficient.

- **Direct storage:** enumerate attached manifests in the active naming scheme,
  excluding detached and staging files, and select the highest version. Consume
  every unordered listing page. An ordered V2 listing may stop at its first
  valid manifest when that ordering guarantee applies.
- **External coordination:** use the coordinator's resolution and finalization
  protocol. Do not substitute an uncoordinated object-store listing.

Read the selected manifest. If it disappears, resolve again; propagate other
listing/read errors. Optimizations must preserve this result despite reclaimed
slots and delayed writes. A backend unable to provide its protocol's visibility
guarantees must refuse acquisition.

Ordinary attached writes that only carry leases and prune expired records may
use the existing commit/retry path. Each attempt **must derive membership and
the sticky capability from its predecessor**, recheck expiration after rebase,
and atomically claim the next version without replacing an occupied slot.
A historical-slot write cannot change the real head's lease state or end retention.
Required transaction recording still applies; subsequent maintenance must
resolve the real head. External handlers must provide equivalent guarantees.

An unsupported handler may keep existing maintenance behavior on a table that
has never enabled leases, but **must reject** any lease capability or record it
encounters. This relies on the same-handler table contract and exclusion of
incompatible maintenance clients at activation, not an old handle's flags.
After activation, all maintenance clients must use a lease-capable path even when
the list is empty. Mixing capable and incapable implementations of the same
handler type does not satisfy this requirement.

Head resolution establishes current lease state, not output-publication validity.
It does not lock the table; the [cleanup and deletion races](retention_leases.md#cleanup-and-deletion-races)
still apply.
