# Atomic snapshot retention

Version 1 is an **unstable**, opt-in table metadata format. It does not change
ordinary table layouts before activation. Its JSON schemas are defined in
[snapshot_retention.schema.json](snapshot_retention.schema.json). Apply the named
`$defs` schema below to each object. Unknown protocol versions, malformed records,
unknown fields, and invalid identities must fail closed.

## Activation and authority

Activation requires draining all queries, writers, reference mutations, cleanup,
version expiry, and submitted storage requests. Every later mutating client must
implement this protocol. Activation is permanent. Query acquisition must fail on
an inactive table; it must never activate a table implicitly. Existing handles
and historical manifests cannot enforce this deployment boundary themselves.

Storage must provide consistent reads and listing, atomic conditional creation,
and conditional replacement against the generation returned by the same GET as
the observed bytes. Activation probes replacement and rejection of a stale
precondition. A backend that ignores conditions is unsupported. A failed probe
may leave `mode: preparing`; this is not activation. The transition to `active`
records the current main head and a fresh nonzero `table_id` UUID.

`_refs/retention/state.json` is the authoritative `root` object. Each successful
replacement increments `revision` and chooses a fresh nonzero `mutation_id` UUID,
including semantically empty recovery fences. Revisions must not overflow.
This prevents a value from returning to the same bytes after intervening changes.
The record is never deleted or reset. Local implementations serialize replacement
through the persistent sibling `state.json.cas-lock` inode; lock files must never
be deleted while clients can use them.

A `head` names an exact committed manifest and its optional storage ETag and
commit-store identity. `scheme` is `V1` or `V2`, as defined by manifest naming.
The ETag is an optimization and must not replace content identity. `head: null`
means no committed head has been recorded. Detached versions do not advance it.

## Addresses and record layout

Publication, plan, head, retired-name, and chunk paths are normalized
slash-separated addresses relative to the authenticated table root. The empty
address identifies the root. In a per-manifest snapshot record,
`identity.manifest_path` is the canonical physical object-store path including
the root prefix; `manifest_key` is its root-relative address. Plan retirement
identities instead use the root-relative address. Transported SnapshotRef paths
remain resolved paths in their authenticated source store.
Absolute paths, escaping components, and objects outside the root are rejected.
Local aliases must resolve to the same canonical physical root and file addresses,
including when a file was already deleted and only its parents survive. Catalog
names and access-context identifiers must not create independent authorities for
one physical manifest.

```text
_refs/
    retention/
        state.json                         root
        snapshots/{manifest_hash}.json     snapshot
        publications/{uuid}/
            manifest                       private manifest copy
            intent.json                    publication
            references-{number}.json       array of path strings
        plans/{uuid}/
            plan.json                      plan
            files-{number}.json            array of plannedFile objects
        retired/{shard}.json               retired
        tag-operations/{uuid}.json          operation
        branch-operations/{uuid}.json       operation
    tags/lance-query-{digest}.json          diagnostic query projection
```

Immutable documents are conditionally created before their root pointer is
published. Replaying the same address is allowed only with identical bytes.
Reference and planned-file chunks contain at most 1,024 entries. Headers and
chunks must belong to the same operation directory. Root IDs must match the
referenced header IDs. Readers validate addresses before following them.

`manifest_hash` is the lowercase 64-digit BLAKE3 hex digest of the normalized
physical manifest address's UTF-8 bytes. `manifest_key` repeats that address and
must match the addressed object. `shard` is the first four lowercase hex digits
of BLAKE3 of a retired physical path's UTF-8 bytes. Retired-name shards store a
set of exact path strings, not path prefixes. Their revision and mutation ID
follow the root replacement rules; shard tombstones never expire.

## Persisted snapshot and release tokens

`protos/snapshot.proto` defines `lance.snapshot.SnapshotRef`, protocol version 1.
`protos/snapshot_retention.proto` defines `lance.dataset.retention.SnapshotRetention`,
protocol version 1. The latter embeds the encoded former in `snapshot`.
These schemas are persisted contracts wherever tokens appear below; they are
also used for execution transport. Neither message contains credentials,
manifest contents, or an authorization capability.

The manifest digest is 32 BLAKE3 bytes over the complete original manifest file,
including metadata sections, dictionaries, inline transactions, length prefixes,
and footer. Protobuf re-encoding must not substitute for the original file bytes.
The optional manifest identity is the authoritative commit-store identity;
absence means that only the digest supplies content identity. Branch absence
identifies main. A snapshot must name an exact positive version and manifest.

The snapshot ID is BLAKE3 of the protobuf encoding of its validated known fields,
serialized in field-number order with absent/default proto3 fields omitted. The
retention name is `lance-query-` followed by lowercase hex BLAKE3 of the 32 snapshot
ID bytes followed immediately by the nonempty retention ID's UTF-8 bytes. There
is no separator. Retention IDs identify distinct query lifetimes and must not be
reused after release.

`protection`, `source_protection`, `old_protection`, and `new_protection` contain
lowercase hexadecimal protobuf release-token bytes. `null` means no protection.
Decode both protocol versions before recovery uses a token. Token validation
alone does not acquire, renew, release, or authorize protection.

## Tags and branches

For activated tables the root's `tags` and `branches` maps overlay the historical
files described in [Branch and Tag Specification](branch_tag.md). An absent map
entry falls back to historical metadata. A `null` tag entry is a deletion
tombstone and suppresses that fallback. New activated references need no
`_refs/tags/{name}.json` or `_refs/branches/{name}.json` file.

Tag `contents` retains the historical tag fields. Branch `contents` retains the
historical branch fields and its `identifier.version_mapping`: an ordered array
of `[parent_version, incarnation_id]` pairs. Each incarnation ID is a 32-digit
UUID hex string. Main has the table UUID as its publication incarnation. A named
branch's publication incarnation is the table UUID, `/`, and compact JSON of its
identifier, preserving the pair order and decimal version encoding.

A tag mutation writes an `operation` journal before acquiring its new source pin.
It commits against `expected_revision` and includes the new token in the root.
Only after that decision may it release the old token. Tag deletion publishes a
null entry before releasing old protection. `tag` names the affected reference;
branch journals use this field for the branch name too. Missing new or old tokens
mean that side of the mutation does not own protection.

A branch transitions `creating` → `active` → `retiring` → `deleted`. Creation
holds `source_protection` until its first manifest publication is finalized. Its
`first_version` exceeds the previous incarnation's `max_version`, including
uncertain submitted publications. The watermark must not decrease or overflow.
Retirement fences new publications and pins before scanning surviving references.
A non-null retirement `plan_id` identifies its committed immutable deletion plan.
Recreation changes the identifier; old manifests and old writers must not be
accepted into the new incarnation. Branch paths remain under `tree/{name}/`.

Query projections use the historical tag schema with metadata keys
`lance.snapshot_retention.snapshot_id` (snapshot ID hex),
`lance.snapshot_retention.manifest_path` (physical address), and
`lance.snapshot_retention.token` (release-token hex). They are diagnostic only.
Missing or stale projections never remove pins. General tag mutation must reject
query projections; only release may remove them.

## Pins and irreversible retirement

A `snapshot` object contains the manifest identity and `phase`. An `active`
phase maps each retention name to its 32 snapshot ID bytes. Pin acquisition
conditionally creates or replaces this object. Replaying the same name is valid
only for the same snapshot ID. After pinning, acquisition rereads the exact
original manifest bytes, validates its digest and commit identity, and checks
branch incarnation before acknowledging protection. Failed or uncertain
acquisition preserves the pin for explicit recovery using the saved intent token.

Release removes only the named pin with the matching snapshot ID. It is
idempotent, including after manifest deletion. The application must stop all
consumers, retries, outstanding reads, and lazy readers before release. Query
pins have no lease, timeout, renewal, or automatic orphan expiry.

Retirement conditionally changes an unpinned `active` observation to `retiring`
with a fresh `decision_id`. An absent state may be retired directly. A pin or
pin/release cycle since the scan invalidates that observation. `digest: null`
is permitted only for irreversible path retirement discovered through listing;
it never authorizes an acknowledged query pin. `commit_identity: null` means no
authoritative record identity was supplied. Completion changes `retiring` to
`deleted` with the same decision. Neither phase can return to active. These
records remain permanently, even when a manifest delete failed.

## Publication and latest reads

Before submitting a manifest, a writer creates a private manifest copy and a
`publication` intent containing all owned physical data, deletion, index,
transaction, managed Blob, FRI, and target manifest references. Foreign roots
require separate owner protection. Every actual handler-generated manifest or
staging name must be registered before that write is submitted.

The writer stamps manifest configuration keys `lance.snapshot_retention.root`,
`lance.snapshot_retention.publication`, and `lance.snapshot_retention.incarnation`
with the table root, publication UUID, and observed incarnation. It checks new
paths against permanent retired-name shards and all committed plans, then
conditionally adds the publication to the observed root. References inherited
from that root's already validated committed head may be reused. The exact
predecessor observation must still win the root CAS, or be revalidated. Shard
reads may run concurrently. A pending intent protects every recorded reference.

After the immutable manifest lands, finalization conditionally advances the head
and removes the pending publication. A verified successful commit stays
successful if bookkeeping fails; retain the intent and recover later. A positively
observed retired or changed incarnation still rejects the stale commit.

Latest reads observe the root once and consider same-branch, same-incarnation
pending publications in descending version order. Resolve each immutable slot and
verify its stamped publication UUID before selecting a landed manifest newer
than the recorded head. Missing slots and slots won by another publication are
ignored. Reads perform no recovery writes. Writers, maintenance, and explicit
recovery may finalize landed publications. A competing immutable commit proves a
pending creation cannot win, allowing its exact intent to be removed. Absence
alone never proves an uncertain write cannot arrive later.

## Cleanup, recovery, and reuse

Cleanup observes the root before scanning reachability. It preserves retained
snapshots, all tag and branch sources, and all pending publication references.
It irreversibly retires eligible unpinned manifests against their original pin
observations, writes exact candidate chunks, and conditionally publishes the
plan against the scanned root before deleting any shared object.

A root conflict caused only by publication registration or finalization may be
revalidated without a full scan: reference contents, phases, source protections,
table identity, mode, and existing plans must be unchanged. Check every added or
changed pending intent and all newly committed manifests against the immutable candidate
chunks. Inspect the attached version suffix after each observed head, including
intermediate commits superseded before readback, and detached manifests; looking
only at the newest head can miss a briefly published reference. Any overlap or any other reference change invalidates the plan. Preserve
the original per-snapshot retirement observations throughout revalidation.

Before a plan's root pointer can be removed, every candidate name must be added
to its permanent retired-name shard. Physical deletion may then be replayed using
only the committed chunks, preserving shared references and the recorded
retirement decisions. `retired_records` maps decimal version strings to external
commit-store record identities, removed only after their manifest deletion.
Private random data `.lance-stage` copies are never published references and may
be deleted on promotion or abort without a retention lookup.

Recovery observes journals, then fences `expected_revision` before releasing
unreferenced tokens. An operation that began after that fence keeps its pin.
Journals remain discoverable while a delayed acquisition could still land;
irreversible retirement is required before dropping that release ownership.
Recovery never releases application query pins automatically. Failed deletion
resumes the same plan and cannot reuse retired names. Explicit forced branch
deletion bypasses this contract and requires draining affected clients.

Inactive tables keep the historical cleanup and branch deletion behavior. They
must not take this protocol's locks or emit retirement records implicitly.
Activated tables must reject uncoordinated version expiry and manifest-path
migration. Deployment upgrades must cover historical handles as well as latest
opens; the root record does not revoke operations already submitted by old code.
