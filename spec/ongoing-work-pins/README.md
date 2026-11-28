# Ongoing-work pins: implementation design

Status: draft for design review. No runtime implementation is included.

This design builds on [format PR #10428](https://github.com/lance-format/lance/pull/10428) and the [design discussion](https://github.com/lance-format/lance/discussions/10395). This local revision incorporates the agreed response to Will's review: registration chooses its source when it commits, ordinary commits remove expired pins, publication can explicitly release a pin, and pin IDs use the existing UUID type. The format vote remains a separate prerequisite.

Review this file with [test cases](tests.md) and [performance expectations](performance.md). The proposed API names, backend matrix, and benchmark workloads below need agreement during this review. The retention and lifecycle decisions carry forward the existing design.

## Scope

Implement the reusable Rust pin mechanism and all maintenance behavior needed to use it safely. A small worker exercises the complete lifecycle in tests. Production async-index integration, language bindings, refresh-column jobs, and their scheduling follow separately.

A pin keeps a fixed source version, intervening publication evidence, and applicable FRI movement history available. It does not establish whether a job's result is still valid. Existing publication conflict checks continue to decide that.

Keep pin state in the current attached manifest and serialize mutations through the existing commit protocol. Use RegisterPin, RenewPin, and ReleasePin. Expiration is part of ordinary attached commits, with no dedicated expiration operation or cleanup fallback commit. Add the optional `Transaction.release_pin_id` field for atomic publication and release. No service, background renewal loop, branch-storage redesign, or Transaction V2 dependency is needed.

## Rust API and recovery

Follow Lance's existing builder and transaction conventions. The proposed `PinRegistrationBuilder` constructs a request before submission, as `TransactionBuilder` constructs a transaction before `CommitBuilder` submits it. Centralize validation in Rust and expose registration, renewal, release, and inspection through a dataset-level pin accessor. These pin APIs are not implemented yet.

Illustrative Rust usage:

```rust
let request = PinRegistrationBuilder::new(dataset.clone())
    .with_duration(Duration::from_secs(24 * 60 * 60))
    .with_job_label("async-index-build")
    .build()
    .await?;

// A job that needs crash recovery saves this request before submitting it.
let outcome = dataset.pins().register(&request).await;
```

`new` and the setters only configure the builder. `build()` resolves the target branch and checks the current head and handler support, but does not choose the job's source or publish a pin. The duration setter is optional because 24 hours is the default. The caller plans and reads only after a Confirmed outcome, using the returned source version.

| Operation | Input and result |
| --- | --- |
| `PinRegistrationBuilder::new(dataset).build().await` | Return a `PinRegistration` containing a fresh transaction UUID, bound table/branch identity, fixed timestamps, and optional label. Its initial read version supports commit retries; it does not fix the pinned source. Configure duration and label through `with_duration(...)` and `with_job_label(...)`. Building publishes nothing. |
| `dataset.pins().register(&request)` | Submit or recover the same request without consuming it. Return `RegistrationOutcome::Confirmed(PinHandle)`, `Rejected(reason)`, or `Indeterminate(reason)`. The caller retains the request whenever the outcome is indeterminate. |
| `renew(handle, expires_at)` | Request one fixed absolute expiration time. Return the current handle after committing an extension, or without a commit if it is already satisfied. A missing pin or deadline already passed at validation is an error. |
| `release(handle)` | Remove this pin ID. An absent ID is success without a commit. This does not cancel an unconfirmed registration still in flight. |
| `list()` | Read active records from the authoritative head for recovery and operator inspection. A record past `expires_at` is still active until removal commits. |
| `Transaction.release_pin_id` | Optionally name one pin to release with an ordinary attached operation. The publication transaction has its own UUID. The field is invalid on pin operations, Clone, and detached commits. |

`PinRegistration` and `PinHandle` are caller-persistable Rust values. Bind them to the table location, branch, and existing named-branch identifier when present. The request holds the exact bytes used for the registration; the handle identifies the pin and its fixed source version. Validate imported values before use. Reject a different table, branch, or branch identifier.

Reuse `BranchContents::from_path` and its existing identifier comparison, including the deterministic fallback for older branch files. Identical legacy metadata can produce the same identifier after recreation. Requests and handles must not be reused across such recreation or whole-table deletion and recreation, even at the same path. The owning service coordinates those lifecycles; the pin API does not detect every replacement. No branch metadata migration is needed.

Use `TransactionBuilder` internally to construct the RegisterPin transaction and generate its UUID, then submit through the existing commit machinery. Parse that UUID into the existing 16-byte UUID type for the pin ID and use typed UUIDs for all pin references. Reject missing or incorrectly sized stored IDs. The pin accessor adds confirmation and typed outcomes; keep the generic `CommitBuilder::execute` public return type unchanged. A `PinRegistration` holds the transaction and target/branch identity needed for recovery. Add publication's optional release ID using the existing transaction-builder conventions.

Use the existing timestamp representation with checked conversion to protobuf timestamps. The registration builder defaults to a 24-hour duration and permits an override. Building the request fixes `created_at` and `expires_at`; submitting or retrying that request never refreshes them or generates another ID. Calling `build()` again creates a new registration request, not a retry. A consumer chooses its renewal margin, for example 30 minutes before expiration. There is no maximum lifetime or automatic renewal cadence.

Use the existing bounded commit retry settings. Retrying an unconfirmed request preserves its ID and times while allowing a newer head. The source is the version created by its successful registration commit. Recovering an already active ID returns that stored source without moving it. Reject a fresh registration whose fixed deadline has passed. Exhausted confirmation reads after a possibly successful commit return `Indeterminate`, not a definite rejection. Do not retry registration after a known release or expiration.

Release by an unambiguous ID remains possible even if the source has become unreadable. Do not make release depend on reading the source's fragments or replaying its transactions.

## Registration and current state

1. Validate the request's table location and named-branch identity. Resolve the authoritative attached head and validate its pin list and writer capabilities. For direct storage, use the format's complete listing rule, with the ordered V2 optimization only where valid. For external coordination, use the commit handler's authoritative resolution and finalization protocol.
2. If the requested ID is already active, recover its stored source and current record without resetting expiration or label.
3. Otherwise, validate that the fixed registration deadline is still in the future. Let H be the current attached head. Its creating transaction's UUID or contents are not required solely for registration.
4. Build and atomically publish H+1 with the new pin's source set to H+1. On conflict, retry against the newer authoritative head with the same ID and times. Registration changes no data; the job uses the data in the version that actually commits.
5. Before returning `Confirmed`, check the acquired named-branch identity, current membership with the stored source version, and that source manifest's readability and version. A matching historical transaction alone is insufficient. Return the stored source, including when recovering after a lost response. A recovered expired record does not authorize continuing work without a new acquisition.

Use authoritative-head resolution for pin operations and maintenance, including restore, overwrite, rewrite/trim validation, cleanup, and branch deletion. A stale hint followed by probing until the first gap is insufficient for these decisions: cleanup can leave gaps.

Add `CommitHandler::resolve_latest_location_for_retention(base_path, object_store) -> Result<Option<ManifestLocation>>` as the shared entry point. Return `Some(head)` for a resolved head and `None` only for confirmed absence under the rules below; uncertainty is an error. Its default returns an unsupported-operation error, so existing custom handlers do not silently acquire a guarantee they cannot provide. Keep the existing `resolve_latest_location` behavior for ordinary readers and carry-only writes.

- Supported direct-storage handlers opt in by using complete manifest listing, with the ordered V2 early stop only where valid. Reuse `resolve_version_from_listing`; do not use the hint-and-probe shortcut.
- Supported external handlers resolve through the coordinator and its finalization protocol. A coordinator miss permits a direct-storage fallback only when that backend's protocol establishes that fallback is valid; it must then use the strict listing path. Never replace an uncertain coordinator result with a raw-listing answer.
- Unsafe handlers remain unsupported. Custom handlers must explicitly implement and satisfy this contract before acquisition is enabled. An unsupported backend rejects acquisition before submission; an unresolved read after possible publication can instead make confirmation indeterminate. A generic not-found error alone is not proof that a branch has no attached manifest.

Ordinary attached writes that carry pins and remove expired records use the existing commit and retry protocol without an additional complete-history listing solely for pins. Start from the predecessor manifest used for that attempt, sample the current UTC time, and omit expired records. Preserve every field of the other pins and the sticky bit. After rebasing, use the new predecessor and recheck expiration; never reuse the original read version's pin list or an old removal decision. Continue to persist required transaction evidence when the resulting manifest contains pins.

This relies on the handler atomically claiming predecessor version plus one without replacing an occupied version. A write that advances head H must therefore use H as its predecessor and carry H's pin state. A stale write that fills a reclaimed slot below H does not become the authoritative head. That existing stale-slot behavior is not a guarantee that the write extended the current table, and historical pin-list copies never become a source of current membership merely because they were written successfully. This argument assumes the branch's current head is not deleted or replaced outside the commit protocol; the existing physical-deletion limitations still apply.

This exemption covers carrying pins and removing expired records during an ordinary commit. Explicit pin operations, publication with `release_pin_id`, FRI rewrite/trim checks, and triggered cleanup still use authoritative resolution. Automatic cleanup resolves current membership even through an older or unflagged Dataset handle. Supported external handlers must provide the same predecessor and atomic-publication guarantees.

These reads do not lock physical deletion. The existing restore/cleanup race and delayed branch deletion after recovery remain outside this change. Do not add tombstones or retire branch names.

## Lifecycle and manifest construction

Only the current attached manifest owns active membership. Historical lists never revive or prolong a removed pin.

| Operation | Commit behavior |
| --- | --- |
| Register | Add the prepared ID/times/label, set its source to the publishing version, and set the sticky writer capability. |
| Renew | Require the current pin's deadline to be in the future when validating each attempt. Change only `expires_at`, and only to a later value. |
| Release | Explicitly remove the target ID, even before its deadline. Absence is a no-op; release creates its own commit when the record exists. |
| Ordinary attached commit | Remove expired records using the attempt's current predecessor and time. Include these removals in the commit already being made. |
| Publication with `release_pin_id` | Validate the operation and publication evidence, then apply its result and the named pin's release in one manifest. A missing target is a no-op only for release. |

Passing the timestamp alone does not remove a record. Every successful attached commit can remove expired pins, including commits that register, renew, or release another pin. A renewal extending the deadline into the future defeats an older removal decision; committed removal first makes renewal fail. No-op pin requests do not create a version solely to remove unrelated expired pins. Real mutations use ordinary version counting.

Cleanup never creates a commit just to expire pins. It retains all pins still listed in the authoritative head, including expired records. It may reclaim their resources only after removal commits and retention is resolved again. On quiet tables this can delay reclamation indefinitely; explicit ReleasePin is the escape hatch and can also unblock branch deletion. The same rule protects against a renewal that passes validation before its deadline but is delayed in storage.

Attached restore and overwrite start from the current list, apply the same expiration and explicit-release rules, and carry the remaining pins and sticky bit. They never restore historical membership. New branches and clones start with empty lists while carrying the inherited bit.

Detached versions serialize an empty pin list and preserve the sticky bit when inherited. Creating one leaves the branch's current membership unchanged. All three pin operations and `release_pin_id` are invalid in detached commits.

## Version and transaction retention

Let F be the oldest active source version and H the authoritative attached head. Retain every attached version in the same branch from F through H, inclusive. Detached versions do not join this numeric window.

```text
versions:       20   21   22   23   24   25 (head)
job A source:   ^
job B source:             ^
retained:       [========================]
after A ends:             [==============]
```

Reuse normal version retention for each version in the window. Keep the manifests, their base data, indices, overlays, deletion files, FRI metadata, external mappings, and referenced transaction sidecars. Also retain coordinator records needed to discover the versions. Shared physical files count once. These must remain fully readable versions.

Every attached commit whose resulting manifest contains pins, or which sets `release_pin_id`, persists its own transaction inline or in its referenced sidecar before publication. This includes registration, ordinary writes, renewal, restore, overwrite, and publication that releases the last pin. Disabling sidecar writes is valid only when the inline representation is sufficient; otherwise reject before publication. Validate complete transaction evidence after F through H. The transaction that created F is not required solely for pin registration or this retention rule.

Cleanup resolves current membership and constructs the retention set without dropping records based on time alone. Missing or unreadable required manifests or transaction evidence stop affected destructive cleanup with a protection/integrity error. Do not interpret a missing version as unused space. Preserve the backend's discovery records until the corresponding retained versions can be reclaimed.

Cleanup preview commits nothing and uses the same retained pin set as execution. Report expired records awaiting committed removal, not their resources as immediately reclaimable. Execution resolves current state again. Explain the oldest retained source, pin count, and unique retained bytes by file class, distinguishing overlapping tags and other retained versions.

Renewal never moves F. Explicit release or expiration committed with another operation can advance it; ending the last pin removes this requirement, subject to other references. Restore and overwrite do not shorten surviving pins' windows. A 24-hour default is not a storage bound when jobs renew or an idle table has no commit to remove expired records.

## Compaction and FRI retention

When a rewrite moves fragments needed by active pins, require deferred remapping. Do not switch modes automatically. Reject known disabled or unsupported recording before doing row-rewrite work. Include pinned fragment coverage even when no index covers those fragments and no FRI entry exists yet.

For FRI version 1, load and validate each pin's source-version manifest, resolve its complete fragment metadata through the declared layout, and take all fragment IDs. A flat source uses `Manifest.fragments`; a fragment-tree source requires its referenced metadata. An empty inline list is not evidence that a tree source is empty. Reject acquisition for unsupported layouts and stop affected maintenance if required metadata cannot be resolved. Reuse the supported source-metadata reader; this work does not add fragment-tree support. No row-data scan is needed.

Follow current-history source-to-destination connections from those IDs to a fixed point. A rewrite needs mappings when its source fragments intersect that required fragment set. Keep whole selected records; when the version 1 history includes legacy records, selecting one includes all its groups and downstream destinations. Pin requirements have no index catch-up shortcut that cancels their need.

For legacy-only FRI (`index_version = 0`), retain applicable current-chain records stamped at or after the oldest source, inclusively. Newly published legacy records use publishing version minus one and are restamped on retry. Existing records keep their stamps. Resolve affected pinned fragment coverage for rewrite checks; inability to establish it causes rejection, not empty coverage.

At every rewrite or trim commit attempt, resolve current pins against the authoritative head and apply only the removals published in that same commit. Check the remaining pins before bare-rewrite admission or no-FRI early returns, including rebases. A plan created before registration must satisfy any new pin still retained. Reject missing mappings and replan; do not invent them during commit.

Reuse existing FRI traversal and consumer checks. Retain the union of pin and existing-consumer requirements. A consumer-check error cannot drop pin-required history. If the required source or fragment set cannot be established, refuse the affected rewrite or trim. Key cached source metadata by table location, named-branch identity where present, and source version. Cache lifetime must respect the same table-deletion/recreation boundary as requests and handles.

Restore and overwrite may discard translation paths. Apply the FRI rule to the resulting current history without reconstructing a discarded path. This does not drop historical version retention or make invalidated output publishable. Removing a current FRI record does not delete mapping files still referenced by retained versions.

Preserve FRI version 1 overlay restrictions: a version 1 entry blocks new overlays table-wide, and recording its transitions rejects overlaid source fragments. A bare rewrite cannot bypass that rejection. Pins can keep required version 1 history, and therefore the restriction, around longer. Overlay support is separate work; releasing protection before publication is not a workaround.

## Branches and clones

Normal and force branch deletion both resolve the current head, inspect writer capabilities, validate membership, and reject observed active pins, including records past `expires_at`. The deleter does not expire pins itself. An orphan with definitively no attached manifest has no pins to check; an unreadable or invalid manifest causes failure. If the selected head disappears, resolve again.

For an orphan without a branch reference, establish absence as follows:

| Backend | Evidence required before returning confirmed absence |
| --- | --- |
| Supported direct storage | A successful, complete listing of the branch's `_versions` directory finds no attached manifest. The store's listing guarantees must make that absence reliable; exclude detached and staging files. |
| DynamoDB coordination | A strongly consistent latest-version query returns no record, and a successful, complete object-store listing finds no attached manifest. Use the existing consistent-read setting and verify the combined behavior in integration tests. |
| Namespace or other external coordination | The coordinator's documented consistency and publication protocol establishes absence, and a successful, complete object-store listing finds no attached manifest. If that service cannot establish reliable absence, return an error. Two empty but potentially stale reads are insufficient. |

Permission, transport, listing, finalization, and manifest-validation failures must never become confirmed absence. A coordinator record whose manifest cannot be read also causes failure. Keep this head-absence result separate from `is_version_not_found_definitive`, which concerns a particular version after a commit attempt. Normal and force deletion apply the same distinction between confirmed absence and an inability to check. These checks do not fence concurrent creation or deletion; the existing concurrency limits still apply.

Child branches have independent pin lists. Parent cascade cleanup must retain parent-owned files referenced by a child's retained versions. A separate shallow clone does not create protection in its source table; that source needs independent protection. Keep the existing physical-deletion concurrency limits explicit.

## Integration and activation

| Existing area | Implementation work |
| --- | --- |
| `lance-table` manifest and transaction modules | Stored records, validation, conversion, three typed operations and the optional publication release field, manifest construction, and compatibility rules. |
| `lance-table::io::commit` and external commit handlers | Add the default-unsupported retention resolver and explicit supported-handler implementations. Share it across pin inspection, mutation, confirmation, and maintenance. |
| Dataset API and `io/commit` | Rebasable registration, typed confirmation, per-attempt expiration, optional atomic release with publication, authoritative resolution, and durable evidence. |
| `dataset/cleanup.rs` | Retain every currently listed pin until committed removal, validate the version window/evidence, retain coordinator records, and explain retention. No expiration commit. |
| `dataset/optimize.rs`, FRI modules, conflict resolver | Record unindexed required movements, validate all rewrite/trim paths, and combine consumer requirements. |
| `dataset/refs.rs`, restore and clone paths | Independent membership, sticky capability, and normal/force deletion checks. |

Keep pin acquisition unavailable until all maintenance paths and the acceptance tests are complete. The format PR's unsupported capability and unknown-operation handling remain the safe intermediate state. Before activation, require pin-aware maintenance clients, exclude incompatible destructive clients, and finish or cancel incompatible operations already in flight.

SRID integration depends on [#10441](https://github.com/lance-format/lance/pull/10441) and [#9960](https://github.com/lance-format/lance/pull/9960), both open when this draft was prepared. Build on their final landed behavior; do not duplicate their implementation or rely on debug-only support. Unsupported recording refuses pinned compaction with an actionable error.

The production consumer protects staged output separately and validates publication. It may set `release_pin_id` on its final publication transaction; the resulting version must retain the output's files and required history. This is explicit, so intermediate job commits keep their pin unless it expires. Standalone release remains available after separately confirmed publication or cancellation.

A failed publication that definitely did not commit leaves the pin untouched. After an uncertain response, recover the publication by its own transaction UUID; an absent pin is not proof of publication. Persist the publication transaction even when no pins remain afterward. This does not extend retention after the pin ends; if later cleanup removes the evidence, recovery may be indeterminate. Ordinary conflict, retry, and publication checks remain required when the target pin is already absent. Keep this in the existing transaction format and do not wait for Transaction V2.

## Review and handoff

The initial PR contains only this design, named tests, and performance expectations. Agree on it before implementation and leave it unmerged. The agent then adds the implementation to the same PR. The author thoroughly reviews and owns that code, with independent AI review as additional evidence. The design reviewer performs the final check against the agreed behavior and verification.

Record minor implementation deviations for final review. Bring material behavior or scope changes back for agreement before continuing. Keep the format-vote PR separate.

Before calling this design approved:

- Review the concrete builder, request, handle, outcome, and inspection types within the agreed builder/transaction API style.
- Agree on the concrete backend matrix in [tests.md](tests.md#backend-coverage).
- Record the workload sizes and numerical performance limits in [performance.md](performance.md#approval-and-regression-gates) after baseline measurements.
- Reconcile and publish the agreed review changes in the format PR and design discussion after local review. Keep format-review follow-ups visible; local agreement does not replace maintainer review and the vote.

Final review evidence should map each named case to its actual test and result, include benchmark comparisons and retained-byte accounting, and list deviations and activation requirements.
