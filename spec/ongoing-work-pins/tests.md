# Ongoing-work pins: acceptance tests

These are proposed test names and expected outcomes for the [implementation design](README.md), not completed tests. Add focused cases to existing Rust suites. Use small multi-fragment fixtures, injected clocks and failures, and barriers for races rather than timing-dependent sleeps. Split parameter combinations so individual unit tests remain lightweight.

## Backend coverage

Both direct storage and external coordination are in scope. This is the proposed concrete matrix for design approval:

| Backend | Required coverage |
| --- | --- |
| Local filesystem with `RenameCommitHandler` | Lifecycle, retention, branch operations, and worker integration. |
| `ConditionalPutCommitHandler` | The same contract with an instrumented store; paginated unordered listings, stale hints, conditional-write conflicts, and ambiguous responses. Add an S3 integration run before advertising that deployment. |
| `ExternalManifestCommitHandler` | Contract tests with a controllable external store: authoritative heads differing from raw listings, staged/finalized paths, delayed visibility, lost responses, and discovery-record retention. |
| DynamoDB external manifest store | Run the external-store cases through its adapter, including its consistency and conditional-publication behavior. Record the integration environment. |
| Lance namespace external manifest store | Run through its adapter and a contract test namespace, including named branches and version-list/discovery retention. Verify the actual service implementation before advertising it; a passing fake is not evidence about a remote service's guarantees. |

Exercise existing V1 and V2 manifest naming for direct storage. Test main and named branches where the handler supports them. Unsafe or custom handlers without the required guarantees reject pin acquisition. Keep unsupported-backend rejection separate from backend failures after a possibly successful commit, which may be indeterminate.

## Registration and lifecycle

| Test | Expected outcome |
| --- | --- |
| `registration_builder_creates_reusable_request` | Builder configuration performs no storage I/O. Building resolves the target and handler support, generates the transaction UUID, and fixes timestamps without publishing or fixing the job's source. Cover the 24-hour default and setters. Saving/restoring preserves the request and branch identity; an indeterminate outcome keeps the same request. Building again creates a distinct request. |
| `register_confirms_committed_version` | Register at V+1 and return source V+1. Confirm current membership, source-manifest readability/version, and branch identity. Missing transaction contents on predecessor V alone do not prevent registration; the new registration commit preserves its own evidence. |
| `register_rebases_before_work_begins` | Advance the head with append, compaction, restore, or overwrite before publication. Rebase with the same ID and times, and pin the successful registration version. The worker plans/reads only that returned version. Reject a different table or changed named-branch identifier; the legacy and whole-table recreation limits still apply. |
| `pin_requests_use_existing_branch_identity` | Compare the complete stored identifier. A newly created replacement branch with a different identifier rejects the old request/handle. Missing legacy identifiers and the all-zero placeholder use the existing deterministic fallback; repeated reads and unrelated metadata changes preserve it. Identical legacy identity inputs after recreation remain indistinguishable, so callers must not reuse requests/handles across that lifecycle. |
| `register_recovers_after_lost_response` | Drop a successful response, carry the pin forward with another write, and retry the saved request. Recover the same pin without a commit or resetting its current expiration/label. Repeat after process restart. |
| `registration_confirmation_is_required` | A changed branch reference, absent membership, or unreadable source/head cannot return Confirmed. Distinguish definite rejection from an unresolved possibly committed request; retain the original request for the latter. |
| `head_resolution_handles_gaps_and_delayed_writes` | Reclaimed slots, stale hints, unordered pages, and a disappearing selected head cannot hide a newer authoritative attached head. Exclude staging and detached files. |
| `external_head_and_discovery_remain_authoritative` | Exercise each external backend's resolution/finalization protocol and preserve discovery records for the window. Do not select a raw-listing head that disagrees with the coordinator. An uncertain coordinator result cannot trigger raw-listing fallback. Where the backend permits fallback, exercise the strict listing path with stale hints and gaps. |
| `retention_resolution_requires_handler_support` | A custom handler inheriting the default retention resolver and the unsafe handler both reject acquisition before submission. Supported implementations satisfy the same head-resolution contract. Keep ordinary reader and carry-only write resolution unchanged. An unresolved confirmation read after possible publication remains indeterminate. |
| `registration_contention_is_bounded` | Competing appends and renewals cause bounded retries using the same ID/times. Only the committed source becomes fixed. Reject an expired fresh request. Retain recovery state for uncertain attempts; never report a possibly committed request as definitely rejected. |
| `two_jobs_release_independently` | Register jobs on successive current versions sharing files. Releasing one ID preserves the other. Two fresh registrations against one unchanged version are not promised because registration advances the head. |
| `renew_preserves_source_and_extends_only` | With a deadline still in the future, change only expiration; equal/earlier requests are no-ops. Competing renewals cannot shorten it. Reject an expired target even for an otherwise satisfied request, and recheck after conflicts. |
| `renewal_and_expiration_follow_commit_order` | Race renewal against an ordinary append that removes expired records. A future extension winning first survives the append's retry; removal first makes renewal fail. Repeat with a prevalidated renewal stalled in storage across the deadline. |
| `ordinary_commit_rechecks_expiration` | Mix expired, unexpired, renewed, and explicitly released IDs. Sample time and use the actual predecessor on each attempt. Preserve remaining fields; removed records never reappear from a stale attempt. Publication of an old reclaimed slot does not authorize cleanup of current pins. |
| `no_op_pin_operations_do_not_commit` | Recovered registration, satisfied unexpired renewal, and absent release create no version, including when unrelated expired records remain. Real mutations count normally. |
| `pin_validation_and_compatibility` | Validate required UUID presence and byte lengths 0, 15, 16, and 17; duplicate 16-byte IDs; timestamp bounds/nanoseconds; checked arithmetic; and nonempty lists missing the capability. Convert the registration's textual UUID to the same 16-byte value. Unsupported writers reject; readers need no pin-only reader capability. |
| `publication_and_release_are_atomic` | Publish an index or column replacement with `release_pin_id`. The result and named removal appear in one version; no extra release commit. Failure without publication leaves both unapplied. Validate normal publication conflicts before removal and preserve unrelated unexpired pins. A missing release target does not skip or authorize the data operation. |
| `publication_release_recovers_lost_response` | Lose the response after publication releases the last pin. Recover from the stored publication transaction without publishing twice. Cover an inline transaction and one too large to inline; with sidecar writes disabled, the latter must fail before publication. Repeat with an already-absent release target. An absent pin alone must not report success; if later cleanup removes the evidence, the unresolved outcome remains indeterminate. |
| `publication_release_rejects_invalid_context` | Reject a malformed release UUID, detached submission, Clone, or any pin operation carrying `release_pin_id`. A valid combined transaction uses a publication UUID distinct from the referenced pin. |

## Version and transaction retention

| Test | Expected outcome |
| --- | --- |
| `cleanup_preview_retains_listed_expired_pins` | Preview and execution retain expired records still present in the head. Report them as awaiting committed removal, not immediately reclaimable. Neither path creates an expiration commit. |
| `busy_table_expiration_needs_no_extra_commit` | A normal write removes expired pins in its one new version. Cleanup resolves that head and reclaims eligible resources without a separate expiration version. |
| `idle_table_retains_until_write_or_release` | Pass the deadline with no new commits and run cleanup: pinned resources and blocking branch membership remain. Explicit release creates a version, allowing eligible cleanup and branch deletion. Repeat with a normal write providing the removal instead. |
| `cleanup_retains_publication_history_window` | After registration at V and commits through H, every attached version V..H remains discoverable and fully readable. Transactions after V remain available. Include intermediate-only data, indices, overlays, deletion files, mappings, and coordinator records; exclude detached versions. |
| `pin_commits_preserve_transaction_evidence` | Register, write, renew, restore, and overwrite with pins, and publish with `release_pin_id` when no pins remain afterward. Each result has durable transaction contents inline or in a referenced sidecar. Cover transactions too large to inline and disabled-sidecar configuration; reject publication that would preserve neither representation. |
| `cleanup_retains_sources_and_mapping_references` | Read the original source after an allowed overlay fold. Retain maps added after registration, maps after FRI metadata replacement, and maps referenced only by an old retained entry. Verify actual reads/translation, not just filenames. |
| `source_integrity_failure_stops_affected_maintenance` | Missing, unreadable, or wrong-version source manifests stop affected cleanup/trim/rewrite. Missing intermediate manifests or required transactions stop affected cleanup. Never substitute empty coverage. Explicit release by an unambiguous ID still works. |
| `cleanup_reclaims_after_last_requirement_ends` | Multiple pins can keep one oldest boundary. Removing the last oldest pin advances it; removing all ends it. Other tags, versions, and consumers still retain their resources. Historical lists do not revive ended pins. Past expiration alone changes nothing. |
| `cleanup_explains_unique_retention` | Report the active boundary and deduplicated bytes by file class. Shared files are not charged once per pin; distinguish overlapping protection from incremental pin retention. |

## Compaction, FRI, and overlays

| Test | Expected outcome |
| --- | --- |
| `unindexed_compaction_requires_deferred_remapping` | A pin on an unindexed table makes required eager-remap work refuse before rewriting. Supported deferred compaction records movements even with no index coverage or existing FRI entry. |
| `compaction_rechecks_pins_at_commit` | Plan before a pin, then register and attempt the stale result. Cover entry-free, FRI version 0, FRI version 1, bare/no-FRI, and eager-remapped results. On the first attempt and rebases, preserve required mappings or reject/replan before any early return. |
| `fri_trimming_preserves_requirement_union` | Both FRI version 0 and version 1 trim paths retain pin plus existing-consumer requirements. Pause trim planning, register a pin, add required history, and retry the stale result. A consumer-check error cannot drop pin-required history. Unknown requirements refuse the trim. |
| `pin_fragment_roots_follow_source_layout` | Pin a multi-fragment source, then move its rows into new fragments. Derive roots from the source version's complete fragment metadata, not only the current head or index coverage. Reject acquisition for an unsupported layout before publishing a pin. Where a fragment-tree layout is supported, resolve its referenced metadata rather than treating the empty inline list as an empty source; missing metadata stops affected maintenance. Obtain IDs without scanning row data, and retain movements for fragments the job has not read yet. |
| `legacy_pin_record_restamps_on_retry` | Newly recorded legacy history is stamped with successful publishing version minus one; existing stamps stay unchanged. Check the inclusive source boundary. |
| `fri_v1_history_releases_completed_job_chain` | For A → B → C → D, jobs at B and C share C → D. Ending the B job can release B → C when no other consumer needs it. Include converging chains, expiration, and physical files still retained by historical versions. |
| `fri_v1_history_retains_whole_legacy_records` | In a version 1 history containing legacy records, selecting one legacy group retains the whole record and every group's downstream dependencies. An unrelated record is not retained for a pin merely because its version stamp is newer. |
| `pin_preserves_fri_v1_overlay_restrictions` | FRI version 1 rejects new overlays, even on unrelated fragments. Overlaid sources cannot record required version 1 mappings or bypass the check with a bare rewrite. Releasing a pin alone does not remove the version 1 entry. Preserve existing FRI version 0 behavior. |
| `unsupported_srid_recording_refuses_pinned_compaction` | Exercise release-equivalent unsupported recording through the injectable support check, then the supported state with the SRID dependencies. Require an actionable error; debug-only success is insufficient. |

## Other writes and end-to-end behavior

| Test | Expected outcome |
| --- | --- |
| `ordinary_writes_preserve_remaining_pins` | Representative writes preserve the attempt's current predecessor fields except explicit or expired removals. Start before registration, renewal, release, or another commit's expiration; force a slot conflict and verify the rebuilt list. Removed pins must not reappear. Unknown operations fail safely. |
| `ordinary_stale_slot_write_does_not_change_current_pins` | With an old cached writer, a stale hint, and reclaimed slots below the current head, attempt an ordinary commit into an old hole. Whether it refuses or fills the historical slot, authoritative membership and the sticky bit stay at the real head. Historical copies cannot revive pins. Any automatic cleanup triggered by that write resolves the real head and preserves its complete window. Include supported external-handler contract tests. |
| `ordinary_writes_add_no_pin_only_head_listing` | Instrument never-flagged, flagged-empty, active, and expired-pin tables. Carrying/pruning adds no complete-history listing beyond the existing commit protocol. Keep stricter resolution for explicit pin APIs, publication with release, restore/overwrite, rewrite/trim checks, and cleanup. |
| `restore_and_overwrite_preserve_remaining_membership` | Start from current pins, apply normal expiration/explicit release, and preserve the remaining list and sticky bit. Never import historical membership. Keep surviving pins' full version windows even when translation paths are discarded. |
| `clone_and_new_branch_start_without_pins` | New lists are empty; the inherited bit is sticky. A child's pin protects parent-owned files through parent cascade cleanup. A separate shallow clone still needs source protection. |
| `detached_version_has_empty_pin_list` | Create a detached version from a source with pins: its list is empty and the inherited capability remains set. The branch's membership is unchanged. All three pin operations and publication with release reject detached submission. Repeat with a flagged-empty source. |
| `branch_deletion_checks_normal_and_force_paths` | Reject active pins, including expired-but-not-removed records. Definitively no attached manifest permits the pin check to pass; listing/read/validation failures do not. Re-resolve a disappearing head. Do not implicitly expire records. |
| `orphan_deletion_requires_confirmed_absence` | With no branch reference, an empty complete direct-storage listing permits the pin check to pass. For external storage, require both reliable coordinator absence and an empty complete object listing. Refuse deletion on stale or uncertain absence, a coordinator record with a missing manifest, either source reporting a manifest that cannot be resolved and validated, partial listings, access errors, or timeouts. A disappearing selected head requires another resolution attempt, not an absence shortcut. Run against each supported adapter; apply the same safety check on normal and force paths. |
| `worker_survives_compaction_and_cleanup` | A small worker confirms registration, reads the source, survives deferred compaction and ordinary cleanup, and translates through required history. After release, otherwise-unreferenced resources become reclaimable. |

Keep tests of the known physical-deletion races as diagnostics of existing limitations, not passing safety promises.

## Production consumer follow-up

Before integrating the first async-index consumer, add staged-index publication tests after appends, deferred compaction, non-indexed-column replacement, and ordinary pin-aware cleanup. Compare indexed results to scans. Restore and indexed-value replacement must reject or withdraw stale coverage as appropriate. Cover crash recovery, separately protected staged output, atomic publication/release, and separately confirmed publication before standalone release.

The local retention experiment supports this follow-up but does not replace pin API, concurrency, remote-backend, or large-scale acceptance tests.
