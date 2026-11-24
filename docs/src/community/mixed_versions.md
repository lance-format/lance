# Guidelines for Mixed Version Support

Lance ships as a library, and one table is often read and written by several
processes that upgrade independently. These could be separate users or services.
They are rarely on the same release at the same moment, and during a rollout they
are guaranteed not to be. A format change that makes newly written data unreadable
by the rest of the deployment breaks the deployment, not just the component that
upgraded.

These guidelines describe how such a change is rolled out so that a mixed
version deployment keeps working.

## The Compatibility Window

**A reader and a writer whose releases are at most three months apart should
interoperate.** Anything a writer produces with default settings should be
readable by any reader from the preceding three months.

Three months is roughly six stable releases at the usual two-week cadence (see
[Guidelines for Releases](./release.md)). The window is measured between stable
release dates; betas and release candidates are not reference points for it.

The window bounds forward compatibility only. Backwards compatibility has no
expiry: a current reader reads data written by any older writer, for as long as
the format that produced it is marked stable. The three-month rule constrains
the other direction — how soon a new writer may start producing something that
readers already in the field cannot read.

## Breaking Format Changes Ship Opt-In First

A change is a **breaking format change** when data written with it cannot be
read by the previous stable release. The common cases are:

- A new file format version or a new encoding in an existing one.
- A new index format version, or a new on-disk layout for an existing index.

Such a change ships **opt-in, default off, for at least three months** after the
first stable release that can *read* it. The clock starts at the release that
can read the new form, not at the release that first offered to write it, and it
is counted in stable releases — a reader capability that exists only in a beta
has not started the clock.

During that period:

- The new form is produced only on explicit request: an index or write
  parameter, or an environment variable for a deployment-wide rollout.
  `LANCE_FTS_FORMAT_VERSION` is an example of the latter.
- No default changes to produce it, in the Rust core or in any binding. A
  binding default that diverges from the core default is the same violation.
- Maintenance operations preserve the existing form rather than upgrading it.
  Compaction, index optimize, incremental indexing, mem-wal flush, clone,
  restore, and branch creation must not raise a table's reader requirements. A
  table should not become unreadable to half the deployment because a background
  job ran.
- The opt-in is documented together with the earliest release in which it may
  become the default, so the deadline is visible to operators and to whoever
  later proposes the flip.

After the window closes, the change may become the default: every reader still
inside the compatibility window can read it.

## Flipping the Default

Making a previously opt-in format the default is a breaking change and is
released as one:

- Label the PR `breaking-change` so the release automation bumps the major
  version (see
  [Breaking Change Detection](https://github.com/lance-format/lance/blob/main/release_process.md#breaking-change-detection)).
- Add a [migration guide](../guide/migration.md) entry that records the minimum
  reader version for data written with the new default, and how to keep
  producing the old form.
- Keep the explicit opt-out working. An operator who discovers a lagging reader
  after upgrading must be able to go back to the old form by configuration,
  without rewriting data that has already been written.

Removing the ability to *write* an old form is a separate, later step from
making the new one the default, and removing the ability to *read* one is
governed by the format's own compatibility contract rather than by this window.

## How Changes Are Fenced

The mechanisms that make a change opt-in already exist in each format surface;
a new change should use the one that matches it rather than inventing another.

| Surface | Fence | How a user opts in |
| ------- | ----- | ------------------ |
| [File format](../format/file/versioning.md) | The data file version number. Older readers reject a version they do not know. | `data_storage_version` on the table, pinned to an exact version rather than the `stable` or `next` alias. |
| [Index formats](../format/index/index.md#index-compatibility) | `IndexMetadata.version`. A reader that does not support the version skips the segment and falls back to scanning the covered fragments. | An index creation parameter, or an environment variable for a rollout-wide default. |

Pair the fence with a test in `python/python/tests/compat` so the contract is
checked against released versions rather than asserted in prose.

## Exceptions

- **Unstable formats.** Versions marked unstable, such as file format 2.3 and
  the `next` alias, carry no compatibility guarantee and are outside the window.
  They are also never the default; promoting one to stable is what starts its
  clock.
- **Data that nothing can read.** When an older writer produced data that no
  reader can decode, fixing the writer does not have a three-month window to
  observe — there is no working deployment to preserve. Document the new minimum
  reader version under
  [Compatibility Caveats](../format/file/versioning.md#compatibility-caveats).
- **Tables that already opted in.** Once a table has opted in, later writes keep
  using the new form. The opt-in, not each individual write, is the point at
  which the operator accepted the new reader requirement.
- **Experimental APIs.** Surfaces documented as experimental may change without
  this window, as they already do for
  [breaking change detection](https://github.com/lance-format/lance/blob/main/release_process.md#breaking-change-detection).

An exception is a decision to record, not to assume: state it in the PR
description, so the reviewer evaluates the exception rather than reconstructing
it.
