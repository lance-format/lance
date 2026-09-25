# Clustering Providers

Clustering describes a physical layout, not different logical table contents.
The format identifies the provider and configuration used to organize fragments;
it does not prescribe an algorithm, boundaries, or maintenance schedule.

The definitions are `Clustering` and `FragmentClustering` in `table.proto`.
This feature is reserved but unsupported until its writer preservation and
invalidation rules are implemented.

## Table Configuration

`Manifest.clustering` holds the current configuration:

- `columns`: an ordered, non-empty list of distinct, non-negative schema field
  IDs that exist in the current schema. These identify all columns on which the
  layout depends.
- `provider`: a non-empty, case-sensitive implementation name. Namespaced names
  are recommended to avoid collisions.
- `version`: a non-zero `uint64` identifying a provider-defined layout or
  configuration, not a dataset version or software release.
- `provider_metadata`: a `map<string, string>`, like `Manifest.config`. An empty
  map means no provider-specific configuration. The provider defines the keys,
  value encoding, schema versioning, and compatibility. Structured values may be
  serialized as JSON strings; Lance does not interpret or normalize them.

Absence of table configuration disables new clustering work but does not clear
existing fragment markers. Appends need not be clustered even when a
configuration is present.

The pair `(provider, version)` must not be reassigned to a different layout
definition within a table. Providers allocate versions, including on divergent
branches. No numerical ordering or compatibility is implied by the version.
Changing the layout definition selects a new version; auxiliary history or
maintenance information in the payload may evolve without redefining that layout.

Lance does not maintain a declaration list or resolve historical versions.
Providers manage any history they require in their own metadata. Switching the
current provider or version does not relabel or rewrite old fragments. A marker
may name an older version or another provider. If its configuration is unavailable
or unsupported, it must not be interpreted using the current configuration.

Provider metadata is inline. Paths in metadata values acquire no file-retention
or clone semantics; external artifacts require a separate lifecycle contract.
Providers must tolerate ordinary writers replacing fragments or clearing markers
without updating opaque bookkeeping. Such bookkeeping is not authoritative table state.

When `Manifest.clustering` is present, its `columns` are the source of clustering
keys. Writers must clear `Field.unenforced_clustering_key_position` and the older
`Field.unenforced_clustering_key` hints rather than maintain a second copy. When
configuration is absent, existing hints retain their prior meaning and do not
imply fragment provenance.

## Fragment Markers

`DataFragment.clustering` contains only `provider` and `version`, with the same
validity rules as the table configuration. Absence makes no clustering claim.
A marker identifies the layout under which the fragment was organized; it does
not prove sort order, disjoint ranges, size, or query coverage. Fragments carry
no opaque payload or arbitrary key-value map.

Only a writer that understands the relevant configuration may assign its marker.
The layout `version` already identifies a configuration generation; there is no
separate generation field. Rewrite groups, buckets, and their fragment membership
are provider-specific concepts, not public fragment fields. Providers that need
them manage their records in table-level `provider_metadata`; no common keys or
value schemas are prescribed. These records must be revalidated against live
fragments and their markers before use, since ordinary writers do not maintain
opaque provider bookkeeping. A marker may remain when the current table
configuration is absent or different.

By default, clustering maintenance selects unstamped fragments or fragments
belonging to the same provider, or an implementation it explicitly supersedes.
Other providers' fragments require an explicit user takeover. Sharing a provider
name does not make all its versions interpretable by every implementation.
An unsupported configuration must not be executed or silently substituted.

These rules govern clustering maintenance, not ordinary table writes. Missing a
provider must not by itself prohibit append, update, delete, or compaction.

## Writer Requirements

Writers supporting the common feature preserve unknown provider names, versions,
and all metadata keys and string values on unrelated commits. Not understanding
a provider does not authorize changing its configuration. It also does not require
stopping ordinary writes: the generic rules below suffice.

- Append preserves existing metadata. New fragments are unstamped unless a
  provider verifies their layout.
- An unchanged fragment retains its marker. Deletion-only changes may retain it:
  layout claims must remain valid for a subset of rows at unchanged offsets,
  without promising current size or clustering quality.
- A new fragment from a generic split, merge, compaction, or row rewrite does
  not inherit a marker. A provider-aware writer may verify and assign one.
- In-place column replacement or overlays changing a clustering column clear
  the affected fragment's marker. If the marker matches the current table pair,
  its columns identify those dependencies. For any other pair, a generic writer
  cannot infer historical dependencies and clears the marker on any value change.
  A provider-aware writer may retain it only after verifying layout validity.
- A rename retaining field IDs and types preserves markers. Dropping a field,
  changing its type, or changing a containing nested type clears affected
  current-layout markers and removes the table configuration if its columns are
  affected. For historical markers with unknown dependencies, such schema changes
  conservatively clear the markers. Missing a provider must not block the change.
- Overwrite discards replaced fragments' markers. It may retain table
  configuration only if its field IDs and types remain compatible; otherwise it
  removes that configuration. Historical columns must not be rebound by name.

Clearing a fragment marker alone does not remove table configuration or modify
its opaque payload. Explicit configuration changes and schema invalidation above
are distinct from unrelated commits that must preserve the configuration.

Each commit must publish valid configuration and markers atomically and revalidate
them against concurrent changes. The provider, version, columns, and metadata
form one configuration; the map shape does not permit generic per-key merging of
concurrent configuration changes. A rewrite may publish unstamped output and mark
it in a later commit only after validating that the output has not changed.
Restore restores the selected snapshot's configuration and markers together.
Clone or import must preserve layout identity to retain markers; conflicting
definitions for the same pair must be rejected or the imported markers cleared.

## Readers and Statistics

Readers may ignore clustering metadata and scan normally. They must not prune
data or assume ordering based on a marker alone. Index coverage, row-address
translation, and deletions keep their own contracts and feature requirements.

Statistics producers and consumers must share value-comparison semantics,
including strings, nulls, NaNs, and typed bounds. Provider normalization cannot
silently change these semantics. Custom statistics require a reader or index
plugin that understands them; otherwise that optimization must not be used.
Missing or incomplete statistics cannot justify skipping data. No particular
statistics format or index is required by this feature.

## Compatibility

A manifest containing table configuration or any fragment marker must set
`FLAG_CLUSTERING_METADATA` (`1 << 12`) in `writer_feature_flags`, not in
`reader_feature_flags`. This single bit protects the common metadata container
and its maintenance rules. Older writers that could discard the new fields
must refuse to write. It may be cleared only when configuration and all markers
are absent.

A writer supporting the common feature can preserve an unknown provider and
perform ordinary writes using the rules above. Requests to execute unsupported
provider configurations are rejected without prohibiting other operations.

Adding a provider, changing layout versions, or evolving a provider's payload
does not allocate another Lance feature bit. Providers do not own global reader
or writer bits. Layout versions identify configurations, not an ordered
compatibility threshold; interpretation belongs to the provider. Payload schema
compatibility can be expressed through provider-defined metadata keys and values.

A provider cannot use opaque metadata to impose extra correctness requirements
on ordinary readers or writers. Behavior that cannot safely be ignored,
preserved, or invalidated under this contract requires a separate table feature,
not just a provider version bump.

## Relationship to Delta

Delta's [clustered-table protocol](https://github.com/delta-io/delta/blob/ef3e91b509365cdc2b23c1afdbbab3b4d3828d63/PROTOCOL.md#clustered-table)
uses shared writer features rather than a feature for each clustering provider.
Its [domain metadata contract](https://github.com/delta-io/delta/blob/ef3e91b509365cdc2b23c1afdbbab3b4d3828d63/PROTOCOL.md#domain-metadata)
requires preservation of unknown domains, while additional correctness obligations
require feature negotiation. This proposal follows that separation and provider
ownership principle without adopting Delta's JSON actions or general file tags.
