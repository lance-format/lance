# Liquid Clustering

Liquid clustering records a desired physical row layout and the provenance of
fragments produced by clustering rewrites. It does not change the logical rows
in a table. Readers may ignore this metadata, but writers must preserve it so
future maintenance can distinguish current layouts from stale or unclustered
fragments.

## Clustering Key

The ordered clustering key is stored on schema fields with
`Field.unenforced_clustering_key_position`. Positions are one-based, unique,
and contiguous. A clustering key contains between one and four top-level
fields. Nested and system fields are not valid clustering fields.

`CLUSTERING_ALGORITHM_TYPED_QUANTILE_RANK_V1` supports Boolean, signed and
unsigned integer, Float32, Float64, Utf8, LargeUtf8, Utf8View, Date32, Date64,
Timestamp, and Decimal128 fields. Null values are supported and sort after all
non-null values.

Metadata updates must not change a non-empty clustering key's field IDs or
order. An operation that replaces the schema with an incompatible key must
disable clustering and clear layout stamps from newly produced or changed
fragments. A compatible rename preserves the key because field IDs, positions,
and data types remain unchanged.

## Table State

`Manifest.liquid_clustering` contains the following fields:

- `enabled` declares whether clustering maintenance is active. It does not
  require ordinary appends to arrive pre-clustered.
- `generation` identifies the desired layout and must be greater than zero.
- `algorithm` identifies how rows in that generation are ordered.

The absence of `liquid_clustering` means there is no table-level clustering
declaration. Disabling clustering, including as the result of an incompatible
schema replacement, retains the previous generation and algorithm.

Enabling clustering for the first time, re-enabling it, or changing its
algorithm creates a generation greater than every generation in the current
table state and fragments. A generation must never decrease or be reused for a
different declaration. Repeating an identical declaration is idempotent.

Readers must preserve an unrecognized non-zero `algorithm` value when carrying
the state into a new manifest. An operation that must interpret an unrecognized
algorithm, including clustering new rows, must reject it as unsupported.

## Fragment Layout Stamps

`DataFragment.clustering_generation` identifies the layout generation applied
when the fragment was produced. Zero means the fragment is unstamped and must
not be assumed to satisfy the current clustering declaration. A non-zero value
may refer to an earlier generation; this is how maintenance identifies stale
fragments after the declaration changes.

`DataFragment.clustering_group_id` is an optional, non-nil UUID shared by every
output fragment from one atomic clustering rewrite group. A group ID may appear
only on fragments with a non-zero `clustering_generation`, and all fragments
with the same group ID must have the same generation. A generation without a
group ID is valid.

Writers must preserve the table state and stamps of retained fragments whose
data files, overlays, row count, and clustering-key schema are unchanged. A
rewrite that does not apply the declared clustering algorithm must not copy a
source fragment's stamp to newly produced fragments. A successful clustering
rewrite stamps every output fragment with the declaration's generation and the
rewrite group's ID.

## Typed Quantile Rank V1

`CLUSTERING_ALGORITHM_TYPED_QUANTILE_RANK_V1` maps each clustering field to a
16-bit empirical-quantile coordinate and orders the resulting points by a
Hilbert curve. Every rewrite group in one clustering operation uses the same
model so its output fragments share one coordinate space.

The model population is every live row in every source fragment selected by
the clustering operation, as visible at the table version on which the
operation is based. Rows hidden by deletions and rows in fragments outside the
operation are excluded. Partial samples must cover disjoint subsets whose union
is exactly this population. If a partition is retried, the coordinator uses
exactly one successful result for that partition.

For each clustering field, construct the model as follows:

1. Order non-null values ascending according to their Arrow logical type.
   Booleans order false before true; integers, dates, timestamps, and decimals
   use numeric order; strings use unsigned UTF-8 byte order; and floating-point
   values use the IEEE 754 `totalOrder` predicate. Positive and negative zero
   are distinct under that predicate.
   Encode each value into bytes with the same unsigned lexicographic order.
   A Boolean or fixed-width number starts with `0x01`. Its payload is big-endian;
   signed integers, dates, timestamps, and Decimal128 also have the most
   significant bit inverted. For Float32 and Float64, invert every bit when the
   sign bit is set, and otherwise invert only the sign bit. An empty string is
   `0x01`. A non-empty string starts with `0x02`: split its first 32 UTF-8 bytes
   into 8-byte blocks and any remaining bytes into 32-byte blocks, zero-pad the
   last block, and append `0xff` after each non-final block and the unpadded
   length after the final block. The encoded-value width below includes these
   marker, padding, and block-length bytes.
2. A non-null field value is a sampling candidate; a null field value is not.
   Associate each candidate with its physical row address in the operation's
   table version: `(fragment_id << 32) | row_offset`.
   `column_index` is the zero-based position of the field in the ordered
   clustering key, not its schema position or field ID. Set `x` to
   `row_address XOR (column_index * 0x9e3779b97f4a7c15) XOR
   0x4c414e43455f514e`. Then set `x` to
   `(x XOR (x >> 30)) * 0xbf58476d1ce4e5b9`, set `x` to
   `(x XOR (x >> 27)) * 0x94d049bb133111eb`, and finally set `x` to
   `x XOR (x >> 31)`. Multiplication wraps modulo 2^64 and shifts are logical.
3. Let `W` be the maximum encoded-value width among all candidates for the
   field, including candidates that are not ultimately retained. Charge every
   retained entry `W + 40` bytes and set
   `K = min(65,536, floor((8 * 1024 * 1024) / (W + 40)))`. If the field has no
   candidates, use `W = 0` and an empty sample. If it has candidates and `K` is
   zero, the rewrite is invalid. Retain the `K` candidates with the smallest
   `(x, row_address)` pairs, or all candidates when there are fewer than `K`.
4. A partial sample records its local `W` and the smallest `K` candidates under
   that local `W`. To merge partial samples, take the maximum of their `W`
   values, recompute `K`, form the union of their retained candidates, and keep
   the smallest `K` `(x, row_address)` pairs. Apply the same rule at every merge
   level. Because the merged `K` is no greater than either input's `K`, the
   result is independent of worker partitioning, merge order, and retries.
5. Sort the retained encoded values in ascending order. Duplicate values remain
   in the sample sequence.

As a sampling conformance vector, use clustering-key position zero and the
following non-null Utf8 values: addresses 62, 50, 17, 6, 24, and 12 contain
`a`, `b`, `c`, `d`, `e`, and `f`, respectively; address 35 contains the ASCII
byte `g` repeated 1,626,848 times. Partition them as `A = {62, 50}`,
`B = {17, 6}`, and `C = {24, 12, 35}`. The value at address 35 has an encoded
width of 1,677,691 bytes, so the merged `W + 40` is 1,677,731 bytes and `K` is
four. In ascending `(x, row_address)` order, the addresses and `x` values are
`62: 0x00190210937e3060`, `17: 0x023d008d552fce12`,
`24: 0x038e84229007c90d`, `12: 0x060599852005aa22`,
`50: 0x063d664f158828df`, `35: 0x1320075888da18fe`, and
`6: 0x1707ae364d30b31e`. A whole-input sample and every partition order and merge
tree, including `(A merge B) merge C` and `A merge (B merge C)`, retain
addresses `{62, 17, 24, 12}` in that priority order, then sort their encoded
values as `a`, `c`, `e`, `f`. Address 35 establishes `W` even though its value
is not retained.

For a non-null input value, let `upper` be the number of samples less than or
equal to its encoded value and let `rank = max(upper - 1, 0)`. With `n` retained
samples, its coordinate is zero when `n <= 1`; otherwise it is
`floor(rank * 65534 / (n - 1))`. A null value has coordinate 65535.

Apply the Hilbert transpose transform to the array of coordinates, where `N`
is the number of clustering fields and all operations use unsigned 16-bit
coordinate values:

1. Starting with `q = 32768` and repeatedly halving `q` while `q > 1`, set
   `p = q - 1` and visit coordinates from 0 through `N - 1`. If coordinate
   `i` has bit `q` set, XOR coordinate 0 with `p`. Otherwise, set `swap` to
   `(coordinate[0] XOR coordinate[i]) AND p`, then XOR both coordinate 0 and
   coordinate `i` with `swap`.
2. For each `i` from 1 through `N - 1`, XOR coordinate `i` with coordinate
   `i - 1`.
3. Set `prefix = 0`. Again start with `q = 32768` and repeatedly halve `q`
   while `q > 1`. Whenever coordinate `N - 1` has bit `q` set, XOR `prefix`
   with `q - 1`.
4. XOR every coordinate with `prefix`.

Interleave the transformed coordinates from bit 15 through bit 0 and, within
each bit, from clustering field 0 through field `N - 1`. Encode the resulting
`16 * N` bits as `2 * N` big-endian bytes. Sort rows lexicographically by this
byte string in ascending order. Rows with equal byte strings have no specified
relative order.

## Feature Flag

A manifest that contains `liquid_clustering` or any non-zero fragment
`clustering_generation` sets `FLAG_CLUSTERING_METADATA` (`1 << 11`) in
`writer_feature_flags`. It does not set the corresponding reader flag because
the metadata affects only physical layout and maintenance decisions.

A writer that does not recognize this flag must refuse to modify the table. A
reader may scan the table without interpreting the clustering metadata because
the stamps do not change logical row values.
