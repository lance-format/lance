# lance-jsonb

JSONB encoding, parsing, and JSONPath evaluation for Lance JSON columns.

This crate is adapted from [databendlabs/jsonb](https://github.com/databendlabs/jsonb)
at commit `fba895c5ebe77ce2539e187f9c652f51cbf195c3` (after release 0.5.6), which is
licensed under Apache-2.0. Lance owns this copy so that it can evolve the encoding
and the read paths together with the Lance file format.

## Differences from upstream

- Numbers are parsed without upstream's `arbitrary_precision` feature, matching the
  configuration Lance used before vendoring: integers outside the `i64`/`u64` range
  and decimals become `f64`. Decimal values that are already encoded are still
  decoded.
- Only the Databend binary layout is kept; the `sqlite` backend and the `jaq`
  integration are removed.
- The public API is limited to what Lance uses: parsing, `RawJsonb` accessors and
  conversions, serde deserialization through `from_raw_jsonb`, and JSONPath
  selection. Upstream's JSON manipulation functions, key paths, and the serde
  `Serializer` are removed.
- JSONPath arithmetic results and filter literals are encoded through the same
  encoder as parsed values, so there is a single writer of JSONB bytes.

## Compatibility

The JSONB bytes are persisted in Lance data files. `tests/it/compat.rs` pins the
bytes that jsonb 0.5.6 produced for representative inputs and the decoded text of
values that Lance does not write but may still read. Any change that alters these
bytes must remain readable by released Lance versions.
