# `zonemap_signed_nan`

A dataset written by the ZoneMap writer that predates the `negative_nan_count`
zone statistic (`main` at `fbec02448`). Its three float columns (`f16`, `f32`,
`f64`) hold the same ten eight-row zones, one per NaN/null shape: ordinary
values with signed zeros and infinities, negative NaN with ordinary values,
positive NaN with ordinary values, both signs, only positive NaN, only
negative NaN, only null, null with negative NaN, null with positive NaN, and
ordinary values with null. Each column has a seeded ZoneMap index
(`rows_per_zone = 8`), and a second identical fragment appended after the
indices were built carries a write seed in its data file.

That writer records any NaN as `max = +NaN` and never records the sign of the
NaNs, so this fixture is how tests prove that a reader which tracks
`negative_nan_count` still reads old indices and old seeds correctly, and that
the old reader sees byte-identical statistics from the new writer.

## Regenerating

```bash
cargo run --release \
    --manifest-path test_data/v14.0.0-beta.10/zonemap_signed_nan/datagen/Cargo.toml -- \
    test_data/v14.0.0-beta.10/zonemap_signed_nan/dataset.lance
```

The generator is a standalone crate pinned to the pre-`negative_nan_count`
writer; do not bump the pin to a build that writes the new column.
