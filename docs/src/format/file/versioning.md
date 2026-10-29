# Versioning

The Lance file format has a single version number for both the overall file format and the encoding strategy. The
major number is changed when the file format itself is modified while the minor number is changed when only the encoding
strategy is modified. Newer versions will typically have better performance and compression but may not be readable
by older versions of Lance.

Any version explicitly labeled unstable, including the current 2.3 format and the `next` alias, should not be used for
production use cases. Unstable formats have no compatibility guarantee: breaking encoding changes may make files
written by one Lance build unreadable by later builds. They should only be used for experimentation and benchmarking
upcoming features.

The `stable` and `next` aliases are resolved by the specific Lance release you are using. During a format rollout
(for example, 2.3), prefer explicit version pinning for deterministic behavior across environments.

The following values are supported:

| Version        | Minimal Lance Version | Maximum Lance Version | Description |
| -------------- | --------------------- | --------------------- | ----------- |
| 0.1            | Any                   | 0.34 (write)          | This is the initial Lance format. It is no longer writable. |
| 2.0            | 0.16.0                | Any                   | Rework of the Lance file format that removed row groups and introduced null support for lists, fixed size lists, and primitives |
| 2.1            | 0.38.1                | Any                   | Enhances integer and string compression, adds support for nulls in struct fields, and improves random access performance with nested fields. |
| 2.2            | None                  | Any                   | Adds support for newer nested type/encoding capabilities (including map support) and 2.2-era storage features. |
| 2.3 (unstable) | None                  | Unspecified           | Adds sparse structural pages and other experimental encodings. |
| legacy         | N/A                   | N/A                   | Alias for 0.1 |
| stable         | N/A                   | N/A                   | Alias for the default version for new datasets in the Lance release you are running. |
| next           | N/A                   | N/A                   | Alias for the latest unstable version in the Lance release you are running.|

## Compatibility Caveats

Stable formats carry a compatibility guarantee, but certain data patterns exposed encoder bugs
that required encoding changes to fix.  Files containing those patterns written by the fixed
encoder are not readable by readers predating the fix.  The affected scenarios are listed here
so operators running mixed-version deployments know the minimum reader version required.

### FixedSizeList with all-null inner values (Lance 11.1.0)

**Affected format**: 2.1 and later.

**Scenario**: A `FixedSizeList` column where every inner value (not the outer list item itself)
is null — for example, `FixedSizeList<nullable Float32, dim=4>` where all eight Float32 values
across two outer rows are null.

**Buggy writer (Lance < 11.1.0)**: The encoder wrote `bits_per_value=0` into the FullZip page
layout.  Readers of any version rejected these pages with an error, so the data was unreadable
regardless of reader version.

**Fixed writer (Lance ≥ 11.1.0)**: The encoder stores per-row validity bytes for the null inner
values, producing `bits_per_value > 0`.  The fixed reader (Lance ≥ 11.1.0) can also decode the
old buggy pages, so old files written before 11.1.0 become readable after upgrading.

**Forward compatibility**: Files containing this pattern written by Lance ≥ 11.1.0 are **not
readable by Lance < 11.1.0**.  Old readers encounter the `Compression::Constant` inner encoding
in the FSL descriptor and panic rather than returning an error.

**Minimum reader version for new files**: Lance 11.1.0.
