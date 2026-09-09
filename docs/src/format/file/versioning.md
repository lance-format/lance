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
