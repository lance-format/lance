---
title: Integrations
description: Connect Lance to query engines, ML frameworks, databases, and catalogs across the data ecosystem.
---

# Integrations

Because Lance is built on Apache Arrow and exposes
a stable Rust core with Python and Java bindings, it can be consumed from query engines, ML frameworks, databases,
and catalog systems without copying or converting data.

This page lists the integrations maintained by the Lance community. Some live in this repository as first-party
Rust/Python APIs; others are maintained as dedicated subprojects under the [lance-format](https://github.com/lance-format)
GitHub organization.

## Query Engines and SQL

| Integration | Description | Source |
|---|---|---|
| [Apache DataFusion](datafusion.md) | Register Lance datasets as DataFusion tables and run SQL with predicate/projection pushdown. Available in both Rust and Python. | Built-in |
| [Apache Spark](spark/index.md) | Read and write Lance datasets from Spark SQL and DataFrames. | [lance-format/lance-spark](https://github.com/lance-format/lance-spark) |
| [DuckDB](duckdb) | Query Lance datasets directly from DuckDB. | [lance-format/lance-duckdb](https://github.com/lance-format/lance-duckdb) |
| [Trino](trino) | Federate Lance into Trino alongside other connectors. | [lance-format/lance-trino](https://github.com/lance-format/lance-trino) |

## Machine Learning and AI

| Integration | Description | Source |
|---|---|---|
| [PyTorch](pytorch.md) | Use `lance.torch.data.LanceDataset` as a `torch.utils.data.IterableDataset` for training and inference. | Built-in |
| [TensorFlow](tensorflow.md) | Use `lance_tensorflow.from_lance` to stream Lance data into `tf.data.Dataset` pipelines. | [lance-format/lance-tensorflow](https://github.com/lance-format/lance-tensorflow) |
| [Ray](ray) | Distributed read/write of Lance datasets with Ray Data. | [lance-format/lance-ray](https://github.com/lance-format/lance-ray) |
| [Hugging Face](huggingface) | Convert and load Hugging Face datasets to and from Lance in a single call. | [lance-format/lance-huggingface](https://github.com/lance-format/lance-huggingface) |

## Databases and Stream Processing

| Integration | Description | Source |
|---|---|---|
| PostgreSQL | Read Lance datasets from PostgreSQL via the `pglance` extension. | [lance-format/pglance](https://github.com/lance-format/pglance) |
| Apache Flink | Stream data into Lance from Flink jobs. | [lance-format/lance-flink](https://github.com/lance-format/lance-flink) |

## Catalogs and Namespaces

| Integration | Description | Source |
|---|---|---|
| [Lance Namespace](../format/namespace) | Specification and codegen SDKs (Rust, Python, Java) for catalog-backed Lance tables. | [lance-format/lance-namespace](https://github.com/lance-format/lance-namespace) |
| [Catalog implementations](../format/namespace/supported-catalogs) | Reference implementations for Apache Hive, Apache Polaris, Apache Gravitino, Unity Catalog, AWS Glue, and others. | [lance-format/lance-namespace-impls](https://github.com/lance-format/lance-namespace-impls) |

## Other Ecosystem Projects

| Integration | Description | Source |
|---|---|---|
| Lance Graph | Cypher-capable graph query engine on top of Lance. | [lance-format/lance-graph](https://github.com/lance-format/lance-graph) |
| Lance Data Viewer | Read-only web interface for browsing Lance datasets. | [lance-format/lance-data-viewer](https://github.com/lance-format/lance-data-viewer) |
| Lance Context | Manage multimodal agentic context lifecycle with Lance. | [lance-format/lance-context](https://github.com/lance-format/lance-context) |

!!! note "Stability"

    Subprojects in the [lance-format](https://github.com/lance-format) organization graduate from incubating
    status once they meet the project's quality bar (CI, tests, established use cases, community adoption).
    Incubating subprojects may have changing APIs — check the project README for the current status. See
    [Community Governance](../community#projects) for details on how integrations are organized.

## Additional Integrations

If there's an integration you'd like to see, but isn't listed above, please
[open an issue](https://github.com/lance-format/lance/issues/new) describing the use case. PRs are always
welcome, though it's recommended to alert the maintainers to avoid duplicated work.
