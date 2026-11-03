---
title: Getting Started with Lance
description: Choose the right entry point in the Lance stack — LanceDB, Python/Rust/Java SDKs, or SQL and compute engines
---

# Getting Started with Lance

Lance is an open lakehouse and columnar table format for multimodal AI. Every tool in the Lance ecosystem reads and writes the same on-disk format, so you can start at the layer that matches your application and interoperate across layers without copying or converting data.

## The Lance Stack

```text
+---------------------------------------------------------------------+
| 1. Database                LanceDB (Embedded & Distributed)         |
+---------------------------------------------------------------------+
| 2. Language SDKs           Python (PyLance), Rust, Java             |
+---------------------------------------------------------------------+
| 3. Query & Compute Engines Spark, DuckDB, Trino, DataFusion, Ray    |
+=====================================================================+
|                    Lance Open Format & Catalog Spec                 |
|        (Columnar Files, ACID Versioning, Blob v2, Indexes)          |
+---------------------------------------------------------------------+
```

## Choose Your Entry Point

### 1. Database — LanceDB

Choose **LanceDB** when you are building an AI application, RAG pipeline, or search service and want table management, vector search, BM25 full-text search, and hybrid reranking out of the box.

- **[LanceDB Quickstart](lancedb.md)** — Create a table, run vector and keyword search, update rows, and time travel in 5 minutes.
- **[LanceDB Documentation ↗](https://docs.lancedb.com/)** — Embeddings, hybrid reranking, TypeScript/Python/Rust database clients, and Cloud/Enterprise deployments.

### 2. Language SDKs — Python, Rust, and Java

Choose a **Language SDK** when you want direct, low-level control over Lance datasets, fragments, custom Arrow schemas, multimodal [Blob v2 columns](../guide/blob.md), or [distributed write transactions](../guide/distributed_write.md).

| Language | Package | Quickstart | API Reference |
|---|---|---|---|
| **Python** | `pylance` | **[Python (PyLance) Quickstart](pylance.md)** · [Versioning](versioning.md) · [Vector Search](vector-search.md) · [Full-Text Search](full-text-search.md) | [Python API Docs ↗](https://lance-format.github.io/lance-python-doc) |
| **Rust** | `lance` | **[Rust Quickstart](rust.md)** | [docs.rs/lance ↗](https://docs.rs/lance/latest/lance) |
| **Java** | `org.lance:lance-core` | **[Java Quickstart](java.md)** | [Javadoc ↗](https://www.javadoc.io/doc/org.lance/lance-core/latest/index.html) |

### 3. Query & Compute Engines

Choose a **Query or Compute Engine** when you want to run distributed SQL, lakehouse ETL, analytics, or large-scale ML training over Lance tables alongside existing data pipelines.

| Engine | Typical Workload | Get Started |
|---|---|---|
| **Apache Spark** | Distributed SQL, DDL/DML, and batch ETL with Spark SQL and PySpark | **[Spark Quickstart](spark.md)** · [Spark Connector Docs](../integrations/spark/index.md) |
| **DuckDB / Trino / DataFusion** | Interactive SQL analytics and federated queries over Lance datasets | [DataFusion](../integrations/datafusion.md) · [All SQL Integrations](../integrations/index.md#query-engines-and-sql) |
| **Ray / PyTorch / Flink** | Distributed data processing, streaming ingestion, and ML training loaders | [PyTorch](../integrations/pytorch.md) · [All ML & Stream Integrations](../integrations/index.md#machine-learning-and-ai) |

## One Format Across Every Layer

Because all layers share the [Lance format specification](../format/index.md), you can mix tools freely in a single pipeline — for example, ingest and backfill features at scale with **Apache Spark**, build vector and full-text indexes with the **Python or Rust SDK**, and serve low-latency hybrid queries from **LanceDB**.
