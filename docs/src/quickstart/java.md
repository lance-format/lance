---
title: Java Quickstart
description: Get started with the Lance Java SDK — write Arrow batches, scan with SQL filters, fetch rows by index, and time travel
---

# Getting Started with Java

The `org.lance:lance-core` artifact is the low-level Java SDK for the Lance format, providing JNI and Apache Arrow C Data Interface bindings to the Rust core. If you are running analytics or data pipelines on the JVM, prefer a query engine such as [Apache Spark](spark.md) or [Trino](../integrations/trino/index.md); use `lance-core` directly when building custom engine connectors or storage integrations.

## Add Dependencies

Add `org.lance:lance-core` and the Apache Arrow vector/allocator dependencies to your build file (JDK 11 or later is supported):

=== "Maven"

    ```xml
    --8<-- "snippets/quickstart/java-maven-deps.xml"
    ```

=== "Gradle (Kotlin)"

    ```kotlin
    --8<-- "snippets/quickstart/java-gradle-deps.kts"
    ```

!!! note "JDK 17+ JVM Flag for Apache Arrow"
    On JDK 17 and newer, Apache Arrow requires reflective access to `java.nio` direct buffers. Pass `--add-opens=java.base/java.nio=ALL-UNNAMED` when launching the JVM.

## Write, Scan, and Time Travel

Create `Quickstart.java` to write an Arrow batch to a Lance dataset, scan rows with a SQL filter, perform random row access with `take`, and check out a specific dataset version:

```java
--8<-- "snippets/quickstart/Quickstart.java"
```

Run the example with `--add-opens=java.base/java.nio=ALL-UNNAMED`:

```output
--8<-- "snippets/quickstart/outputs/Quickstart.java.run.txt"
```

## Next Steps

- Browse the [Java API Reference (Javadoc) ↗](https://www.javadoc.io/doc/org.lance/lance-core/latest/index.html) for complete class and builder documentation.
- See the [Apache Spark Quickstart](spark.md) for distributed SQL and DataFrame operations built on the Lance Java core.
- Read the [User Guide](../guide/read_and_write.md) for core Lance concepts including indexing, schema evolution, and storage configuration.
