---
title: Java Quickstart
description: Get started with the Lance Java SDK — write Arrow batches, scan with SQL filters, fetch rows by index, and time travel
---

# Getting Started with Java

The `org.lance:lance-core` artifact provides Java bindings to the Lance Rust core via JNI and Apache Arrow's C Data Interface. It powers JVM compute connectors (such as [Apache Spark](spark.md), Trino, and Flink) and can be embedded directly in Java and Scala services.

## Add Dependencies

Add `org.lance:lance-core` and the Apache Arrow vector/allocator dependencies to your build file (JDK 11 or later is supported):

=== "Maven"

    ```xml
    <dependencies>
      <dependency>
        <groupId>org.lance</groupId>
        <artifactId>lance-core</artifactId>
        <version>14.0.2</version>
      </dependency>
      <dependency>
        <groupId>org.apache.arrow</groupId>
        <artifactId>arrow-vector</artifactId>
        <version>18.3.0</version>
      </dependency>
      <dependency>
        <groupId>org.apache.arrow</groupId>
        <artifactId>arrow-memory-netty</artifactId>
        <version>18.3.0</version>
      </dependency>
    </dependencies>
    ```

=== "Gradle (Kotlin)"

    ```kotlin
    dependencies {
        implementation("org.lance:lance-core:14.0.2")
        implementation("org.apache.arrow:arrow-vector:18.3.0")
        implementation("org.apache.arrow:arrow-memory-netty:18.3.0")
    }
    ```

!!! note "JDK 17+ JVM Flag for Apache Arrow"
    On JDK 17 and newer, Apache Arrow requires reflective access to `java.nio` direct buffers. Pass `--add-opens=java.base/java.nio=ALL-UNNAMED` when launching the JVM.

## Write, Scan, and Time Travel

Create `Quickstart.java` to write an Arrow batch to a Lance dataset, scan rows with a SQL filter, perform random row access with `take`, and check out a specific dataset version:

```java
package example;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.lance.Dataset;
import org.lance.WriteParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

public class Quickstart {
  private static byte[] buildArrowStream(BufferAllocator allocator, Schema schema)
      throws Exception {
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector ids = (IntVector) root.getVector("id");
      VarCharVector names = (VarCharVector) root.getVector("name");
      Float4Vector prices = (Float4Vector) root.getVector("price");

      int[] idValues = {1, 2, 3};
      String[] nameValues = {"red running shoes", "trail running shoes", "rain jacket"};
      float[] priceValues = {89.0f, 120.0f, 99.0f};

      for (int i = 0; i < idValues.length; i++) {
        ids.setSafe(i, idValues[i]);
        names.setSafe(i, nameValues[i].getBytes(StandardCharsets.UTF_8));
        prices.setSafe(i, priceValues[i]);
      }
      root.setRowCount(idValues.length);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      }
      return out.toByteArray();
    }
  }

  public static void main(String[] args) throws Exception {
    String uri = "/tmp/lance-java-quickstart.lance";
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.notNullable("name", new ArrowType.Utf8()),
                Field.notNullable(
                    "price", new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE))));

    try (BufferAllocator allocator = new RootAllocator()) {
      // 1. Create and write a Lance dataset
      byte[] ipcBytes = buildArrowStream(allocator, schema);
      try (ArrowReader inputReader =
              new ArrowStreamReader(new ByteArrayInputStream(ipcBytes), allocator);
          Dataset dataset =
              Dataset.write()
                  .allocator(allocator)
                  .reader(inputReader)
                  .uri(uri)
                  .mode(WriteParams.WriteMode.OVERWRITE)
                  .execute()) {
        System.out.println(
            "Created dataset version "
                + dataset.version()
                + " with "
                + dataset.countRows()
                + " rows");

        // 2. Scan with a SQL filter and column projection
        ScanOptions options =
            new ScanOptions.Builder()
                .filter("price < 100.0")
                .columns(Arrays.asList("id", "name", "price"))
                .build();
        try (LanceScanner scanner = dataset.newScan(options);
            ArrowReader scanReader = scanner.scanBatches()) {
          while (scanReader.loadNextBatch()) {
            VectorSchemaRoot batch = scanReader.getVectorSchemaRoot();
            System.out.println("Filtered batch rows: " + batch.getRowCount());
          }
        }

        // 3. Random access by row index
        try (ArrowReader takeReader =
            dataset.take(List.of(0L, 2L), Arrays.asList("id", "name"))) {
          while (takeReader.loadNextBatch()) {
            VectorSchemaRoot batch = takeReader.getVectorSchemaRoot();
            for (int i = 0; i < batch.getRowCount(); i++) {
              System.out.println(
                  batch.getVector("id").getObject(i) + ": " + batch.getVector("name").getObject(i));
            }
          }
        }

        // 4. Time travel to a specific version
        try (Dataset v1 = dataset.checkoutVersion(1L)) {
          System.out.println("Checked out version " + v1.version() + " rows: " + v1.countRows());
        }
      }
    }
  }
}
```

Run the example with `--add-opens=java.base/java.nio=ALL-UNNAMED`:

```output
Created dataset version 1 with 3 rows
Filtered batch rows: 2
1: red running shoes
3: rain jacket
Checked out version 1 rows: 3
```

## Next Steps

- Browse the [Java API Reference (Javadoc) ↗](https://www.javadoc.io/doc/org.lance/lance-core/latest/index.html) for complete class and builder documentation.
- See the [Apache Spark Quickstart](spark.md) for distributed SQL and DataFrame operations built on the Lance Java core.
- Read the [User Guide](../guide/read_and_write.md) for core Lance concepts including indexing, schema evolution, and storage configuration.
