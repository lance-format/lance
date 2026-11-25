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
