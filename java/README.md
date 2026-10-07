# Java bindings and SDK for Lance

<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

**The Open Lakehouse Format for Multimodal AI**
</p></div>

Lance is an open lakehouse format for multimodal AI. It contains a file format, table format, and catalog spec that allows you to build a complete lakehouse on top of object storage to power your AI workflows.

The key features of Lance include:

* **Expressive hybrid search:** Combine vector similarity search, full-text search (BM25), and SQL analytics on the same dataset with accelerated secondary indices.

* **Lightning-fast random access:** 100x faster than Parquet or Iceberg for random access without sacrificing scan performance.

* **Native multimodal data support:** Store images, videos, audio, text, and embeddings in a single unified format with efficient blob encoding and lazy loading.

* **Data evolution:** Efficiently add columns with backfilled values without full table rewrites, perfect for ML feature engineering.

* **Zero-copy versioning:** ACID transactions, time travel, and automatic versioning without needing extra infrastructure.

* **Rich ecosystem integrations:** Apache Arrow, Pandas, Polars, DuckDB, Apache Spark, Ray, Trino, Apache Flink, and open catalogs (Apache Polaris, Unity Catalog, Apache Gravitino).

For more details, see the full [Lance format specification](https://lance.org/format).

## Quick start

Introduce the Lance SDK Java Maven dependency(It is recommended to choose the latest version.):

```shell
<dependency>
    <groupId>org.lance</groupId>
    <artifactId>lance-core</artifactId>
    <version>0.35.0</version>
</dependency>
```

### Basic I/O

* create empty dataset

```java
void createDataset() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("write_stream").toString();
    Schema schema =
            new Schema(
                    Arrays.asList(
                            Field.nullable("id", new ArrowType.Int(32, true)),
                            Field.nullable("name", new ArrowType.Utf8())),
                    null);
    try (BufferAllocator allocator = new RootAllocator();) {
        Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build());
        try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build());) {
            dataset.version();
            dataset.latestVersion();
        }
    }
}
```

* create and write a Lance dataset

```java
void createAndWriteDataset() throws IOException, URISyntaxException {
    Path path = "";     // the original source path
    String datasetPath = "";    // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator();
        ArrowFileReader reader =
            new ArrowFileReader(
                new SeekableReadChannel(
                    new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))), allocator);
        ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, arrowStream);
        try (Dataset dataset =
                     Dataset.create(
                             allocator,
                             arrowStream,
                             datasetPath,
                             new WriteParams.Builder()
                                     .withMaxRowsPerFile(10)
                                     .withMaxRowsPerGroup(20)
                                     .withMode(WriteParams.WriteMode.CREATE)
                                     .withStorageOptions(new HashMap<>())
                                     .build())) {
            // access dataset
        }
    }
}
```
* read dataset

```java
void readDataset() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            dataset.countRows();
            dataset.getSchema();
            dataset.version();
            dataset.latestVersion();
            // access more information
        }
    }
}
```

* drop dataset

```java
void dropDataset() {
    String datasetPath = tempDir.resolve("drop_stream").toString();
    Dataset.drop(datasetPath, new HashMap<>());
}
```

### Random Access

```java
void randomAccess() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            List<Long> indices = Arrays.asList(1L, 4L);
            List<String> columns = Arrays.asList("id", "name");
            try (ArrowReader reader = dataset.take(indices, columns)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot result = reader.getVectorSchemaRoot();
                    result.getRowCount();

                    for (int i = 0; i < indices.size(); i++) {
                        result.getVector("id").getObject(i);
                        result.getVector("name").getObject(i);
                    }
                }
            }
        }
    }
}
```

### Schema evolution

* add columns

```java
void addColumnsByExpressions() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            SqlExpressions sqlExpressions = new SqlExpressions.Builder().withExpression("double_id", "id * 2").build();
            dataset.addColumns(sqlExpressions, Optional.empty());
        }
    }
}

void addColumnsBySchema() {
  String datasetPath = ""; // specify a path point to a dataset
  try (BufferAllocator allocator = new RootAllocator()) {
    try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
      SqlExpressions sqlExpressions = new SqlExpressions.Builder().withExpression("double_id", "id * 2").build();
      dataset.addColumns(new Schema(
          Arrays.asList(
              Field.nullable("id", new ArrowType.Int(32, true)),
              Field.nullable("name", new ArrowType.Utf8()),
              Field.nullable("age", new ArrowType.Int(32, true)))), Optional.empty());
    }
  }
}
```

* alter columns

```java
void alterColumns() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            ColumnAlteration nameColumnAlteration =
                    new ColumnAlteration.Builder("name")
                            .rename("new_name")
                            .nullable(true)
                            .castTo(new ArrowType.Utf8())
                            .build();

            dataset.alterColumns(Collections.singletonList(nameColumnAlteration));
        }
    }
}
```

* drop columns

```java
void dropColumns() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            dataset.dropColumns(Collections.singletonList("name"));
        }
    }
}
```

## JVM Engine Connectors

JVM engine connectors can be built using the Lance Java SDK. Here are some connectors maintained in lancedb Github organization:

* [Spark Lance connector](https://github.com/lancedb/lance-spark)
* [Flink Lance connector](https://github.com/lancedb/lance-flink)
* [Trino Lance connector](https://github.com/lancedb/lance-trino)

## Contributing

From the codebase dimension, the lance project is a multiple-lang project. All Java-related code is located in the `java` directory.
And the whole `java` dir is a standard maven project can be imported into any IDEs support java project.

Standard Build (Java + JNI)

```shell
mvn clean package
```
This command executes the base Maven build process to compile all Java code in the `java` directory and generate the JNI native library.

Java-Only Build: 

```shell
mvn clean package -Dskip.build.jni=true
 ```
This will skip the JNI code compilation step and only process the Java module. Useful when focusing on Java feature development without needing native libraries, reducing build time.

Product Release Build:

```shell
mvn clean package -Drust.release.build=true
```
This will enable product environment optimization configurations (e.g., code shrinking, debug symbol removal, performance tuning) to generate deployment packages suitable for production environments. The optimized package is smaller in size and runs more efficiently.

If you only want to build rust code(`lance-jni`), you can run the following command:

```shell
cd lance-jni && cargo build
```

The java module uses `spotless` maven plugin to format the code and check the license header. 
And it is applied in the `validate` phase automatically.

### Environment(IDE) setup

Firstly, clone the repository into your local machine:

```shell
git clone https://github.com/lancedb/lance.git
```

Then, import the `java` directory into your favorite IDEs, such as IntelliJ IDEA, Eclipse, etc.

Due to the java module depends on the features provided by rust module. So, you also need to make sure you have installed rust in your local.

To install rust, please refer to the [official documentation](https://www.rust-lang.org/tools/install).

And you also need to install the rust plugin for your IDE.

Then, you can build the whole java module:

```shell
mvn clean package
```

Running these commands, it builds the rust jni binding codes automatically.
