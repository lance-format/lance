# Data Types

Lance uses [Apache Arrow](https://arrow.apache.org/) as its in-memory data format. This guide covers the supported data types with a focus on array types, which are essential for vector embeddings and machine learning applications.

## Arrow Type System

Lance supports the full Apache Arrow type system. When writing data through Python (PyArrow) or Rust (arrow-rs), the Arrow types are automatically mapped to Lance's internal representation.

### Primitive Types

| Arrow Type | Description | Example Use Case |
|------------|-------------|------------------|
| `Boolean` | True/false values | Flags, filters |
| `Int8`, `Int16`, `Int32`, `Int64` | Signed integers | IDs, counts |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | Unsigned integers | IDs, indices |
| `Float16`, `Float32`, `Float64` | Floating point numbers | Measurements, scores |
| `Decimal128`, `Decimal256` | Fixed-precision decimals | Financial data |
| `Date32`, `Date64` | Date values | Birth dates, event dates |
| `Time32`, `Time64` | Time values | Time of day |
| `Timestamp` | Date and time with timezone | Event timestamps |
| `Duration` | Time duration | Elapsed time |

### String and Binary Types

| Arrow Type | Description | Example Use Case |
|------------|-------------|------------------|
| `Utf8` | Variable-length UTF-8 string | Text, names |
| `LargeUtf8` | Large UTF-8 string (64-bit offsets) | Large documents |
| `Binary` | Variable-length binary data | Raw bytes |
| `LargeBinary` | Large binary data (64-bit offsets) | Large blobs |
| `FixedSizeBinary(n)` | Fixed-length binary data | UUIDs, hashes |

### Blob Type for Large Binary Objects

Lance provides a specialized **Blob** type for efficiently storing and retrieving very large binary objects such as videos, images, audio files, or other multimedia content. Blob columns support planned full-payload reads as well as lazy file-like access for streaming, seeking, and partial reads.

For new datasets, use blob v2 (`lance.blob.v2`) via `blob_field` and `blob_array`.

Blob versioning follows dataset file format rules:

- `data_storage_version` is the Lance file format version of a dataset.
- A dataset's `data_storage_version` is fixed once created.
- For `data_storage_version >= 2.2`, legacy blob metadata (`lance-encoding:blob`) is rejected on write.
- Legacy metadata-based blob write remains available for `0.1`, `2.0`, and `2.1`.

```python
import lance
import pyarrow as pa
from lance import blob_array, blob_field

schema = pa.schema([
    pa.field("id", pa.int64()),
    blob_field("video"),
])

table = pa.table(
    {
        "id": [1],
        "video": blob_array([b"sample-video-bytes"]),
    },
    schema=schema,
)

ds = lance.write_dataset(table, "./videos_v22.lance", data_storage_version="2.2")
_row_address, payload = ds.read_blobs("video", indices=[0])[0]
```

For legacy compatibility (`data_storage_version <= 2.1`), you can still write blob columns using `LargeBinary` with `lance-encoding:blob=true`.

To create a blob column with the legacy path, add the `lance-encoding:blob` metadata to a `LargeBinary` field:

```python
import pyarrow as pa
import lance

# Define schema with a blob column for videos
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("filename", pa.utf8()),
    pa.field("video", pa.large_binary(), metadata={"lance-encoding:blob": "true"}),
])

# Read video file
with open("sample_video.mp4", "rb") as f:
    video_data = f.read()

# Create and write dataset
table = pa.table({
    "id": [1],
    "filename": ["sample_video.mp4"],
    "video": [video_data],
}, schema=schema)

ds = lance.write_dataset(
    table,
    "./videos_legacy.lance",
    schema=schema,
    data_storage_version="2.1",
)
```

To read complete blob payloads into memory, use `read_blobs()`:

```python
rows = ds.read_blobs("video", indices=[0])
_row_address, payload = rows[0]
```

Use `take_blobs()` only when you need file-like objects for lazy, partial, or seek-based reading:

```python
# Retrieve blob as a file-like object (lazy loading)
blobs = ds.take_blobs("video", indices=[0])

# Use with libraries that accept file-like objects
import av  # pip install av
with av.open(blobs[0]) as container:
    for frame in container.decode(video=0):
        # Process video frames without loading entire video into memory
        pass
```

For more details, see the [Blob API Guide](blob.md).

## Array Types for Vector Embeddings

Lance provides excellent support for array types, which are critical for storing vector embeddings in AI/ML applications.

### FixedSizeList - The Preferred Type for Vector Embeddings

`FixedSizeList` is the recommended type for storing fixed-dimensional vector embeddings. Each vector has the same number of dimensions, making it highly efficient for storage and computation.

=== "Python"

    ```python
    import lance
    import pyarrow as pa
    import numpy as np

    # Create a schema with a vector embedding column
    # This defines a 128-dimensional float32 vector
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), 128)),  # FixedSizeList of 128 floats
    ])

    # Create sample data with embeddings
    num_rows = 1000
    vectors = np.random.rand(num_rows, 128).astype(np.float32)

    table = pa.Table.from_pydict({
        "id": list(range(num_rows)),
        "text": [f"document_{i}" for i in range(num_rows)],
        "vector": [v.tolist() for v in vectors],
    }, schema=schema)

    # Write to Lance format
    ds = lance.write_dataset(table, "./embeddings.lance")
    print(f"Created dataset with {ds.count_rows()} rows")
    ```

=== "Rust"

    ```rust
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use lance::dataset::WriteParams;
    use lance::Dataset;
    use std::sync::Arc;

    #[tokio::main]
    async fn main() -> lance::Result<()> {
        // Define schema with a 128-dimensional vector column
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    128,
                ),
                false,
            ),
        ]));

        // Create sample data
        let ids = Int64Array::from(vec![0, 1, 2]);
        let texts = StringArray::from(vec!["doc_0", "doc_1", "doc_2"]);
        
        // Create vector embeddings (128-dimensional)
        let values: Vec<f32> = (0..384).map(|i| i as f32 / 100.0).collect();
        let values_array = Float32Array::from(values);
        let vectors = FixedSizeListArray::try_new_from_values(values_array, 128)?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids) as ArrayRef,
                Arc::new(texts) as ArrayRef,
                Arc::new(vectors) as ArrayRef,
            ],
        )?;

        // Write to Lance
        let dataset = Dataset::write(
            vec![batch].into_iter().map(Ok),
            "embeddings.lance",
            WriteParams::default(),
        )
        .await?;

        println!("Created dataset with {} rows", dataset.count_rows().await?);
        Ok(())
    }
    ```

### Vector Search with Embeddings

Once you have vector embeddings stored in Lance, you can perform efficient vector similarity search:

```python
import lance
import numpy as np

# Open the dataset
ds = lance.dataset("./embeddings.lance")

# Create a query vector (same dimension as stored vectors)
query_vector = np.random.rand(128).astype(np.float32).tolist()

# Perform vector search - find 10 nearest neighbors
results = ds.to_table(
    nearest={
        "column": "vector",
        "q": query_vector,
        "k": 10,
    }
)
print(results.to_pandas())
```

For production workloads with large datasets, create a vector index for much faster search:

```python
# Create an IVF-PQ index for fast approximate nearest neighbor search
ds.create_index(
    "vector",
    index_type="IVF_PQ",
    num_partitions=256,  # Number of IVF partitions
    num_sub_vectors=16,  # Number of PQ sub-vectors
)

# Search with the index (automatically used)
results = ds.to_table(
    nearest={
        "column": "vector",
        "q": query_vector,
        "k": 10,
        "nprobes": 20,  # Number of partitions to search
    }
)
```

### List and LargeList - Variable-Length Arrays

For variable-length arrays where each row may have a different number of elements, use `List` or `LargeList`:

```python
import lance
import pyarrow as pa

# Schema with variable-length arrays
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("tags", pa.list_(pa.utf8())),      # Variable number of string tags
    pa.field("scores", pa.list_(pa.float32())), # Variable number of float scores
])

table = pa.Table.from_pydict({
    "id": [1, 2, 3],
    "tags": [["python", "ml"], ["rust"], ["data", "analytics", "ai"]],
    "scores": [[0.9, 0.8], [0.95], [0.7, 0.85, 0.9]],
}, schema=schema)

ds = lance.write_dataset(table, "./variable_arrays.lance")
```

## Nested and Complex Types

### Struct Types

Store structured data with multiple named fields:

```python
import lance
import pyarrow as pa

# Schema with nested struct
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("metadata", pa.struct([
        pa.field("source", pa.utf8()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("embedding_model", pa.utf8()),
    ])),
    pa.field("vector", pa.list_(pa.float32(), 384)),  # 384-dim embedding
])

table = pa.Table.from_pydict({
    "id": [1, 2],
    "metadata": [
        {"source": "web", "timestamp": "2024-01-15T10:30:00", "embedding_model": "text-embedding-3-small"},
        {"source": "api", "timestamp": "2024-01-15T11:45:00", "embedding_model": "text-embedding-3-small"},
    ],
    "vector": [
        [0.1] * 384,
        [0.2] * 384,
    ],
}, schema=schema)

ds = lance.write_dataset(table, "./with_metadata.lance")
```

### Map Types

Store key-value pairs with dynamic keys:
Map writes require Lance file format version 2.2 or later.

```python
import lance
import pyarrow as pa

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("attributes", pa.map_(pa.utf8(), pa.utf8())),
])

table = pa.Table.from_pydict({
    "id": [1, 2],
    "attributes": [
        [("color", "red"), ("size", "large")],
        [("color", "blue"), ("material", "cotton")],
    ],
}, schema=schema)

ds = lance.write_dataset(table, "./with_maps.lance", data_storage_version="2.2")
```

## Data Type Mapping for Integrations

When integrating Lance with other systems (like Apache Flink, Spark, or Presto), the following type mappings apply:

| External Type | Lance/Arrow Type | Notes |
|--------------|------------------|-------|
| `BOOLEAN` | `Boolean` | |
| `TINYINT` | `Int8` | |
| `SMALLINT` | `Int16` | |
| `INT` / `INTEGER` | `Int32` | |
| `BIGINT` | `Int64` | |
| `FLOAT` | `Float32` | |
| `DOUBLE` | `Float64` | |
| `DECIMAL(p,s)` | `Decimal128(p,s)` | |
| `STRING` / `VARCHAR` | `Utf8` | |
| `CHAR(n)` | `Utf8` | Fixed-width in source system; stored as variable-length Utf8 |
| `DATE` | `Date32` | |
| `TIME` | `Time64` | Microsecond precision |
| `TIMESTAMP` | `Timestamp` | |
| `TIMESTAMP WITH LOCAL TIMEZONE` | `Timestamp` | With timezone info |
| `BINARY` / `VARBINARY` | `Binary` | |
| `BYTES` | `Binary` | |
| `BLOB` | Blob v2 extension type (`lance.blob.v2`) | Use `blob_field` / `blob_array` for new datasets; legacy metadata path applies to `data_storage_version <= 2.1` |
| `ARRAY<T>` | `List(T)` | Variable-length array |
| `ARRAY<T>(n)` | `FixedSizeList(T, n)` | Fixed-length array (vectors) |
| `ROW` / `STRUCT` | `Struct` | Nested structure |
| `MAP<K,V>` | `Map(K, V)` | Key-value pairs |

### Vector Embeddings in Integrations

For vector embedding columns, use `ARRAY<FLOAT>(n)` or `ARRAY<DOUBLE>(n)` where `n` is the embedding dimension:

```sql
-- Example: Creating a table with vector embeddings in SQL-compatible systems
CREATE TABLE embeddings (
    id BIGINT,
    text STRING,
    vector ARRAY<FLOAT>(384)  -- 384-dimensional vector
);
```

This maps to Lance's `FixedSizeList(Float32, 384)` type, which is optimized for:

- Efficient columnar storage
- SIMD-accelerated distance computations
- Vector index creation and search

## Best Practices for Vector Data

1. **Use FixedSizeList for embeddings**: Always use `FixedSizeList` (not variable-length `List`) for vector embeddings to enable efficient storage and indexing.

2. **Choose appropriate precision**: 
   - `Float32` is the standard choice, balancing precision and storage
   - `Float16` or `BFloat16` can reduce storage by 50% with minimal accuracy loss
   - `Int8` for quantized embeddings

3. **Align dimensions for SIMD**: Vector dimensions divisible by 8 enable optimal SIMD acceleration. Common dimensions: 128, 256, 384, 512, 768, 1024, 1536.

4. **Create indices for large datasets**: For datasets with more than ~10,000 vectors, create an ANN index for fast search:

    ```python
    # IVF_PQ is recommended for most use cases
    ds.create_index("vector", index_type="IVF_PQ", num_partitions=256, num_sub_vectors=16)
    
    # IVF_HNSW_SQ offers better recall at the cost of more memory
    ds.create_index("vector", index_type="IVF_HNSW_SQ", num_partitions=256)
    ```

5. **Store metadata alongside vectors**: Lance efficiently handles mixed workloads with both vector and scalar data:

    ```python
    # Combine vector search with metadata filtering
    results = ds.to_table(
        filter="category = 'electronics'",
        nearest={"column": "vector", "q": query, "k": 10}
    )
    ```

## See Also

- [Vector Search Tutorial](../quickstart/vector-search.md) - Complete guide to vector search with Lance
- [Blob API Guide](blob.md) - Storing and retrieving large binary objects (videos, images)
- [Extension Arrays](arrays.md) - Special array types for ML (BFloat16, images)
- [Performance Guide](performance.md) - Optimization tips for large-scale deployments
