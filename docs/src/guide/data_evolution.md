# Data Evolution

Lance supports traditional schema evolution: adding, removing, and altering columns in a
dataset. Most of these operations can be performed *without* rewriting the
data files in the dataset, making them very efficient operations.
In addition, Lance supports **data evolution**, which allows you to also backfill existing rows
with the new column data without rewriting the data files in the dataset,
making it highly suitable for use cases like ML feature engineering.

In general, schema changes will conflict with most other concurrent write
operations. For example, if you change the schema of the dataset while someone
else is appending data to it, either your schema change or the append will fail,
depending on the order of the operations. Thus, it's recommended to perform
schema changes when no other writes are happening.

## Adding new columns 

### Schema only

A common use case we've seen in production is to add a new column to a dataset without
populating it. This is useful to later run a large distributed job to populate the column
lazily. To do this, you can use the `lance.LanceDataset.add_columns` method to
add columns with `pyarrow.Field` or `pyarrow.Schema`.

```python
table = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(table, "null_columns")

# With pyarrow Field
dataset.add_columns(pa.field("embedding", pa.list_(pa.float32(), 128)))
assert dataset.schema == pa.schema([
    ("id", pa.int64()),
    ("embedding", pa.list_(pa.float32(), 128)),
])

# With pyarrow Schema
dataset.add_columns(pa.schema([
    ("label", pa.string()),
    ("score", pa.float32()),
]))
assert dataset.schema == pa.schema([
    ("id", pa.int64()),
    ("embedding", pa.list_(pa.float32(), 128)),
    ("label", pa.string()),
    ("score", pa.float32()),
])
```

This operation is very fast, as it only updates the metadata of the dataset.

For Lance file format `<= 2.1`, adding sub-columns under an existing `struct` is not supported.
Starting with Lance file format `2.2`, schema-only add can also extend nested `struct` fields
(including `struct` fields nested inside list types), for example by adding
`people.item.location` under `list<struct<...>>`.

### With data backfill

New columns can be added and populated within a single operation using the
`lance.LanceDataset.add_columns` method. There are two ways to specify
how to populate the new columns: first, by providing a SQL expression for each
new column, or second, by providing a function to generate the new column data.

SQL expressions can either be independent expressions or reference existing
columns. SQL literal values can be used to set a single value for all
existing rows.

```python
table = pa.table({"name": pa.array(["Alice", "Bob", "Carla"])})
dataset = lance.write_dataset(table, "names")
dataset.add_columns({
    "hash": "sha256(name)",
    "status": "'active'",
})
print(dataset.to_table().to_pandas())
#     name                                               hash  status
# 0  Alice  b';\xc5\x10b\x97<E\x8dZo-\x8dd\xa0#$cT\xad~\x0...  active
# 1    Bob  b'\xcd\x9f\xb1\xe1H\xcc\xd8D.Z\xa7I\x04\xccs\x...  active
# 2  Carla  b'\xad\x8d\x83\xff\xd8+Z\x8e\xd4)\xe8Y+\\\xb3\...  active
```

You can also provide a Python function to generate the new column data. This can
be used, for example, to compute a new embedding column. This function should
take a PyArrow RecordBatch and return either a PyArrow RecordBatch or a Pandas
DataFrame. The function will be called once for each batch in the dataset.

If the function is expensive to compute and can fail, it is recommended to set
a checkpoint file in the UDF. This checkpoint file saves the state of the UDF
after each invocation, so that if the UDF fails, it can be restarted from the
last checkpoint. Note that this file can get quite large, since it needs to store
unsaved results for up to an entire data file.

```python
import lance
import pyarrow as pa
import numpy as np

table = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(table, "ids")

@lance.batch_udf(checkpoint_file="embedding_checkpoint.sqlite")
def add_random_vector(batch):
    embeddings = np.random.rand(batch.num_rows, 128).astype("float32")
    return pa.RecordBatch.from_arrays(
        [pa.FixedSizeListArray.from_arrays(embeddings.flatten(), 128)],
        names=["embedding"]
    )
dataset.add_columns(add_random_vector)
```

### Using merge

If you have pre-computed one or more new columns, you can add them to an existing
dataset using the `lance.LanceDataset.merge` method. This allows filling in
additional columns without having to rewrite the whole dataset.

To use the `merge` method, provide a new dataset that includes the columns you
want to add, and a column name to use for joining the new data to the existing
dataset.

For example, imagine we have a dataset of embeddings and ids:

```python
table = pa.table({
   "id": pa.array([1, 2, 3]),
   "embedding": pa.array([np.array([1, 2, 3]), np.array([4, 5, 6]),
                          np.array([7, 8, 9])])
})
dataset = lance.write_dataset(table, "embeddings", mode="overwrite")
```

Now if we want to add a column of labels we have generated, we can do so by merging a new table:

```python
new_data = pa.table({
   "id": pa.array([1, 2, 3]),
   "label": pa.array(["horse", "rabbit", "cat"])
})
dataset.merge(new_data, "id")
print(dataset.to_table().to_pandas())
#    id  embedding   label
# 0   1  [1, 2, 3]   horse
# 1   2  [4, 5, 6]  rabbit
# 2   3  [7, 8, 9]     cat
```

## Dropping columns

Finally, you can drop columns from a dataset using the `lance.LanceDataset.drop_columns`
method. This is a metadata-only operation and does not delete the data on disk. This makes
it very quick.

```python
table = pa.table({"id": pa.array([1, 2, 3]),
                 "name": pa.array(["Alice", "Bob", "Carla"])})
dataset = lance.write_dataset(table, "names", mode="overwrite")
dataset.drop_columns(["name"])
print(dataset.schema)
# id: int64
```

Starting with Lance file format `2.2`, nested sub-column removal is supported for
nested types (for example `people.item.city` on `list<struct<...>>`), instead of
being limited to `struct` only.

To actually remove the data from disk, the files must be rewritten to remove the
columns and then the old files must be deleted. This can be done using
`dataset.optimize.compact_files(...)` followed by
`lance.LanceDataset.cleanup_old_versions()`.

!!! warning

    `drop_columns` is metadata-only and remains reversible as long as old versions are retained.
    After `compact_files()` rewrites data files and `cleanup_old_versions()` removes old manifests/files,
    removed data may become permanently unrecoverable.

    For production workflows, use a rollback window:
    - create a tag (or snapshot/backup) before nested column drops
    - delay cleanup until the rollback window has passed
    - only run aggressive cleanup after rollback validation

## Renaming columns

Columns can be renamed using the `lance.LanceDataset.alter_columns` method.

```python
table = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(table, "ids")
dataset.alter_columns({"path": "id", "name": "new_id"})
print(dataset.to_table().to_pandas())
#    new_id
# 0       1
# 1       2
# 2       3
```

This works for nested columns as well. To address a nested column, use a dot
(`.`) to separate the levels of nesting. For example:

```python
data = [
  {"meta": {"id": 1, "name": "Alice"}},
  {"meta": {"id": 2, "name": "Bob"}},
]
schema = pa.schema([
    ("meta", pa.struct([
        ("id", pa.int32()),
        ("name", pa.string()),
    ]))
])
dataset = lance.write_dataset(data, "nested_rename")
dataset.alter_columns({"path": "meta.id", "name": "new_id"})
print(dataset.to_table().to_pandas())
#                                  meta
# 0  {'new_id': 1, 'name': 'Alice'}
# 1    {'new_id': 2, 'name': 'Bob'}
```

## Casting column data types

In addition to changing column names, you can also change the data type of a
column using the `lance.LanceDataset.alter_columns` method. This
requires rewriting that column to new data files, but does not require rewriting
the other columns.

!!! note

    If the column has an index, the index will be dropped if the column type is
    changed.

This method can be used to change the vector type of a column. For example, we
can change a float32 embedding column into a float16 column to save disk space
at the cost of lower precision:

```python
table = pa.table({
   "id": pa.array([1, 2, 3]),
   "embedding": pa.FixedShapeTensorArray.from_numpy_ndarray(
       np.random.rand(3, 128).astype("float32"))
})
dataset = lance.write_dataset(table, "embeddings")
dataset.alter_columns({"path": "embedding",
                       "data_type": pa.list_(pa.float16(), 128)})
print(dataset.schema)
# id: int64
# embedding: fixed_size_list<item: halffloat>[128]
#   child 0, item: halffloat
```
