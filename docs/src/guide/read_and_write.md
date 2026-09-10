# Read and Write Data

## Writing Lance Dataset

If you're familiar with [Apache PyArrow](https://arrow.apache.org/docs/python/getstarted.html), you'll find that creating a Lance dataset is straightforward.
Begin by writing a `pyarrow.Table` using the `lance.write_dataset` function.

```python
import lance
import pyarrow as pa

table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
                              {"name": "Bob", "age": 30}])
ds = lance.write_dataset(table, "./alice_and_bob.lance")
```

If the dataset is too large to fully load into memory, you can stream data using `lance.write_dataset`
also supports `Iterator` of `pyarrow.RecordBatch` es.
You will need to provide a `pyarrow.Schema` for the dataset in this case.

```python
from typing import Iterator

def producer() -> Iterator[pa.RecordBatch]:
    """An iterator of RecordBatches."""
    yield pa.RecordBatch.from_pylist([{"name": "Alice", "age": 20}])
    yield pa.RecordBatch.from_pylist([{"name": "Bob", "age": 30}])

schema = pa.schema([
    ("name", pa.string()),
    ("age", pa.int32()),
])

ds = lance.write_dataset(producer(),
                         "./alice_and_bob.lance",
                         schema=schema, mode="overwrite")
print(ds.count_rows())  # Output: 2
```

`lance.write_dataset` supports writing `pyarrow.Table`, `pandas.DataFrame`,
`pyarrow.dataset.Dataset`, and `Iterator[pyarrow.RecordBatch]`.

## Adding Rows

To insert data into your dataset, you can use either `LanceDataset.insert`
or `lance.write_dataset` with `mode=append`.

```python
import lance
import pyarrow as pa

table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
                              {"name": "Bob", "age": 30}])
ds = lance.write_dataset(table, "./insert_example.lance")

new_table = pa.Table.from_pylist([{"name": "Carla", "age": 37}])
ds.insert(new_table)
print(ds.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   37

new_table2 = pa.Table.from_pylist([{"name": "David", "age": 42}])
ds = lance.write_dataset(new_table2, ds, mode="append")
print(ds.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   37
# 3  David   42
```

## Deleting rows

Lance supports deleting rows from a dataset using a SQL filter, as described in [Filter push-down](#filter-push-down).
For example, to delete Bob's row from the dataset above, one could use:

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.delete("name = 'Bob'")
dataset2 = lance.dataset("./alice_and_bob.lance")
print(dataset2.to_table().to_pandas())
#     name  age
# 0  Alice   20
```

!!! note

    [Lance Format is immutable](../format/index.md). Each write operation creates a new version of the dataset,
    so users must reopen the dataset to see the changes. Likewise, rows are removed by marking
    them as deleted in a separate deletion index, rather than rewriting the files. This approach
    is faster and avoids invalidating any indices that reference the files, ensuring that subsequent
    queries do not return the deleted rows.

## Updating rows

Lance supports updating rows based on SQL expressions with the
`lance.LanceDataset.update` method. For example, if we notice
that Bob's name in our dataset has been sometimes written as `Blob`, we can fix
that with:

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.update({"name": "'Bob'"}, where="name = 'Blob'")
```

The update values are SQL expressions, which is why `'Bob'` is wrapped in single
quotes. This means we can use complex expressions that reference existing columns if
we wish. For example, if two years have passed and we wish to update the ages
of Alice and Bob in the same example, we could write:

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.update({"age": "age + 2"})
```

If you are trying to update a set of individual rows with new values then it is often
more efficient to use the merge insert operation described below.

```python
import lance

# Change the ages of both Alice and Bob
new_table = pa.Table.from_pylist([{"name": "Alice", "age": 30},
                                  {"name": "Bob", "age": 20}])

# This works, but is inefficient, see below for a better approach
dataset = lance.dataset("./alice_and_bob.lance")
for idx in range(new_table.num_rows):
  name = new_table[0][idx].as_py()
  new_age = new_table[1][idx].as_py()
  dataset.update({"age": new_age}, where=f"name='{name}'")
```

## Merge Insert

Lance supports a merge insert operation. This can be used to add new data in bulk
while also (potentially) matching against existing data. This operation can be used
for a number of different use cases.

### Bulk Update

The `lance.LanceDataset.update` method is useful for updating rows based on
a filter. However, if we want to replace existing rows with new rows then a `lance.LanceDataset.merge_insert`
operation would be more efficient:

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30

# Change the ages of both Alice and Bob
new_table = pa.Table.from_pylist([{"name": "Alice", "age": 2},
                                  {"name": "Bob", "age": 3}])
# This will use `name` as the key for matching rows.  Merge insert
# uses a JOIN internally and so you typically want this column to
# be a unique key or id of some kind.
rst = dataset.merge_insert("name") \
       .when_matched_update_all() \
       .execute(new_table)
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice    2
# 1    Bob    3
```

Note that, similar to the update operation, rows that are modified will
be removed and inserted back into the table, changing their position to
the end. Also, the relative order of these rows could change because we
are using a hash-join operation internally.

### Insert if not Exists

Sometimes we only want to insert data if we haven't already inserted it
before. This can happen, for example, when we have a batch of data but
we don't know which rows we've added previously and we don't want to
create duplicate rows. We can use the merge insert operation to achieve
this:

```python
# Bob is already in the table, but Carla is new
new_table = pa.Table.from_pylist([{"name": "Bob", "age": 30},
                                  {"name": "Carla", "age": 37}])

dataset = lance.dataset("./alice_and_bob.lance")

# This will insert Carla but leave Bob unchanged
_ = dataset.merge_insert("name") \
       .when_not_matched_insert_all() \
       .execute(new_table)
# Verify that Carla was added but Bob remains unchanged
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   37
```

### Update or Insert (Upsert)

Sometimes we want to combine both of the above behaviors. If a row
already exists we want to update it. If the row does not exist we want
to add it. This operation is sometimes called "upsert". We can use
the merge insert operation to do this as well:

```python
import lance
import pyarrow as pa

# Change Carla's age and insert David
new_table = pa.Table.from_pylist([{"name": "Carla", "age": 27},
                                  {"name": "David", "age": 42}])

dataset = lance.dataset("./alice_and_bob.lance")

# This will update Carla and insert David
_ = dataset.merge_insert("name") \
       .when_matched_update_all() \
       .when_not_matched_insert_all() \
       .execute(new_table)
# Verify the results
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   27
# 3  David   42
```

### Replace a Portion of Data

A less common, but still useful, behavior can be to replace some region
of existing rows (defined by a filter) with new data. This is similar
to performing both a delete and an insert in a single transaction. For
example:

```python
import lance
import pyarrow as pa

new_table = pa.Table.from_pylist([{"name": "Edgar", "age": 46},
                                  {"name": "Francene", "age": 44}])

dataset = lance.dataset("./alice_and_bob.lance")
print(dataset.to_table().to_pandas())
#       name  age
# 0    Alice   20
# 1      Bob   30
# 2  Charlie   45
# 3    Donna   50

# This will remove anyone above 40 and insert our new data
_ = dataset.merge_insert("name") \
       .when_not_matched_insert_all() \
       .when_not_matched_by_source_delete("age >= 40") \
       .execute(new_table)
# Verify the results - people over 40 replaced with new data
print(dataset.to_table().to_pandas())
#        name  age
# 0     Alice   20
# 1       Bob   30
# 2     Edgar   46
# 3  Francene   44
```

## Reading Lance Dataset

To open a Lance dataset, use the `lance.dataset` function:

```python
import lance
ds = lance.dataset("s3://bucket/path/imagenet.lance")
# Or local path
ds = lance.dataset("./imagenet.lance")
```

!!! note

    Lance supports local file system, AWS `s3` and Google Cloud Storage(`gs`) as storage backends
    at the moment. Read more in [Object Store Configuration](object_store.md).

The most straightforward approach for reading a Lance dataset is to utilize the `lance.LanceDataset.to_table`
method in order to load the entire dataset into memory.

```python
table = ds.to_table()
```

Due to Lance being a high-performance columnar format, it enables efficient reading of subsets of the dataset by utilizing
**Column (projection)** push-down and **filter (predicates)** push-downs.

```python
table = ds.to_table(
    columns=["image", "label"],
    filter="label = 2 AND text IS NOT NULL",
    limit=1000,
    offset=3000)
```

Lance understands the cost of reading heavy columns such as `image`.
Consequently, it employs an optimized query plan to execute the operation efficiently.

### Iterative Read

If the dataset is too large to fit in memory, you can read it in batches
using the `lance.LanceDataset.to_batches` method:

```python
for batch in ds.to_batches(columns=["image"], filter="label = 10"):
    # do something with batch
    compute_on_batch(batch)
```

Unsurprisingly, `lance.LanceDataset.to_batches` takes the same parameters
as `lance.LanceDataset.to_table` function.

### Filter push-down

Lance embraces the utilization of standard SQL expressions as predicates for dataset filtering.
By pushing down the SQL predicates directly to the storage system,
the overall I/O load during a scan is significantly reduced.

Currently, Lance supports a growing list of expressions.

* `>`, `>=`, `<`, `<=`, `=`
* `AND`, `OR`, `NOT`
* `IS NULL`, `IS NOT NULL`
* `IS TRUE`, `IS NOT TRUE`, `IS FALSE`, `IS NOT FALSE`
* `IN`
* `LIKE`, `NOT LIKE`
* `regexp_match(column, pattern)`
* `CAST`

For example, the following filter string is acceptable:

```sql
((label IN [10, 20]) AND (note['email'] IS NOT NULL))
    OR NOT note['created']
```

Nested fields can be accessed using the subscripts. Struct fields can be
subscripted using field names, while list fields can be subscripted using
indices.

If your column name contains special characters or is a [SQL Keyword](https://docs.rs/sqlparser/latest/sqlparser/keywords/index.html),
you can use backtick (`` ` ``) to escape it. For nested fields, each segment of the
path must be wrapped in backticks.

```sql
`CUBE` = 10 AND `column name with space` IS NOT NULL
  AND `nested with space`.`inner with space` < 2
```

!!! warning

    Field names containing periods (`.`) are not supported.

Literals for dates, timestamps, and decimals can be written by writing the string
value after the type name. For example

```sql
date_col = date '2021-01-01'
and timestamp_col = timestamp '2021-01-01 00:00:00'
and decimal_col = decimal(8,3) '1.000'
```

For timestamp columns, the precision can be specified as a number in the type
parameter. Microsecond precision (6) is the default.

| SQL | Time unit |
|-----|-----------|
| `timestamp(0)` | Seconds |
| `timestamp(3)` | Milliseconds |
| `timestamp(6)` | Microseconds |
| `timestamp(9)` | Nanoseconds |

Lance internally stores data in Arrow format. The mapping from SQL types to Arrow is:

| SQL type | Arrow type |
|----------|------------|
| `boolean` | `Boolean` |
| `tinyint` / `tinyint unsigned` | `Int8` / `UInt8` |
| `smallint` / `smallint unsigned` | `Int16` / `UInt16` |
| `int` or `integer` / `int unsigned` or `integer unsigned` | `Int32` / `UInt32` |
| `bigint` / `bigint unsigned` | `Int64` / `UInt64` |
| `float` | `Float32` |
| `double` | `Float64` |
| `decimal(precision, scale)` | `Decimal128` |
| `date` | `Date32` |
| `timestamp` | `Timestamp` (1) |
| `string` | `Utf8` |
| `binary` | `Binary` |

(1) See precision mapping in previous table.

### Random read

One distinct feature of Lance, as columnar format, is that it allows you to read random samples quickly.

```python
# Access the 2nd, 101th and 501th rows
data = ds.take([1, 100, 500], columns=["image", "label"])
```

The ability to achieve fast random access to individual rows plays a crucial role in facilitating various workflows
such as random sampling and shuffling in ML training.
Additionally, it empowers users to construct secondary indices,
enabling swift execution of queries for enhanced performance.

## Table Maintenance

Some operations over time will cause a Lance dataset to have a poor layout. For
example, many small appends will lead to a large number of small fragments. Or
deleting many rows will lead to slower queries due to the need to filter out
deleted rows.

To address this, Lance provides methods for optimizing dataset layout.

### Compact data files

Data files can be rewritten so there are fewer files. When passing a
`target_rows_per_fragment` to `lance.dataset.DatasetOptimizer.compact_files`,
Lance will skip any fragments that are already above that row count, and rewrite
others. Fragments will be merged according to their fragment ids, so the inherent
ordering of the data will be preserved.

!!! note

    Compaction creates a new version of the table. It does not delete the old
    version of the table and the files referenced by it.

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.optimize.compact_files(target_rows_per_fragment=1024 * 1024)
```

During compaction, Lance can also remove deleted rows. Rewritten fragments will
not have deletion files. This can improve scan performance since the soft deleted
rows don't have to be skipped during the scan.

When files are rewritten, the original row addresses are invalidated. This means the
affected files are no longer part of any ANN index if they were before. Because
of this, it's recommended to rewrite files before re-building indices.

<!-- TODO: remove this last comment once stable row ids are default. -->

### Cleanup old versions

Lance is an immutable format — every write creates a new version. The new version
only writes the data that changed, so an insert writes the new rows and an update
rewrites the affected columns for the affected rows. Even a delete creates a
small deletion file. However, old versions still reference the previous data
files, so those files are kept on disk until explicitly removed. Over time this
means storage grows with each operation — inserts, updates, and deletes alike.

Keeping old versions has important benefits: readers that opened an older version
can continue reading it without interference from concurrent writers, providing
snapshot isolation. Old versions also enable time travel queries, letting you
read the dataset as it existed at any prior point in time.

`cleanup_old_versions` deletes old version metadata and any data files that are
no longer referenced by any version, reclaiming the accumulated storage.

!!! warning

    Once old versions are cleaned up, time travel queries to those versions are
    no longer possible. Choose your retention window (`older_than`) accordingly —
    any version removed by cleanup cannot be recovered.

```python
import lance

dataset = lance.dataset("./my_dataset.lance")
dataset.cleanup_old_versions()
```

By default, versions older than 7 days are removed. You can override this with
the `older_than` parameter (a `timedelta`):

```python
from datetime import timedelta

dataset.cleanup_old_versions(older_than=timedelta(days=1))
```

!!! note

    Tagged versions are exempt from cleanup. See [Tags and Branches](tags_and_branches.md)
    for details.

By default, Lance only removes files that it can **verify** are no longer needed.
A file is verified when Lance can see that it was referenced by an older version
and is no longer referenced by any newer version. However, some orphaned files
cannot be verified this way — for example, files left behind by aborted or failed
commits that were never recorded in any version. These files are
indistinguishable from files being written by an in-progress operation.

Cleanup will never delete the current (active) version. This means passing
`older_than=timedelta(0)` is safe and will delete all versions except the current
one.

The `delete_unverified` flag enables a more aggressive strategy that will also
delete these unverified files:

```python
dataset.cleanup_old_versions(
    older_than=timedelta(hours=2),
    delete_unverified=True,
)
```

!!! danger

    Only use `delete_unverified=True` when you are confident that no other
    concurrent operation has been in-progress for longer than the `older_than`
    duration. Lance uses the file's age to decide whether an unverified file is
    safe to remove, so any operation that is still running past the `older_than`
    window risks having its files deleted out from under it.

    In particular, combining `delete_unverified=True` with `older_than=timedelta(0)`
    is **extremely dangerous** — if any other operation is in-progress at all,
    its data files may be deleted, leading to dataset corruption.

### Automatic cleanup

Instead of calling `cleanup_old_versions` manually, you can configure Lance to
clean up old versions automatically during writes. When auto cleanup is enabled,
Lance will run cleanup every *N* commits (the **interval**), removing versions
older than a specified duration.

Auto cleanup can be enabled when creating a new dataset:

```python
import lance
import pyarrow as pa
from lance.dataset import AutoCleanupConfig

table = pa.table({"id": range(100)})
ds = lance.write_dataset(
    table,
    "./my_dataset.lance",
    auto_cleanup_options=AutoCleanupConfig(
        interval=20,             # run cleanup every 20 commits
        older_than_seconds=3600, # remove versions older than 1 hour
    ),
)
```

Or enabled on an existing dataset:

```python
ds = lance.dataset("./my_dataset.lance")
ds.optimize.enable_auto_cleanup(
    AutoCleanupConfig(
        interval=20,
        older_than_seconds=3600,
    )
)
```

And disabled again:

```python
ds.optimize.disable_auto_cleanup()
```

Auto cleanup parameters can also be set directly via dataset config keys:

```python
ds.update_config({
    "lance.auto_cleanup.interval": "20",
    "lance.auto_cleanup.older_than": "3600s",
})
```

!!! warning

    Auto cleanup runs as part of the commit path. If your writer does not have
    delete permissions, or you are doing high-frequency writes where the extra
    latency matters, pass `skip_auto_cleanup=True` to `write_dataset` to skip it
    on a per-write basis.

### Other cleanup strategies

It is common to run cleanup as a periodic background task on a dedicated server
(for example, via a cron job or scheduled workflow). This keeps cleanup off the
write path entirely, avoiding any impact to write latency, but requires setting
up and maintaining additional infrastructure.
