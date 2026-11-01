"""
Generate test data with Lance 0.29.0 to create an index without created_at field.
This tests backward compatibility for the created_at field added in 0.30.0.
"""

import lance
import pyarrow as pa
import pyarrow.compute as pc

# To generate the test file, we should be running this older version of lance
assert lance.__version__ == "0.29.0", f"Expected lance 0.29.0, got {lance.__version__}"

# Create a simple dataset with a vector index
ndims = 16
nvecs = 256

data = pa.table(
    {
        "id": pa.array(range(nvecs)),
        "vec": pa.FixedSizeListArray.from_arrays(
            pc.random(ndims * nvecs).cast(pa.float32()), ndims
        ),
    }
)

# Write dataset
dataset = lance.write_dataset(data, "index_without_created_at")

# Create a vector index
dataset.create_index(
    "vec",
    "IVF_PQ",
    num_partitions=1,
    num_sub_vectors=4,
)

print("Created dataset with IVF_PQ index on 'vec' field")
print(f"Dataset version: {dataset.version}")
print(f"Index created with Lance {lance.__version__}")
