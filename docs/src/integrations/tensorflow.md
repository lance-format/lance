---
title: TensorFlow
description: Stream Lance datasets into TensorFlow tf.data pipelines with lance-tensorflow.
---

# TensorFlow Integration

The TensorFlow integration is maintained in the
[lance-format/lance-tensorflow](https://github.com/lance-format/lance-tensorflow)
project.

The main Lance Python package no longer includes `lance.tf`. Install
`lance-tensorflow` and import `lance_tensorflow` instead.

```bash
pip install lance-tensorflow
```

## Reading from Lance

Use `lance_tensorflow.from_lance` to create a `tf.data.Dataset` from a Lance
dataset.

```python
from lance_tensorflow import from_lance

ds = from_lance(
    "s3://my-bucket/my-dataset",
    columns=["image", "label"],
    filter="split = 'train'",
    batch_size=256,
)

for batch in ds:
    print(batch["label"])
```

## Dataset Convenience Methods

If you want `tf.data.Dataset.from_lance`, register the convenience methods
explicitly after importing `lance_tensorflow`.

```python
import tensorflow as tf
import lance_tensorflow

lance_tensorflow.register_tensorflow_dataset()

ds = tf.data.Dataset.from_lance("s3://my-bucket/my-dataset")
```

## Migration

Replace old imports:

```python
import lance.tf.data

ds = lance.tf.data.from_lance(uri)
```

with:

```python
from lance_tensorflow import from_lance

ds = from_lance(uri)
```

See the [lance-tensorflow README](https://github.com/lance-format/lance-tensorflow)
for the current installation and compatibility details.
