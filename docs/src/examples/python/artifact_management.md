# Deep Learning Artifact Management using Lance

Along with datasets, Lance file format can also be used for saving and versioning deep learning model weights. 
In fact deep learning artifact management can be made more streamlined (compared to vanilla weight saving methods) using Lance file format for PyTorch model weights.

In this example we will be demonstrating how you save, version and load a PyTorch model's weights using Lance. More specifically we will be loading a pre-trained ResNet model, saving it in Lance file format, loading it back to PyTorch and verifying if the weights are still indeed the same.
We will also be demonstrating how you can version your model weights in a single lance dataset thanks to our Zero-copy, automatic versioning.

**Key Idea:** When you save a model's weights (read: state dictionary) in PyTorch, weights are stored as key-value pairs in an `OrderedDict` with the keys representing the weight's name and the value representing the corresponding weight tensor.
To emulate this as closely as possible, we will be saving the weights in three columns. The first column will have the name of the weight, the second will have the weight itself but flattened in a list and the third will have the original shape of the weights so they can be reconstructed for loading into a model.

## Imports and Setup

We will start by importing and loading all the necessary modules.

```python
import os
import shutil
import lance
import pyarrow as pa
import torch
from collections import OrderedDict
```

We will also define a `GLOBAL_SCHEMA` that will dictate how the weights table will look like.

```python
GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        pa.field("value", pa.list_(pa.float64(), -1)),
        pa.field("shape", pa.list_(pa.int64(), -1)), # Is a list with variable shape because weights can have any number of dims
    ]
)
```

As we covered earlier, the weights table will have three columns - one for storing the weight name, one for storing the flattened weight value and one for storing the original weight shape for loading them back.

## Saving and Versioning Models

First we will focus on the model saving part. Let's start by writing a utility function that iterates through the model's `state_dict`, flattens each weight tensor, and yields PyArrow `RecordBatch` objects in batches of parameters rather than creating a single-row batch per tensor.

```python
def _save_model_writer(state_dict, batch_size=64):
    """Yields batched RecordBatches of parameters from the model state dict"""
    names, values, shapes = [], [], []
    for param_name, param in state_dict.items():
        names.append(param_name)
        values.append(param.detach().cpu().flatten().tolist())
        shapes.append(list(param.size()))

        if len(names) == batch_size:
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(names, pa.string()),
                    pa.array(values, pa.list_(pa.float64(), -1)),
                    pa.array(shapes, pa.list_(pa.int64(), -1)),
                ],
                schema=GLOBAL_SCHEMA,
            )
            names, values, shapes = [], [], []

    if names:
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(names, pa.string()),
                pa.array(values, pa.list_(pa.float64(), -1)),
                pa.array(shapes, pa.list_(pa.int64(), -1)),
            ],
            schema=GLOBAL_SCHEMA,
        )
```

Now about versioning: Let's say you trained your model on some new data but don't want to overwrite your old checkpoint, you can now just save these newly trained model weights as a version in Lance weights dataset.
This will allow you to load specific version of weights from one lance weight dataset instead of making separate folders for each model checkpoint to make.

Let's write a function that handles the work for saving the model, whether with versions or without them.

```python
def save_model(state_dict: OrderedDict, file_name: str, version=False):
    """Saves a PyTorch model in lance file format

    Args:
        state_dict (OrderedDict): Model state dict
        file_name (str): Lance model name
        version (bool): Whether to save as a new version or overwrite the existing versions,
            if the lance file already exists
    """
    # Create a reader
    reader = pa.RecordBatchReader.from_batches(
        GLOBAL_SCHEMA, _save_model_writer(state_dict)
    )

    if os.path.exists(file_name):
        if version:
            # If we want versioning, we use the overwrite mode to create a new version
            lance.write_dataset(
                reader, file_name, schema=GLOBAL_SCHEMA, mode="overwrite"
            )
        else:
            # If we don't want versioning, we delete the existing file and write a new one
            shutil.rmtree(file_name)
            lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)
    else:
        # If the file doesn't exist, we write a new one
        lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)
```

The above function will take in the model state dict, the lance saved file name and the weights version. The function will start by making a `RecordBatchReader` using the global schema and the utility function we wrote above.
If the weights lance dataset already exists in the directory, we will just save it as a new version (if versioning is enabled) or delete the old file and save the weights as new. Otherwise the weights saving will be done normally.

## Loading Models

Loading weights from a Lance weight dataset into a model is just the reverse of saving them. The key part is to reshape the flattened weights back to their original shape, which is easier thanks to the shape that you saved corresponding to the weights.
We will divide this into three functions for better readability.

The first function will be the `_load_weight` function which will take a "weight" retrieved from the Lance weight dataset and return the weight as a torch tensor in its original shape. The "weight" that we retrieve from the Lance weight dataset will be a dict with value corresponding to each column in form of a key.

```python
def _load_weight(weight: dict) -> torch.Tensor:
    """Converts a weight dict to a torch tensor"""
    return torch.tensor(weight["value"], dtype=torch.float64).reshape(weight["shape"])
```

Optionally, you could also add an option to specify the datatype of the weights.

The next function will be on loading all the weights from the lance weight dataset into a state dictionary, which is what PyTorch will expect when we load the weights into our model.

```python
def _load_state_dict(file_name: str, version: int = 1, map_location=None) -> OrderedDict:
    """Reads the model weights from lance file and returns a model state dict
    If the model weights are too large, this function will fail with a memory error.

    Args:
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on

    Returns:
        OrderedDict: Model state dict
    """
    ds = lance.dataset(file_name, version=version)
    state_dict = OrderedDict()

    for batch in ds.to_batches():
        for weight in batch.to_pylist():
            state_dict[weight["name"]] = _load_weight(weight).to(map_location)

    return state_dict
```

The `load_state_dict` function will expect a lance weight dataset file name, a version and a device where the weights will be loaded into. 
We essentially load all the weights from the lance weight dataset into our memory and iteratively convert them into weights using the utility function we wrote earlier and then put them on the device.

One thing to note here is that this function will fail if the saved weights are larger than memory. For the sake of simplicity, we assume the weights to be loaded can fit in the memory and we don't have to deal with any sharding.

Finally, we will write a higher level function is the only one we will call to load the weights.

```python
def load_model(
    model: torch.nn.Module, file_name: str, version: int = 1, map_location=None
):
    """Loads the model weights from lance file and sets them to the model

    Args:
        model (torch.nn.Module): PyTorch model
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on
    """
    state_dict = _load_state_dict(file_name, version=version, map_location=map_location)
    model.load_state_dict(state_dict)
```

The `load_model` function will require the model, the lance weight dataset name, the version of weights to load in and the map location. This will just call the `_load_state_dict` utility to get the state dict and then load that state dict into the model.

## Conclusion

In conclusion, you only need to call the two functions: `save_model` and `load_model` to save and load the models respectively and as long as the weights can be fit in the memory and are in PyTorch, it should be fine.

Although experimental, this approach defines a new way of doing deep learning artifact management. 
