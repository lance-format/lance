# PyTorch Integration

Machine learning users can use `lance.torch.data.LanceDataset`, a
subclass of `torch.utils.data.IterableDataset`, that to use
Lance data directly PyTorch training and inference loops.

It starts with creating a ML dataset for training. With the [HuggingFace integration](huggingface),
it takes just one line of Python to convert a HuggingFace dataset to a Lance dataset.

```python
import datasets # pip install datasets
import lance

hf_ds = datasets.load_dataset(
    "poloclub/diffusiondb",
    split="train",
    # name="2m_first_1k",  # for a smaller subset of the dataset
)
lance.write_dataset(hf_ds, "diffusiondb_train.lance")
```

Then, you can use the Lance dataset in PyTorch training and inference loops.

Note:

1. the PyTorch dataset automatically convert data into `torch.Tensor`

2. lance is not fork-safe. If you are using multiprocessing, use spawn instead.
The safe dataloader uses the spawning method.

## UnSafe Dataloader

```python
import torch
import lance.torch.data

# Load lance dataset into a PyTorch IterableDataset.
# with only columns "image" and "prompt".
dataset = lance.torch.data.LanceDataset(
    "diffusiondb_train.lance",
    columns=["image", "prompt"],
    batch_size=128,
    batch_readahead=8,  # Control multi-threading reads.
)

# Create a PyTorch DataLoader
dataloader = torch.utils.data.DataLoader(dataset)

# Inference loop
for batch in dataloader:
    inputs, targets = batch["prompt"], batch["image"]
    outputs = model(inputs)
    ...
```

## Safe Dataloader

```python
from lance.torch.data import SafeLanceDataset, get_safe_loader

dataset = SafeLanceDataset(temp_lance_dataset)
# use spawn method to avoid fork-safe issue
loader = get_safe_loader(
    dataset,
    num_workers=2,
    batch_size=16,
    drop_last=False,
)

total_samples = 0
for batch in loader:
    total_samples += batch["id"].shape[0]
```

`lance.torch.data.LanceDataset` can composite with the `lance.sampler.Sampler` classes
to control the sampling strategy. For example, you can use `lance.sampler.ShardedFragmentSampler`
to use it in a distributed training environment. If not specified, it is a full scan.

```python
from lance.sampler import ShardedFragmentSampler
from lance.torch.data import LanceDataset

# Load lance dataset into a PyTorch IterableDataset.
# with only columns "image" and "prompt".
dataset = LanceDataset(
    "diffusiondb_train.lance",
    columns=["image", "prompt"],
    batch_size=128,
    batch_readahead=8,  # Control multi-threading reads.
    sampler=ShardedFragmentSampler(
        rank=1,  # Rank of the current process
        world_size=8,  # Total number of processes
    ),
)
```

Available samplers:

- `lance.sampler.ShardedFragmentSampler`
- `lance.sampler.ShardedBatchSampler`

!!! warning

    For multiprocessing you should probably not use fork as lance is
    multi-threaded internally and fork and multi-thread do not work well.
    Refer to [this discussion](https://discuss.python.org/t/concerns-regarding-deprecation-of-fork-with-alive-threads/33555). 