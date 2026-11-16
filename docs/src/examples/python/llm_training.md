# Training LLMs using a Lance text dataset

Using a Lance text dataset for pre-training / fine-tuning a Large Language model is straightforward and memory-efficient.
This example follows up on the [Creating text dataset for LLM training using Lance](llm_dataset_creation.md) example.
Check it out if you haven't already.

In this example, we will be training an LLM using 🤗 transformers on the tokenized "wikitext_500K" lance dataset we created in the aforementioned example.

## Imports and Setup

Let's setup our environment by doing all the necessary imports and defining a few basic things.

```python
import numpy as np
import lance
from lance.torch.data import get_safe_loader

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# We'll be training the pre-trained GPT2 model in this example
model_name = 'gpt2'

# Also define some hyperparameters
lr = 3e-4
nb_epochs = 10
block_size = 1024
batch_size = 8
device = 'cuda:0'
dataset_path = 'wikitext_500K.lance'
```

Now that the basic setup is out of the way, let's define our custom PyTorch `Dataset` for loading batches of tokenized sequences from our Lance dataset.

## Data-loading Setup

We start by defining a utility function that pads or truncates each tokenized sample to `block_size` and masks padding positions with `-100` in `labels` so they do not contribute to the causal language modeling loss.

```python
def format_sequence(tokens, block_size, pad_token_id):
    """Truncate or right-pad a token sequence to `block_size`."""
    seq = tokens[:block_size]
    pad_len = block_size - len(seq)
    input_ids = seq + [pad_token_id] * pad_len
    labels = seq + [-100] * pad_len
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
```

Now let's define a process-safe custom dataset. Because Lance uses an internal multi-threaded runtime, opening a `LanceDataset` in the main process and forking DataLoader workers is not safe. Instead, we read the row count once during initialization, open the dataset lazily inside each worker, and implement `__getitems__` so each batch performs a single vectorized `ds.take(indices)` call rather than one random read per sample.

```python
class LanceDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        block_size,
        pad_token_id,
    ):
        self.dataset_path = dataset_path
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self._ds = None

        ds = lance.dataset(dataset_path)
        self.length = ds.count_rows()
        del ds

    def _get_ds(self):
        if self._ds is None:
            self._ds = lance.dataset(self.dataset_path)
        return self._ds

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        """Fetch a full batch of rows in a single Lance `take` call."""
        rows = self._get_ds().take(indices, columns=["input_ids"]).to_pylist()
        return [
            format_sequence(row["input_ids"], self.block_size, self.pad_token_id)
            for row in rows
        ]
```

Because each row in `wikitext_500K.lance` already holds a distinct tokenized sample (`list<int64>`), PyTorch's DataLoader can pass a batch of row indices directly to `__getitems__`. We pair the dataset with `lance.torch.data.get_safe_loader`, which configures the `"spawn"` multiprocessing context and persistent workers automatically.

And that's it for data loading! Now all that is left is to train the model.

## Model Training

Now you train the model just like you would with any other dataset!

```python
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define the dataset and process-safe dataloader
    dataset = LanceDataset(dataset_path, block_size, tokenizer.pad_token_id)
    dataloader = get_safe_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Define the optimizer, training loop and train the model!
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(nb_epochs):
        print(f"========= Epoch: {epoch+1} / {nb_epochs} =========")
        epoch_loss = []
        prog_bar = tqdm(dataloader, total=len(dataloader))
        for batch in prog_bar:
            optimizer.zero_grad(set_to_none=True)

            # Put both input_ids and labels to the device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Perform one forward pass and get the loss
            outputs = model(**batch)
            loss = outputs.loss

            # Perform backward pass
            loss.backward()
            optimizer.step()

            prog_bar.set_description(f"loss: {loss.item():.4f}")

            epoch_loss.append(loss.item())

        # Calculate training perplexity for this epoch
        try:
            perplexity = np.exp(np.mean(epoch_loss))
        except OverflowError:
            perplexity = float("-inf")

        print(f"train_perplexity: {perplexity}")
```

One tip: If your lance dataset is huge (like the wikitext_500K is), and you want to debug the model to look out for errors, you may want to wrap the dataloader in an `iter()` function and only run it for a couple batches.

And that's basically it!

The best part about using Lance, the custom Dataset and Sampler is that you get a whopping **95%** average GPU utilisation and minimal CPU overhead thanks to the lightning fast random access that Lance provides 🚀 
