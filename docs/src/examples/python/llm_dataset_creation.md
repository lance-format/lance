# Creating text dataset for LLM training using Lance

Lance can be used for creating and caching a text (or code) dataset for pre-training / fine-tuning of Large Language Models.
The need for this arises when one needs to train a model on a subset of data or process the data in chunks without downloading
all of it on the disk at once. This becomes a considerable problem when you just want a subset of a Terabyte or Petabyte-scale dataset.

In this example, we will be bypassing this problem by downloading a text dataset in parts, tokenizing it and saving it as a Lance dataset. 
This can be done for as many or as few data samples as you wish with average memory consumption approximately 3-4 GBs!

For this example, we are working with the [wikitext](https://huggingface.co/datasets/Salesforce/wikitext) dataset,
which is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.

## Preparing and pre-processing the raw dataset

Let's first define the dataset and the tokenizer

```python
import lance
import pyarrow as pa

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm  # optional for progress tracking

tokenizer = AutoTokenizer.from_pretrained('gpt2')

dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', streaming=True)['train']
dataset = dataset.shuffle(seed=1337)
```

The `streaming` argument in `load_dataset` is especially important because if you run it without setting it to 
`True`, the datasets library will download the entire dataset first, even though you only wish to use a subset of it.
With `streaming` set to `True`, the samples will be downloaded as they are needed.

Now we will define a function to help us with tokenizing our samples, one-by-one.

```python
def tokenize(sample, field='text'):
    return tokenizer(sample[field])['input_ids']
```

This function will receive a sample from a huggingface dataset and tokenize the values in the `field` column. This is the main text you want 
to tokenize.

## Creating a Lance dataset

Now that we have set up our raw dataset and pre-processing code, 
let's define the main function that takes in the dataset, number of samples and field, and returns a pyarrow batch that will later be written into a lance dataset.

```python
def process_samples(dataset, num_samples=100_000, batch_size=1024, field='text'):
    current_sample = 0
    batch = []
    for sample in tqdm(dataset, total=num_samples):
        # Stop once we have collected the requested number of samples
        if current_sample == num_samples:
            break
        if not sample[field]:
            continue
        # Tokenize the current sample
        tokenized_sample = tokenize(sample, field)
        if not tokenized_sample:
            continue
        batch.append(tokenized_sample)
        # Increment the counter
        current_sample += 1
        # Yield a PyArrow RecordBatch once we accumulate `batch_size` samples
        if len(batch) == batch_size:
            yield pa.RecordBatch.from_arrays(
                [pa.array(batch, type=pa.list_(pa.int64()))],
                names=["input_ids"],
            )
            batch = []

    if batch:
        yield pa.RecordBatch.from_arrays(
            [pa.array(batch, type=pa.list_(pa.int64()))],
            names=["input_ids"],
        )
```

This function iterates over the Hugging Face dataset, tokenizes each non-empty sample into a list of token IDs, and yields a PyArrow `RecordBatch`
every `batch_size` samples until it reaches `num_samples` or the end of the dataset.

Please note that by 'sample', we mean one example (row) in the original dataset. What one example exactly means will depend on the dataset itself as it could 
be one line or an entire file of text. In this example, it varies in length between a line and a paragraph of text.

We also need to define a schema to tell Lance what type of data we are expecting in our table. Each row stores the token IDs for one sample as a variable-length list of 64-bit integers (`pa.list_(pa.int64())`).

```python
schema = pa.schema([
    pa.field("input_ids", pa.list_(pa.int64()))
])
```

Finally, we need to define a `reader` that will be reading a stream of record batches from our `process_samples` function that yields 
batches of tokenized samples.

```python
reader = pa.RecordBatchReader.from_batches(
    schema, 
    process_samples(dataset, num_samples=500_000, batch_size=1024, field='text')  # For 500K samples
)
```

And finally we use the `lance.write_dataset` which will write the dataset to the disk.

```python
# Write the dataset to disk
lance.write_dataset(
    reader, 
    "wikitext_500K.lance",
    schema
)
```

If you want to apply some other pre-processing to the tokens before saving it to the disk (like masking, etc), you may add it in the 
`process_samples` function.

And that's it! Your dataset has been tokenized and saved to the disk! 
