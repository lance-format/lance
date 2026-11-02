# Creating Multi-Modal datasets using Lance

Thanks to Lance file format's ability to store data of different modalities, one of the important use-cases that Lance shines in is storing Multi-modal datasets.
In this brief example we will be going over how you can take a Multi-modal dataset and store it in Lance file format. 

The dataset of choice here is [Flickr8k dataset](https://github.com/goodwillyoga/Flickr8k_dataset). Flickr8k is a benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. 
The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.

We will be creating an Image-caption pair dataset for Multi-modal model training by using the above mentioned Flickr8k dataset, saving it in form of a Lance dataset with image file names, all captions for every image (order preserved) and the image bytes stored in a Lance [Blob v2 column](../../guide/blob.md).

## Imports and Setup

We assume that you downloaded the dataset, more specifically the "Flickr8k.token.txt" file and the "Flicker8k_Dataset/" folder and both are present in the current directory.
These can be downloaded from [here](https://github.com/goodwillyoga/Flickr8k_dataset?tab=readme-ov-file) (download both the dataset and text zip files).

We also assume you have pyarrow and pylance installed as well as opencv (for reading in images) and tqdm (for pretty progress bars).

Now let's start with imports and defining the caption file and image dataset folder.

```python
import os
import cv2
import random

import lance
from lance import blob_array, blob_field
import pyarrow as pa

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

captions = "Flickr8k.token.txt"
image_folder = "Flicker8k_Dataset/"
```

## Loading and Processing

In flickr8k dataset, each image has multiple corresponding captions that are ordered. 
We are going to put all these captions in a list corresponding to each image with their position in the list representing the order in which they originally appear.
Let's load the annotations (the image path and corresponding captions) in a list with each element of the list being a tuple consisting of image name, caption number and caption itself.

```python
with open(captions, "r") as fl:
    annotations = fl.readlines()

# Converts the annotations where each element of this list is a tuple consisting of image file name, caption number and caption itself
annotations = list(map(lambda x: tuple([*x.split('\t')[0].split('#'), x.split('\t')[1]]), annotations))
```

Now, for all captions of the same image, we will put them in a list sorted by their ordering.

```python
captions = []
image_ids = set(ann[0] for ann in annotations)
for img_id in tqdm(image_ids):
    current_img_captions = []
    for ann_img_id, num, caption in annotations:
        if img_id == ann_img_id:
            current_img_captions.append((num, caption))
            
    # Sort by the annotation number
    current_img_captions.sort(key=lambda x: x[0])
    captions.append((img_id, tuple([x[1] for x in current_img_captions])))
```

## Converting to a Lance Dataset

Now that our captions list is in a proper format, we will define the table schema using `blob_field("image")` for the raw image bytes and write a `process()` function that accumulates rows into batches of `batch_size` images before yielding a PyArrow `RecordBatch`.

```python
schema = pa.schema([
    pa.field("image_id", pa.string()),
    blob_field("image"),
    pa.field("captions", pa.list_(pa.string(), -1)),
])
```

We are including the `image_id` (which is the original image name) so it can be easier to reference and debug in the future.

```python
def process(captions, batch_size=256):
    image_ids, images, caption_lists = [], [], []
    for img_id, img_captions in tqdm(captions):
        try:
            with open(os.path.join(image_folder, img_id), 'rb') as im:
                binary_im = im.read()
                
        except FileNotFoundError:
            print(f"img_id '{img_id}' not found in the folder, skipping.")
            continue
        
        image_ids.append(img_id)
        images.append(binary_im)
        caption_lists.append(img_captions)

        if len(image_ids) == batch_size:
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(image_ids, type=pa.string()),
                    blob_array(images),
                    pa.array(caption_lists, pa.list_(pa.string(), -1)),
                ],
                schema=schema,
            )
            image_ids, images, caption_lists = [], [], []

    if image_ids:
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(image_ids, type=pa.string()),
                blob_array(images),
                pa.array(caption_lists, pa.list_(pa.string(), -1)),
            ],
            schema=schema,
        )
```

Finally, we define a reader to iteratively read those record batches and write them to a Lance dataset on disk using `data_storage_version="2.2"` for Blob v2 support.

```python
reader = pa.RecordBatchReader.from_batches(schema, process(captions))
lance.write_dataset(
    reader,
    "flickr8k.lance",
    schema=schema,
    data_storage_version="2.2",
)
```

And that's basically it! If you want to execute this in a notebook form, you can check out this example in our deeplearning-recipes repository [here](https://github.com/lancedb/lance-deeplearning-recipes/tree/main/examples/flickr8k-dataset).

For more Deep learning related examples using Lance dataset, be sure to check out the [lance-deeplearning-recipes](https://github.com/lancedb/lance-deeplearning-recipes) repository! 
