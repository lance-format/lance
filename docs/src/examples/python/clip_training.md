# Training Multi-Modal models using a Lance dataset

In this example we will be training a CLIP model for natural image based search using a Lance image-text dataset. 
In particular, we will be using the [flickr_8k Lance dataset](https://www.kaggle.com/datasets/heyytanay/flickr-8k-lance).

The model architecture and part of the training code are adapted from Manan Goel's [Implementing CLIP with PyTorch Lightning](https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1) with necessary changes to for a minimal, lance-compatible training example.

## Imports and Setup

Along with Lance, we will be needing [PyTorch](https://pytorch.org/get-started/locally/) and [timm](https://github.com/huggingface/pytorch-image-models) for our CLIP model to train.

```python
import cv2
import lance
from lance.torch.data import get_safe_loader

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import timm
from transformers import AutoModel, AutoTokenizer

import itertools
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')
```

Now, we will define a Config class that will house all the hyper-parameters required for training.

```python
class Config:
    img_size = (128, 128)
    bs = 32
    head_lr = 1e-3
    img_enc_lr = 1e-4
    text_enc_lr = 1e-5
    max_len = 18
    img_embed_dim = 2048
    text_embed_dim = 768
    projection_dim = 256
    temperature = 1.0
    num_epochs = 2
    img_encoder_model = 'resnet50'
    text_encoder_model = 'bert-base-cased'
```

And also two utility functions that will help us decode each image payload and pick a caption. 
Remember, our Lance dataset has images, image names and all the captions for a given image. We only need the images and one of those captions. 
For simplicity, when loading captions, we will be choosing the one that is the longest (with the rather naive assumption that it has more information about the image).

```python
def decode_image(raw_bytes):
    # Convert raw image bytes into an OpenCV RGB image
    raw_img = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def select_caption(captions):
    # Return the longest caption for an image
    return max(captions, key=len)
```

`decode_image()` converts the raw image bytes retrieved from Lance into an image array using NumPy and OpenCV.

## Dataset and Augmentations

Since our CLIP model expects resized images and tokenized captions, we will define a custom PyTorch dataset that opens the Lance dataset lazily inside each worker process and implements `__getitems__` so an entire batch of images and captions is fetched in one batched operation instead of calling `ds.take([idx])` twice per sample.

```python
class CLIPLanceDataset(Dataset):
    """Custom Dataset to load images and their corresponding captions"""
    def __init__(self, lance_path, max_len=18, tokenizer=None, transforms=None):
        self.lance_path = lance_path
        self.max_len = max_len
        # Init a new tokenizer if not specified already
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') if not tokenizer else tokenizer
        self.transforms = transforms
        self._ds = None

        ds = lance.dataset(lance_path)
        self._len = ds.count_rows()
        del ds

    def _get_ds(self):
        if self._ds is None:
            self._ds = lance.dataset(self.lance_path)
        return self._ds

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        ds = self._get_ds()
        # Read a full batch of image payloads from the Blob v2 column and their captions.
        # (If your dataset stores `image` as a plain binary column instead of a blob column,
        # use `ds.take(indices, columns=["image", "captions"]).to_pydict()`.)
        image_bytes = [
            payload
            for _, payload in ds.read_blobs(
                "image", indices=indices, preserve_order=True
            )
        ]
        caption_rows = ds.take(indices, columns=["captions"]).to_pydict()["captions"]

        samples = []
        for raw_bytes, captions in zip(image_bytes, caption_rows):
            img = decode_image(raw_bytes)
            if self.transforms:
                img = self.transforms(img)

            tokenized = self.tokenizer(
                select_caption(captions),
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt',
            )
            tokenized = {k: v.flatten() for k, v in tokenized.items()}
            samples.append((img, tokenized))

        return samples
```

Now that our custom dataset is ready, we also define some very basic augmentations for our images.

```python
train_augments = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(Config.img_size),
        transforms.Normalize([0.5], [0.5]),
    ]
)
```

The transformations are very basic: resizing all the images to be of the same shape and then normalizing them to stabilize the training later on.

## Model and Setup

Since we our training a CLIP model, we have the following:
* `ImageEncoder` that uses a pre-trained vision model (`resnet50` in this case) to convert images into feature vectors.
* `TextEncoder` that uses a pre-trained language model (`bert-base-cased` in this case) to transform text captions into feature vectors.
* `Head` which is a Projection module projects these feature vectors into a common embedding space.

Going into deeper details of the CLIP model and its architectural nuances are out of the scope of this example, however if you wish to read more on it, you can read the official paper [here](https://arxiv.org/abs/2103.00020/).

Now that we have understood the general summary of the model, let's define all the required modules.

```python
class ImageEncoder(nn.Module):
    """Encodes the Image"""
    def __init__(self, model_name, pretrained = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img):
        return self.backbone(img)

class TextEncoder(nn.Module):
    """Encodes the Caption"""
    def __init__(self, model_name):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, captions):
        output = self.backbone(**captions)
        return output.last_hidden_state[:, 0, :]

class Head(nn.Module):
    """Projects both into Embedding space"""
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected

        return self.layer_norm(x)
```

Along with the model definition, we will be defining two utility functions to simplify the training: `forward()` which will do one forward pass through the combined models and `loss_fn()` which will take the image and text embeddings output from `forward` function and then calculate the loss using them.

```python
def loss_fn(img_embed, text_embed, temperature=0.2):
    """
    https://arxiv.org/abs/2103.00020/
    """
    # Calculate logits, image similarity and text similarity
    logits = (text_embed @ img_embed.T) / temperature
    img_sim = img_embed @ img_embed.T
    text_sim = text_embed @ text_embed.T
    # Calculate targets by taking the softmax of the similarities
    targets = F.softmax(
        (img_sim + text_sim) / 2 * temperature, dim=-1
    )
    img_loss = (-targets.T * nn.LogSoftmax(dim=-1)(logits.T)).sum(1)
    text_loss = (-targets * nn.LogSoftmax(dim=-1)(logits)).sum(1)
    return (img_loss + text_loss) / 2.0

def forward(img, caption, img_encoder, img_head, text_encoder, text_head):
    # Transfer to device
    img = img.to('cuda')
    for k, v in caption.items():
        caption[k] = v.to('cuda')

    # Get embeddings for both img and caption
    img_embed = img_head(img_encoder(img))
    text_embed = text_head(text_encoder(caption))

    return img_embed, text_embed
```

In order for us to train, we will define a helper that initializes the models, tokenizer, and optimizer.

```python
def init_training_components():
    # Define image encoder, image head, text encoder, text head and a tokenizer for tokenizing the caption
    img_encoder = ImageEncoder(model_name=Config.img_encoder_model).to('cuda')
    img_head = Head(Config.img_embed_dim, Config.projection_dim).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(Config.text_encoder_model)
    text_encoder = TextEncoder(model_name=Config.text_encoder_model).to('cuda')
    text_head = Head(Config.text_embed_dim, Config.projection_dim).to('cuda')

    # Since we are optimizing two different models together, we will define parameters manually
    parameters = [
        {"params": img_encoder.parameters(), "lr": Config.img_enc_lr},
        {"params": text_encoder.parameters(), "lr": Config.text_enc_lr},
        {
            "params": itertools.chain(
                img_head.parameters(),
                text_head.parameters(),
            ),
            "lr": Config.head_lr,
        },
    ]

    optimizer = torch.optim.Adam(parameters)
    return img_encoder, img_head, text_encoder, text_head, tokenizer, optimizer
```

## Training

Before we actually train the model, one last step remains: initializing our Lance dataset and a process-safe dataloader inside a `__main__` guard so spawned worker processes do not re-run model initialization.

```python
if __name__ == "__main__":
    img_encoder, img_head, text_encoder, text_head, tokenizer, optimizer = (
        init_training_components()
    )

    # We assume the flickr8k.lance dataset is in the same directory
    dataset = CLIPLanceDataset(
        lance_path="flickr8k.lance",
        max_len=Config.max_len,
        tokenizer=tokenizer,
        transforms=train_augments,
    )

    dataloader = get_safe_loader(
        dataset,
        shuffle=False,
        batch_size=Config.bs,
        num_workers=2,
        pin_memory=True,
    )

    img_encoder.train()
    img_head.train()
    text_encoder.train()
    text_head.train()

    for epoch in range(Config.num_epochs):
        print(f"{'='*20} Epoch: {epoch+1} / {Config.num_epochs} {'='*20}")

        prog_bar = tqdm(dataloader)
        for img, caption in prog_bar:
            optimizer.zero_grad(set_to_none=True)

            img_embed, text_embed = forward(
                img, caption, img_encoder, img_head, text_encoder, text_head
            )
            loss = loss_fn(img_embed, text_embed, temperature=Config.temperature).mean()

            loss.backward()
            optimizer.step()

            prog_bar.set_description(f"loss: {loss.item():.4f}")
        print()
```

The training loop is quite self-explanatory. We set image encoder, image head, text encoder and text head models to training mode. 
Then in each epoch, we iterate over our lance dataset, training the model and reporting the lance to the progress bar.

```console
==================== Epoch: 1 / 2 ====================
loss: 2.0799: 100%|██████████| 253/253 [02:14<00:00,  1.88it/s]

==================== Epoch: 2 / 2 ====================
loss: 1.3064: 100%|██████████| 253/253 [02:10<00:00,  1.94it/s]
```

And that's basically it! Using Lance dataset for training any type of model is very similar to using any other type of dataset but it also comes with increased speed and ease of use! 
