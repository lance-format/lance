# Extension Arrays

Lance provides extensions for Arrow arrays and Pandas Series to represent data
types for machine learning applications.

## BFloat16

[BFloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
is a 16-bit floating point number that is designed for machine learning use cases.
Intuitively, it only has 2-3 digits of precision, but it has the same range as
a 32-bit float: ~1e-38 to ~1e38. By comparison, a 16-bit float has a range of
~5.96e-8 to 65504.

Lance provides an Arrow extension array (`lance.arrow.BFloat16Array`)
and a Pandas extension array (`lance._arrow.PandasBFloat16Type`) for BFloat16.
These are compatible with the [ml_dtypes](https://github.com/jax-ml/ml_dtypes)
bfloat16 NumPy extension array.

If you are using Pandas, you can use the `lance.bfloat16` dtype string to create
the array:

```python
import lance.arrow

pd.Series([1.1, 2.1, 3.4], dtype="lance.bfloat16")
# 0    1.1015625
# 1      2.09375
# 2      3.40625
# dtype: lance.bfloat16
```

To create an Arrow array, use the `lance.arrow.bfloat16_array` function:

```python
from lance.arrow import bfloat16_array

bfloat16_array([1.1, 2.1, 3.4])
# <lance.arrow.BFloat16Array object at 0x000000016feb94e0>
# [
#   1.1015625,
#   2.09375,
#   3.40625
# ]
```

Finally, if you have a pre-existing NumPy array, you can convert it into either:

```python
import numpy as np
from ml_dtypes import bfloat16
from lance.arrow import PandasBFloat16Array, BFloat16Array

np_array = np.array([1.1, 2.1, 3.4], dtype=bfloat16)
PandasBFloat16Array.from_numpy(np_array)
# <PandasBFloat16Array>
# [1.1015625, 2.09375, 3.40625]
# Length: 3, dtype: lance.bfloat16
BFloat16Array.from_numpy(np_array)
# <lance.arrow.BFloat16Array object at 0x...>
# [
#   1.1015625,
#   2.09375,
#   3.40625
# ]
```

When reading, these can be converted back to to the NumPy bfloat16 dtype using
each array class's `to_numpy` method.

## ImageURI

`lance.arrow.ImageURIArray` is an array that stores the URI location of images
in some other storage system. For example, `file:///path/to/image.png` for a local
filesystem or `s3://bucket/path/image.jpeg` for an image on AWS S3. Use this
array type when you want to lazily load images from an existing storage medium.

It can be created by calling `lance.arrow.ImageURIArray.from_uris`
with a list of URIs represented by either `pyarrow.StringArray` or an
iterable that yields strings. Note that the URIs are not strongly validated and images
are not read into memory automatically.

```python
from lance.arrow import ImageURIArray

ImageURIArray.from_uris([
   "/tmp/image1.jpg",
   "file:///tmp/image2.jpg",
   "s3://example/image3.jpg"
])
# <lance.arrow.ImageURIArray object at 0x...>
# ['/tmp/image1.jpg', 'file:///tmp/image2.jpg', 's3://example/image3.jpg']
```

`lance.arrow.ImageURIArray.read_uris` will read images into memory and return
them as a new `lance.arrow.EncodedImageArray` object.

```python
from lance.arrow import ImageURIArray

relative_path = "images/1.png"
uris = [os.path.join(os.path.dirname(__file__), relative_path)]
ImageURIArray.from_uris(uris).read_uris()
# <lance.arrow.EncodedImageArray object at 0x...>
# [b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00...']
```

## EncodedImage

`lance.arrow.EncodedImageArray` is an array that stores jpeg and png images in
their encoded and compressed representation as they would appear written on disk.
Use this array when you want to manipulate images in their compressed format such as
when you're reading them from disk or embedding them into HTML.

It can be created by calling `lance.arrow.ImageURIArray.read_uris` on an existing
`lance.arrow.ImageURIArray`. This will read the referenced images into memory.
It can also be created by calling `lance.arrow.ImageArray.from_array` and passing
it an array of encoded images already read into `pyarrow.BinaryArray` or by
calling `lance.arrow.ImageTensorArray.to_encoded`.

A `lance.arrow.EncodedImageArray.to_tensor` method is provided to decode
encoded images and return them as `lance.arrow.FixedShapeImageTensorArray`, from
which they can be converted to numpy arrays.
For decoding images, it will first attempt to use a decoder provided via the optional
function parameter. If decoder is not provided it will attempt to use
[Pillow](https://pillow.readthedocs.io/en/stable/). If neither Pillow nor a custom
decoder is available an exception will be raised.

```python
from lance.arrow import ImageURIArray

uris = [os.path.join(os.path.dirname(__file__), "images/1.png")]
encoded_images = ImageURIArray.from_uris(uris).read_uris()
print(encoded_images.to_tensor())

def pillow_decoder(images):
    import io
    import numpy as np
    from PIL import Image

    return np.stack(
        np.asarray(Image.open(io.BytesIO(img.as_py()))) for img in images.storage
    )

print(encoded_images.to_tensor(pillow_decoder))
# <lance.arrow.FixedShapeImageTensorArray object at 0x...>
# [[42, 42, 42, 255]]
# <lance.arrow.FixedShapeImageTensorArray object at 0x...>
# [[42, 42, 42, 255]]
```

## FixedShapeImageTensor

`lance.arrow.FixedShapeImageTensorArray` is an array that stores images as
tensors where each individual pixel is represented as a numeric value. Typically images
are stored as 3 dimensional tensors shaped (height, width, channels). In color images
each pixel is represented by three values (channels) as per
[RGB color model](https://en.wikipedia.org/wiki/RGB_color_model).
Images from this array can be read out as numpy arrays individually or stacked together
into a single 4 dimensional numpy array shaped (batch_size, height, width, channels).

It can be created by calling `lance.arrow.EncodedImageArray.to_tensor` on a
previously existing `lance.arrow.EncodedImageArray`. This will decode encoded
images and return them as a `lance.arrow.FixedShapeImageTensorArray`. It can also be
created by calling `lance.arrow.ImageArray.from_array` and passing in a
`pyarrow.FixedShapeTensorArray`.

It can be encoded into to `lance.arrow.EncodedImageArray` by calling
`lance.arrow.FixedShapeImageTensorArray.to_encoded` and passing custom encoder
If encoder is not provided it will attempt to use
[Pillow](https://pillow.readthedocs.io/en/stable/). The default encoder will encode
to PNG. If neither Pillow nor a custom encoder is available it will raise an exception.

```python
from lance.arrow import ImageURIArray

uris = [image_uri]
tensor_images = ImageURIArray.from_uris(uris).read_uris().to_tensor()
tensor_images.to_encoded()
# <lance.arrow.EncodedImageArray object at 0x...>
# [...
# b'\x89PNG\r\n\x1a...'
```
