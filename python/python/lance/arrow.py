# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Extensions to PyArrows."""

import json
import typing
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc

from ._arrow.bf16 import (  # noqa: F401
    BFloat16,
    BFloat16Array,
    BFloat16Type,
    PandasBFloat16Array,
)
from .dependencies import numpy as np
from .lance import bfloat16_array

__all__ = [
    "BFloat16",
    "BFloat16Array",
    "BFloat16Type",
    "PandasBFloat16Array",
    "bfloat16_array",
    "cast",
    "EncodedImageType",
    "FixedShapeImageTensorType",
    "ImageArray",
    "ImageScalar",
    "ImageURIArray",
    "ImageURIType",
]


def _is_pyarrow_string_type(t: pa.DataType) -> bool:
    # TODO: allow string_view once available?
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _is_pyarrow_binary_type(t: pa.DataType) -> bool:
    # TODO: allow binary_view once available?
    return pa.types.is_binary(t) or pa.types.is_large_binary(t)


class ImageURIType(pa.ExtensionType):
    def __init__(self, storage_type: pa.DataType = pa.string()):
        # TODO: allow string_view once available?
        if not _is_pyarrow_string_type(storage_type):
            raise ValueError("storage_type must be a string type")
        pa.ExtensionType.__init__(self, storage_type, "lance.arrow.image_uri")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ImageURIType(storage_type)

    def __arrow_ext_class__(self):
        return ImageURIArray

    def __arrow_ext_scalar_class__(self):
        return ImageURIScalar

    def __reduce__(self):
        # Workaround to ensure pickle works in earlier versions of PyArrow
        # https://github.com/apache/arrow/issues/35599
        return type(self).__arrow_ext_deserialize__, (
            self.storage_type,
            self.__arrow_ext_serialize__(),
        )


class EncodedImageType(pa.ExtensionType):
    def __init__(self, storage_type: pa.DataType = pa.binary()):
        # TODO: use pa.BinaryView once available?
        if not _is_pyarrow_binary_type(storage_type):
            raise ValueError("storage_type must be a binary type")
        pa.ExtensionType.__init__(self, storage_type, "lance.arrow.encoded_image")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return EncodedImageType(storage_type)

    def __arrow_ext_class__(self):
        return EncodedImageArray

    def __arrow_ext_scalar_class__(self):
        return EncodedImageScalar

    def __reduce__(self):
        # Workaround to ensure pickle works in earlier versions of PyArrow
        # https://github.com/apache/arrow/issues/35599
        return type(self).__arrow_ext_deserialize__, (
            self.storage_type,
            self.__arrow_ext_serialize__(),
        )


class FixedShapeImageTensorType(pa.ExtensionType):
    def __init__(self, arrow_type: pa.DataType, shape):
        self.shape = shape
        self.arrow_type = arrow_type
        assert len(shape) > 0

        length = 1
        for dim in shape:
            length *= dim

        pa.ExtensionType.__init__(
            self,
            pa.list_(arrow_type, length),
            "lance.arrow.fixed_shape_image_tensor",
        )

    def __arrow_ext_serialize__(self):
        serialized = json.dumps({"shape": self.shape}).encode()
        return serialized

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        deserialized = json.loads(serialized.decode())
        return FixedShapeImageTensorType(storage_type.value_type, deserialized["shape"])

    def __arrow_ext_class__(self):
        return FixedShapeImageTensorArray

    def __arrow_ext_scalar_class__(self):
        return FixedShapeImageTensorScalar

    def __reduce__(self):
        # Workaround to ensure pickle works in earlier versions of PyArrow
        # https://github.com/apache/arrow/issues/35599
        return type(self).__arrow_ext_deserialize__, (
            self.storage_type,
            self.__arrow_ext_serialize__(),
        )


pa.register_extension_type(ImageURIType())
pa.register_extension_type(EncodedImageType())
pa.register_extension_type(FixedShapeImageTensorType(pa.uint8(), (0,)))


class ImageArray(pa.ExtensionArray):
    def __repr__(self):
        return "<lance.arrow.%s object at 0x%016x>\n%s" % (
            type(self).__name__,
            id(self),
            repr(self.to_pylist()),
        )

    @classmethod
    def from_array(cls, images):
        """
        Create an one of subclasses of ImageArray from input data.

        Parameters
        ----------
        images : Union[pa.StringArray, pa.BinaryArray, pa.FixedShapeTensorArray,
            Iterable]

        Returns
        -------
        Union[ImageURIArray, EncodedImageArray, FixedShapeImageTensorArray]
        """

        if isinstance(images, (pa.StringArray, pa.LargeStringArray)):
            return pa.ExtensionArray.from_storage(ImageURIType(images.type), images)
        elif isinstance(images, (pa.BinaryArray, pa.LargeBinaryArray)):
            return pa.ExtensionArray.from_storage(EncodedImageType(images.type), images)
        elif isinstance(images, pa.FixedShapeTensorArray):
            shape = images.type.shape
            value_type = images.type.value_type
            typ = FixedShapeImageTensorType(value_type, shape)
            return pa.ExtensionArray.from_storage(typ, images.storage)
        elif isinstance(
            images, (ImageURIArray, EncodedImageArray, FixedShapeImageTensorArray)
        ):
            return images
        elif isinstance(images, (list, tuple, Iterable)):
            return pa.ExtensionArray.from_storage(
                ImageURIType(), pa.array(images, type=pa.string())
            )

        else:
            raise TypeError("Cannot build a ImageArray from {}".format(type(images)))


class ImageURIArray(ImageArray):
    """
    Array of image URIs. URIs may represent local files or remote files on the web,
    S3 or GCS if they are accessible by the machine executing this.
    """

    @classmethod
    def from_uris(
        cls,
        uris: Union[pa.StringArray, pa.LargeStringArray, Iterable[Union[str, Path]]],
    ):
        """
        Create an ImageURIArray from an array or iterable of URIs (such as a list).

        Parameters
        ----------
        uris : Union[pa.StringArray, pa.LargeStringArray, Iterable[Union[str, Path]]]

        Returns
        -------
        ImageURIArray
            Array of image URIs

        Examples
        --------
        >>> uris = ["file::///tmp/1.png"]
        >>> ImageURIArray.from_uris(uris)
        <lance.arrow.ImageURIArray object at 0x...>
        ['file::///tmp/1.png']
        """
        storage: pa.Array
        if isinstance(uris, (pa.StringArray, pa.LargeStringArray)):
            storage = uris
        elif isinstance(uris, Iterable):
            storage = pa.array((str(uri) for uri in uris), type=pa.string())
        else:
            raise TypeError("Cannot build a ImageURIArray from {}".format(type(uris)))

        return cls.from_storage(ImageURIType(storage.type), storage)

    def read_uris(self, storage_type=pa.binary()) -> "EncodedImageArray":
        """
        Read the images from the URIs into memory and return an EncodedImageArray

        Parameters
        ----------
        storage_type : pa.DataType, optional
            The storage type to use for the encoded images. Default is pa.binary().
            To support arrays with more than 2GiB of data, use pa.large_binary().

        Returns
        -------
        EncodedImageArray
            Array of encoded images

        Examples
        --------
        >>> import os
        >>> uris = [os.path.join(os.path.dirname(__file__), "../tests/images/1.png")]
        >>> uri_array = ImageURIArray.from_uris(uris)
        >>> uri_array.read_uris()
        <lance.arrow.EncodedImageArray object at 0x...>
        ...
        """
        from urllib.error import URLError
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        from pyarrow import fs

        def download(url):
            req = Request(url)
            try:
                return urlopen(req).read()
            except URLError as e:
                if hasattr(e, "reason"):
                    print("Failed to reach the server: ", e.reason)
                elif hasattr(e, "code"):
                    print(
                        "The server could not fulfill the request. Error code: ",
                        getattr(e, "code"),
                    )

        images = []
        for uri in self.storage:
            parsed_uri = urlparse(uri.as_py())
            if parsed_uri.scheme in ("http", "https"):
                images.append(download(uri))
            else:
                filesystem, path = fs.FileSystem.from_uri(  # pyright: ignore[reportPrivateImportUsage]
                    uri.as_py()
                )
                with filesystem.open_input_stream(path) as f:
                    images.append(f.read())

        return EncodedImageArray.from_storage(
            EncodedImageType(storage_type), pa.array(images, type=storage_type)
        )


class EncodedImageArray(ImageArray):
    """
    Array of encoded images. Images may be encoded in any format. Format is not stored
    by the array but can typically be inferred from the image bytes. Alternatively a
    separate record of encoding can be kept in a separate array outside this library.
    """

    def __repr__(self):
        def pillow_metadata_decoder(images):
            import io

            from PIL import Image  # pyright: ignore[reportMissingImports]

            img = Image.open(io.BytesIO(images[0].as_py()))
            return img

        decoders = (("PIL", pillow_metadata_decoder),)
        decoder = None

        for libname, metadata_decoder in decoders:
            try:
                __import__(libname)
                decoder = metadata_decoder
                break
            except ImportError:
                pass

        return "<lance.arrow.%s object at 0x%016x>%s\n%s" % (
            type(self).__name__,
            id(self),
            "\n[" + repr(decoder(self)) + ", ..]" if decoder else None,
            repr(self.to_pylist()[0][:30]),
        )

    def to_tensor(
        self,
        decoder: Optional[
            Callable[[Union[pa.BinaryArray, pa.LargeBinaryArray]], np.ndarray]
        ] = None,
    ):
        """
        Decode encoded images and return a FixedShapeImageTensorArray

        Parameters
        ----------
        decoder : Callable[pa.binary()], optional
            A function that takes a binary array and returns a numpy.ndarray
            or pa.fixed_shape_tensor. If not provided, will attempt to use
            pillow.

        Returns
        -------
        FixedShapeImageTensorArray
            Array of images as tensors

        Examples
        --------
        >>> import os
        >>> uris = [os.path.join(os.path.dirname(__file__), "../tests/images/1.png")]
        >>> encoded_image_array = ImageURIArray.from_uris(uris).read_uris()
        >>> encoded_image_array.to_tensor()
        <lance.arrow.FixedShapeImageTensorArray object at 0x...>
        [[42, 42, 42, 255]]
        """

        if not hasattr(pa, "FixedShapeTensorType"):
            raise NotImplementedError("This function requires PyArrow >= 12.0.0")

        if not decoder:

            def pillow_decoder(images) -> "np.ndarray":
                import io

                from PIL import Image  # pyright: ignore[reportMissingImports]

                return np.stack(
                    [
                        np.asarray(Image.open(io.BytesIO(img)))
                        for img in images.to_pylist()
                    ]
                )

            decoders = [("PIL", pillow_decoder)]
            for libname, decoder_function in decoders:
                try:
                    __import__(libname)
                    decoder = decoder_function
                    break
                except ImportError:
                    pass
            else:
                raise ValueError(
                    "No image decoder available. Please install pillow or pass a "
                    "decoder argument."
                )

        image_array = decoder(self.storage)
        if isinstance(image_array, pa.FixedShapeTensorType):
            tensor = typing.cast("pa.Array", image_array)
            shape = tensor.shape
            arrow_type = tensor.storage_type
            tensor_array = tensor
        else:
            shape = image_array.shape[1:]
            arrow_type = pa.from_numpy_dtype(image_array.dtype)
            tensor_array = pa.FixedShapeTensorArray.from_numpy_ndarray(image_array)

        return pa.ExtensionArray.from_storage(
            FixedShapeImageTensorType(arrow_type, shape), tensor_array.storage
        )


# TODO: add VariableShapeImageTensorType once pa.VariableShapeTensorArray is available
class FixedShapeImageTensorArray(ImageArray):
    """
    Array of fixed shape tensors representing image pixels.
    """

    def to_numpy(self):
        """
        Convert FixedShapeImageTensorArray to a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            Array of images

        Examples
        --------

        >>> import os
        >>> uris = [os.path.join(os.path.dirname(__file__), "../tests/images/1.png")]
        >>> tensor_image_array = ImageURIArray.from_uris(uris).read_uris().to_tensor()
        >>> tensor_image_array.to_numpy()
        array([[[[ 42,  42,  42, 255]]]], dtype=uint8)
        """
        ext_type = pa.fixed_shape_tensor(self.storage.type.value_type, self.type.shape)
        tensor_array = pa.ExtensionArray.from_storage(ext_type, self.storage)
        return tensor_array.to_numpy_ndarray()

    def to_encoded(self, encoder=None, storage_type=pa.binary()) -> "EncodedImageArray":
        """
        Encode FixedShapeImageTensorArray to PNG bytes and return an EncodedImageArray.

        Parameters
        ----------
        encoder : Callable[np.ndarray], optional
            An encoder function that takes a numpy.ndarray and returns an encoded image.
        storage_type : pa.DataType, optional
            The storage type to use for the encoded images. Default is pa.binary().
            To support arrays with more than 2GiB of data, use pa.large_binary().

        Returns
        -------
        EncodedImageArray
            Array of encoded images

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([[[[42, 42, 42, 255]]]], dtype=np.uint8)
        >>> arrow_type = pa.from_numpy_dtype(arr.dtype)
        >>> shape = arr.shape[1:]
        >>> tensor_array = pa.FixedShapeTensorArray.from_numpy_ndarray(arr)
        >>> tensor_image_array = FixedShapeImageTensorArray.from_storage(
        ... FixedShapeImageTensorType(arrow_type, shape), tensor_array.storage)
        >>> tensor_image_array.to_encoded()
        <lance.arrow.EncodedImageArray object at 0x...>
        ...
        """

        def pillow_encoder(x):
            import io

            from PIL import Image  # pyright: ignore[reportMissingImports]

            encoded_images = []
            for y in x:
                with io.BytesIO() as buf:
                    Image.fromarray(y).save(buf, format="PNG")
                    encoded_images.append(buf.getvalue())
            return pa.array(encoded_images, type=storage_type)

        if not encoder:
            encoders = (("PIL", pillow_encoder),)
            for libname, encoder_function in encoders:
                try:
                    __import__(libname)
                    encoder = encoder_function
                    break
                except ImportError:
                    pass
            else:
                raise ValueError(
                    "No image encoder available. Please install pillow or pass an "
                    "encoder argument."
                )

        return EncodedImageArray.from_storage(
            EncodedImageType(storage_type), encoder(self.to_numpy())
        )


class ImageScalar(pa.ExtensionScalar):
    def as_py(self, **kwargs):
        return self.value.as_py(**kwargs)


class ImageURIScalar(ImageScalar):
    pass


class EncodedImageScalar(ImageScalar):
    pass


class FixedShapeImageTensorScalar(ImageScalar):
    pass


# TODO: Deprecate this method once the upstream work (apache/arrow#33103)
# is merged and released.
def cast(
    arr: pa.Array, target_type: Union[pa.DataType, str], *args, **kwargs
) -> pa.Array:
    """Cast an array to another data type.

    Extends :meth:`pyarrow.compute.cast` for lance defined extension types.
    In case where the casting can be handled by pyarrow natively, it falls back
    to pyarrow.

    Supported operations:

    - Cast between floating (``float16``, ``float32``, ``float64``) arrays
      and ``bfloat16`` arrays.
    - Cast between FixedSizeListArray of floats (``float16``, ``float32``, ``float64``,
      ``bfloat16``) with the same list size.

    Parameters
    ----------
    arr : pyarrow.Array
        Array to cast.
    target_type : pyarrow.DataType or str
        Target data type. Accepts anything :meth:`pyarrow.compute.cast` accepts.
        Additionally, accepts strings ``"bfloat16"``, ``"bf16"`` or
        :py:class:`~lance._arrow.bf16.BFloat16Type`.

    """
    from ml_dtypes import bfloat16

    if isinstance(arr.type, BFloat16Type):
        # Casting bf16 to other float types
        if not pa.types.is_floating(target_type):
            raise ValueError(
                "Only support casting bfloat16 array to floating array,"
                + f"got: {target_type}"
            )
        np_arr = arr.to_numpy()
        float_type = typing.cast("pa.DataType", target_type)
        float_arr = np_arr.astype(float_type.to_pandas_dtype())
        return pa.array(float_arr)
    elif isinstance(target_type, BFloat16Type) or target_type in ["bfloat16", "bf16"]:
        if not pa.types.is_floating(arr.type):
            raise ValueError(
                "Only support casting floating array to bfloat16 array,"
                + f"got: {arr.type}"
            )
        np_arr = arr.to_numpy()
        bf16_arr = np_arr.astype(bfloat16)
        return BFloat16Array.from_numpy(bf16_arr)
    elif pa.types.is_fixed_size_list(arr.type) and pa.types.is_fixed_size_list(
        target_type
    ):
        # Casting fixed size list to fixed size list
        list_type = typing.cast("pa.DataType", target_type)
        if arr.type.list_size != list_type.list_size:
            raise ValueError(
                "Only support casting fixed size list to fixed size list "
                f"with the same size, got: {arr.type} to {target_type}"
            )
        values = cast(arr.values, list_type.value_type)
        return pa.FixedSizeListArray.from_arrays(
            values=values, list_size=list_type.list_size
        )

    # Fallback to normal cast.
    return pc.cast(arr, target_type, *args, **kwargs)
