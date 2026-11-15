# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Bfloat16 support."""

from typing import Any, Optional, Union

import pyarrow as pa

from lance.dependencies import numpy as np

from ..lance import BFloat16
from ..lance import bfloat16_array as bfloat16_array


class BFloat16Array(pa.ExtensionArray):
    """Bfloat16 PyArrow Array."""

    def __repr__(self):
        return "<lance.arrow.BFloat16Array object at 0x%016x>\n%s" % (
            id(self),
            str(self),
        )

    def __str__(self):
        if len(self) < 22:
            return "[\n" + ",\n".join([f"  {v}" for v in self]) + "\n]"
        return (
            "[\n"
            + ",\n".join([f"  {v}" for v in self[:10]])
            + ",\n...\n"
            + ",\n".join([f"  {v}" for v in self[-10:]])
            + "\n]"
        )

    def to_numpy(self, zero_copy_only=False):
        """Convert to a NumPy array.

        This will do a zero-copy conversion.

        The conversion will fail if the array contains null values."""
        if self.null_count > 0:
            raise ValueError("Cannot convert null values to numpy")

        from ml_dtypes import bfloat16

        storage = self.storage
        buffer = storage.buffers()[1]
        array = np.frombuffer(
            buffer, dtype=bfloat16, count=len(storage), offset=storage.offset * 2
        )

        return array

    @classmethod
    def from_numpy(cls, array: np.ndarray):
        """Create a :class:`BFloat16Array` from a NumPy array.

        Can only convert from a NumPy array of dtype ``bfloat16`` from the ``ml_dtypes``
        module.

        Examples
        --------

        >>> import numpy as np
        >>> from ml_dtypes import bfloat16
        >>> from lance.arrow import BFloat16Array
        >>> arr = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
        >>> print(BFloat16Array.from_numpy(arr))
        [
          1,
          2,
          3
        ]
        """
        from ml_dtypes import bfloat16

        if array.dtype != bfloat16:
            raise ValueError("Cannot convert non-bfloat16 values to BFloat16Array")
        if array.ndim != 1:
            raise ValueError("Cannot convert multi-dimensional array to BFloat16Array")
        data = pa.py_buffer(array.tobytes())
        return pa.Array.from_buffers(BFloat16Type(), len(array), [None, data])


class BFloat16Scalar(pa.ExtensionScalar):
    def as_py(self, **kwargs) -> Optional[BFloat16]:
        if self.value is None:
            return None
        else:
            return BFloat16.from_bytes(self.value.as_py(**kwargs))

    def __eq__(self, other: Any):
        from ml_dtypes import bfloat16

        if isinstance(other, BFloat16):
            return self.as_py() == other
        elif isinstance(other, BFloat16Scalar):
            return self.as_py() == other.as_py()
        elif isinstance(other, bfloat16):
            return self.as_py() == BFloat16.from_bytes(other.tobytes())
        else:
            return False


class BFloat16Type(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.binary(2), "lance.bfloat16")

    def __arrow_ext_serialize__(self):
        # TODO: encode endianness
        return b""

    @classmethod
    def __arrow_ext_deserialize__(self, storage_type, serialized):
        # TODO: decode endianness
        return BFloat16Type()

    def __arrow_ext_class__(self):
        return BFloat16Array

    def __arrow_ext_scalar_class__(self):
        return BFloat16Scalar

    def to_pandas_dtype(self):
        return PandasBFloat16Type()


try:
    from pandas.api.extensions import (
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
    )
except ImportError:
    pass
else:
    # Define Pandas and register Pandas extensions
    @register_extension_dtype
    class PandasBFloat16Type(ExtensionDtype):
        kind = "f"
        na_value = None
        name = "lance.bfloat16"
        names = None
        type = BFloat16
        _is_numeric = True

        def __from_arrow__(
            self, array: Union[pa.Array, pa.ChunkedArray]
        ) -> ExtensionArray:
            return PandasBFloat16Array(array)

        def construct_array_type(self):
            return PandasBFloat16Array

        @classmethod
        def construct_from_string(cls, string):
            if string == "lance.bfloat16":
                return cls()
            else:
                raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    class PandasBFloat16Array(ExtensionArray):
        dtype = PandasBFloat16Type()

        def __init__(self, data):
            self.data = data

        @classmethod
        def _from_sequence(
            cls, scalars, *, dtype: PandasBFloat16Type, copy: bool = False
        ):
            return PandasBFloat16Array(bfloat16_array(scalars))

        def __getitem__(self, item):
            if isinstance(item, int):
                return self.data[item].as_py()
            elif isinstance(item, slice):
                return PandasBFloat16Array(self.data[item])
            elif isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.bool_):
                return PandasBFloat16Array(self.data.filter(pa.array(item)))
            else:
                raise NotImplementedError()

        def __len__(self):
            return len(self.data)

        def isna(self):
            return self.data.is_null().to_numpy(zero_copy_only=False)

        def to_numpy(self, *args, **kwargs):
            return self.data.to_numpy()

        def __arrow_array__(self, type=None):
            return self.data

        @classmethod
        def from_numpy(cls, array):
            inner = BFloat16Array.from_numpy(array)
            return cls(inner)


pa.register_extension_type(BFloat16Type())
