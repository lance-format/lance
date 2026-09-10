# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import pickle
import re
from pathlib import Path

import lance
import lance.arrow
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest  # pyright: ignore[reportMissingImports]
from lance.arrow import (
    BFloat16,
    BFloat16Array,
    FixedShapeImageTensorType,
    ImageArray,
    ImageURIArray,
    PandasBFloat16Array,
    bfloat16_array,
)
from lance.file import LanceFileReader
from ml_dtypes import bfloat16


def test_bf16_value():
    val = BFloat16(1.124565)
    assert val == BFloat16(1.124565)
    assert str(val) == "1.125"

    should_be_true = [
        BFloat16(1.1) > BFloat16(1.0),
        BFloat16(1.1) >= BFloat16(1.0),
        BFloat16(1.1) != BFloat16(1.0),
        BFloat16(1.0) < BFloat16(1.1),
        BFloat16(1.0) <= BFloat16(1.1),
    ]
    assert all(comparison for comparison in should_be_true)

    should_be_false = [
        BFloat16(1.1) < BFloat16(1.0),
        BFloat16(1.1) <= BFloat16(1.0),
        BFloat16(1.1) == BFloat16(1.0),
        BFloat16(1.0) >= BFloat16(1.1),
        BFloat16(1.0) >= BFloat16(1.1),
    ]
    assert not any(comparison for comparison in should_be_false)


def test_bf16_from_bytes_roundtrip():
    assert BFloat16.from_bytes(b"\xc0\x3f") == BFloat16(1.5)


@pytest.mark.parametrize("bad", [b"", b"\x00", b"\x00\x00\x00", b"\x00" * 4])
def test_bf16_from_bytes_invalid_length_raises(bad):
    with pytest.raises(ValueError, match="expected 2 bytes"):
        BFloat16.from_bytes(bad)


def test_bf16_repr():
    data = [1.1, None, 3.4]
    arr = bfloat16_array(data)
    assert isinstance(arr, BFloat16Array)

    expected = [BFloat16(1.1), None, BFloat16(3.4)]
    assert arr[0].as_py() == expected[0]
    assert arr[1].as_py() == expected[1]
    assert arr[2].as_py() == expected[2]

    assert arr.to_pylist() == expected

    expected_re = r"""<lance.arrow.BFloat16Array object at 0x[\w\d]+>
\[
  1.1015625,
  None,
  3.40625
\]"""
    assert re.match(expected_re, repr(arr))

    # TODO: uncomment tests once fixed upstream.
    # https://github.com/apache/arrow/issues/36648


#     tab = pa.table({"x": arr})
#     expected_re = r"""pyarrow.Table
# x: extension<lance.bfloat16<BFloat16Type>>
# ---
# x: \[\[1.1015625, None, 3.40625\]\]"""
#     assert re.match(expected_re, repr(tab))


def test_bf16_array_str():
    from ml_dtypes import bfloat16

    np_arr = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
    arr = BFloat16Array.from_numpy(np_arr)
    assert str(arr) == "[\n  1,\n  2,\n  3\n]"


def test_bf16_pandas(provide_pandas):
    data = [1.1, None, 3.4]
    arr = bfloat16_array(data)
    arr_pandas = arr.to_pandas()
    assert arr_pandas[0] == BFloat16(1.1)
    assert arr_pandas[1] is None
    assert arr_pandas[2] == BFloat16(3.4)

    # Can instantiate with dtype string
    series = pd.Series(arr_pandas, dtype="lance.bfloat16")
    pd.testing.assert_series_equal(arr_pandas, series)

    # Can roundtrip to Arrow
    arr_arrow = pa.array(arr_pandas)
    assert isinstance(arr_arrow, BFloat16Array)
    assert arr == arr_arrow

    pd.testing.assert_series_equal(arr_arrow.to_pandas(), arr_pandas)


def test_bf16_numpy():
    import numpy as np
    from ml_dtypes import bfloat16

    data = [1.1, 2.1, 3.4]
    arr = bfloat16_array(data)
    arr_numpy = arr.to_numpy()

    expected = np.array(data, dtype=bfloat16)

    np.testing.assert_array_equal(arr_numpy, expected)

    # Can roundtrip to Pandas
    arr_pandas = PandasBFloat16Array.from_numpy(arr_numpy)
    np.testing.assert_array_equal(arr_pandas.to_numpy(), expected)

    # Can roundtrip to Arrow
    arr_arrow = BFloat16Array.from_numpy(arr_numpy)
    assert arr == arr_arrow
    np.testing.assert_array_equal(arr_arrow.to_numpy(), expected)


def test_bf16_array_cast():
    for dt in [np.float16, np.float32, np.float64]:
        floats = pa.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=dt))
        bf16_arr = lance.arrow.cast(floats, "bfloat16")
        assert isinstance(bf16_arr, BFloat16Array)
        assert bf16_arr[0] == BFloat16(1.0)
        assert bf16_arr[0] == bfloat16(1.0)

        casted = lance.arrow.cast(bf16_arr, floats.type)
        assert casted == floats


def test_bf16_fixed_size_list_cast():
    for dt in [np.float16, np.float32, np.float64]:
        floats = pa.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=dt))
        fsl = pa.FixedSizeListArray.from_arrays(floats, 2)
        bf16_fsl = lance.arrow.cast(fsl, pa.list_(lance.arrow.BFloat16Type(), 2))
        assert bf16_fsl.values == BFloat16Array.from_numpy(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=bfloat16)
        )

        casted = lance.arrow.cast(bf16_fsl, fsl.type)
        assert casted == fsl


def test_bf16_roundtrip(tmp_path: Path):
    import numpy as np
    from ml_dtypes import bfloat16

    values = BFloat16Array.from_numpy(np.random.random(9).astype(bfloat16))
    vectors = pa.FixedSizeListArray.from_arrays(values, 3)
    tensors = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(values.type, [3]), vectors
    )
    data = pa.table(
        {
            "values": values.slice(0, 3),
            "vector": vectors,
            "tensors": tensors,
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    assert ds.schema == data.schema
    assert ds.to_table() == data


def test_roundtrip_take_ext_types(tmp_path: Path):
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 3])
    inner = pa.array([float(x) for x in range(0, 18)], pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(inner, 6)
    tensor_arr = pa.ExtensionArray.from_storage(tensor_type, storage)

    tbl = pa.Table.from_arrays([tensor_arr], ["tensor"])
    lance.write_dataset(tbl, tmp_path)

    tbl2 = lance.dataset(tmp_path)
    rows = tbl2.take([0, 2])
    assert rows["tensor"].to_pylist() == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    ]


def test_struct_variable_children_roundtrip(tmp_path: Path):
    struct_type = pa.struct(
        [
            pa.field("id", pa.int32()),
            pa.field("text", pa.utf8()),
            pa.field("payload", pa.binary()),
        ]
    )
    struct_field = pa.field(
        "record",
        struct_type,
        metadata={"lance-encoding:packed": "true"},
    )
    struct_array = pa.StructArray.from_arrays(
        [
            pa.array([1, 2, 3, 4], pa.int32()),
            pa.array(["alpha", "beta", "gamma", "delta"], pa.utf8()),
            pa.array(
                [b"\x01\x02", b"\xff", b"\x03\x04\x05", b"\x10"],
                type=pa.binary(),
            ),
        ],
        type=struct_type,
    )
    table = pa.Table.from_arrays([struct_array], schema=pa.schema([struct_field]))

    dataset_uri = tmp_path / "struct.lance"
    ds = lance.write_dataset(table, dataset_uri, data_storage_version="2.2")

    round_trip = ds.to_table()
    assert round_trip.schema == table.schema
    assert round_trip.equals(table)
    assert round_trip.to_pylist() == table.to_pylist()

    data_file = next((dataset_uri / "data").glob("*.lance"))
    metadata = LanceFileReader(str(data_file)).metadata()
    encodings = [page.encoding for page in metadata.columns[0].pages]
    assert any("VariablePackedStruct" in encoding for encoding in encodings)


@pytest.fixture
def png_uris():
    local_path = "file://" + os.path.join(os.path.dirname(__file__), "images/1.png")
    png_uris = [local_path] * 10
    if os.name == "nt":
        png_uris = [str(Path(x)) for x in png_uris]
    return png_uris


def test_image_uri_arrays(tmp_path: Path, png_uris):
    from_list = ImageURIArray.from_uris(png_uris)
    from_pyarrow = ImageURIArray.from_uris(pa.array(png_uris, pa.string()))
    from_pyarrow_large = ImageURIArray.from_uris(pa.array(png_uris, pa.large_string()))
    for arr in [from_list, from_pyarrow, from_pyarrow_large]:
        assert arr.to_pylist() == png_uris

    image_array = ImageURIArray.from_uris(png_uris)
    assert ImageArray.from_array(image_array) == image_array

    encoded_image_array = image_array.read_uris()
    assert len(ImageArray.from_array(encoded_image_array.storage)) == 10
    assert ImageArray.from_array(encoded_image_array) == encoded_image_array

    large_array = encoded_image_array.storage.cast(pa.large_binary())
    large_array = ImageArray.from_array(large_array)
    assert large_array.to_pylist() == encoded_image_array.to_pylist()


def test_image_tensor_arrays(tmp_path: Path, png_uris):
    n = 10

    encoded_image_array = ImageURIArray.from_uris(png_uris).read_uris()

    tensor_image_array = encoded_image_array.to_tensor()
    fixed_shape_tensor_array = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(
            tensor_image_array.storage.type.value_type, tensor_image_array.type.shape
        ),
        tensor_image_array.storage,
    )
    assert (
        ImageArray.from_array(fixed_shape_tensor_array).storage
        == fixed_shape_tensor_array.storage
    )
    assert ImageArray.from_array(tensor_image_array) == tensor_image_array

    assert len(ImageArray.from_array(tensor_image_array)) == n
    assert len(tensor_image_array) == n
    assert tensor_image_array.storage.type == pa.list_(pa.uint8(), 4)
    assert tensor_image_array[2].as_py() == [42, 42, 42, 255]

    test_tensor = np.array([42, 42, 42, 255] * n, dtype=np.uint8).reshape((n, 1, 1, 4))

    assert test_tensor.shape == (n, 1, 1, 4)
    assert np.array_equal(tensor_image_array.to_numpy(), test_tensor)
    assert tensor_image_array.to_encoded().to_tensor() == tensor_image_array

    def png_encoder(images):
        import io

        from PIL import Image  # pyright: ignore[reportMissingImports]

        encoded_images = []
        for image in images:
            with io.BytesIO() as buf:
                Image.fromarray(image).save(buf, format="PNG")
                encoded_images.append(buf.getvalue())
        return pa.array(encoded_images, type=pa.binary())

    assert tensor_image_array.to_encoded(png_encoder).to_tensor() == tensor_image_array
    uris = [
        "file://" + os.path.join(os.path.dirname(__file__), "images/1.png"),
        "file://" + os.path.join(os.path.dirname(__file__), "images/2.jpeg"),
    ]
    if os.name == "nt":
        uris = [str(Path(x)) for x in uris]

    encoded_image_array = ImageArray.from_array(uris).read_uris()
    with pytest.raises(ValueError, match="all input arrays must have the same shape"):
        encoded_image_array.to_tensor()

    pattern = r"(object at) 0x[\w\d]+(:?>)"
    repl = r"\1 0x..\2"
    repr_ = re.sub(pattern, repl, encoded_image_array.__repr__())
    assert repr_.startswith("<lance.arrow.EncodedImageArray object at 0x..>\n[<PIL.")
    assert "image mode=RGBA size=1x1" in repr_
    assert (
        "b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00"
        "\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f'"
    ) in repr_


def test_roundtrip_image_tensor(tmp_path: Path):
    import os

    png_uris = [
        os.path.join(os.path.dirname(__file__), "images/1.png"),
    ] * 5
    uri_array = ImageURIArray.from_uris(png_uris)
    encoded_image_array = uri_array.read_uris()
    tensor_image_array = encoded_image_array.to_tensor()

    tbl = pa.Table.from_arrays(
        [uri_array, encoded_image_array, tensor_image_array],
        ["uri_array", "encoded_image_array", "tensor_image_array"],
    )
    lance.write_dataset(tbl, tmp_path)
    tbl2 = lance.dataset(tmp_path)
    indices = list(range(len(png_uris)))

    assert tbl.take(indices).to_pylist() == tbl2.take(indices).to_pylist()
    tensor_image_array_2 = tbl2.take(indices).column(2)

    assert tensor_image_array_2.type == tensor_image_array.type


def test_image_array_pickle(tmp_path: Path, png_uris):
    # Note: this test will only fail in PyArrow 12.0.0. It was fixed in 13.0.0.
    uri_array = ImageURIArray.from_uris(png_uris)
    encoded_array = uri_array.read_uris()
    tensor_array = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.random.random((10, 1, 1, 4)).astype(np.uint8)
    )
    tensor_array = pa.ExtensionArray.from_storage(
        FixedShapeImageTensorType(pa.uint8(), tensor_array.type.shape),
        tensor_array.storage,
    )

    arrays = [uri_array, encoded_array, tensor_array]
    for arr in arrays:
        data = pickle.dumps(arr)
        arr2 = pickle.loads(data)
        assert arr == arr2
