# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import collections.abc
import pickle

import pyarrow as pa
import pytest
from lance.bitmap import Bitmap


def test_construct_from_list():
    b = Bitmap([4, 1, 2, 1])
    assert len(b) == 3
    assert set(b) == {1, 2, 4}


def test_construct_from_range():
    b = Bitmap(range(1000))
    assert len(b) == 1000
    assert 0 in b
    assert 999 in b
    assert 1000 not in b


def test_construct_empty():
    assert len(Bitmap()) == 0
    assert len(Bitmap([])) == 0


def test_construct_from_bitmap():
    original = Bitmap([1, 2, 3])
    copy = Bitmap(original)
    assert copy == original


@pytest.mark.parametrize(
    "arrow_type",
    [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
    ],
)
def test_construct_from_pyarrow_array(arrow_type):
    arr = pa.array(range(100), type=arrow_type)
    b = Bitmap(arr)
    assert len(b) == 100
    assert set(b) == set(range(100))


def test_construct_from_chunked_array():
    chunked = pa.chunked_array(
        [pa.array([1, 2, 3], type=pa.int32()), pa.array([4, 5], type=pa.int32())]
    )
    b = Bitmap(chunked)
    assert set(b) == {1, 2, 3, 4, 5}


def test_construct_from_sliced_pyarrow_array():
    # Values are read from the array's native buffer, which the slice shares
    # with the unsliced array — the slice's offset and length must be honored.
    arr = pa.array([10, 20, 30, 40], type=pa.int32()).slice(1, 2)
    assert set(Bitmap(arr)) == {20, 30}


def test_construct_from_unaligned_pyarrow_buffer():
    # A buffer arriving over the Arrow C data interface may not meet the
    # type's native alignment (here, an int32 array over a byte-offset slice).
    # That must be handled by copying, not by panicking inside `make_array`.
    raw = bytearray([0xFF, 1, 0, 0, 0])
    buffer = pa.py_buffer(raw).slice(1, 4)
    arr = pa.Array.from_buffers(pa.int32(), 1, [None, buffer])

    assert arr.to_pylist() == [1]
    assert set(Bitmap(arr)) == {1}


@pytest.mark.parametrize(
    "empty",
    [
        pa.array([], type=pa.int32()),
        pa.chunked_array([], type=pa.int32()),
        pa.chunked_array([pa.array([], type=pa.int32())]),
    ],
    ids=["array", "chunked_no_chunks", "chunked_empty_chunk"],
)
def test_construct_from_empty_pyarrow(empty):
    assert len(Bitmap(empty)) == 0


def test_construct_from_pyarrow_rejects_nulls():
    arr = pa.array([1, 2, None], type=pa.int32())
    with pytest.raises(ValueError):
        Bitmap(arr)


def test_construct_from_pyarrow_rejects_non_integer():
    arr = pa.array([1.0, 2.0], type=pa.float64())
    with pytest.raises(ValueError, match="integer pyarrow array"):
        Bitmap(arr)


@pytest.mark.parametrize("value", [-1, 2**32], ids=["negative", "too_large"])
def test_construct_rejects_out_of_range_values(value):
    # The error names the offending value, rather than being a bare
    # OverflowError from the u32 conversion.
    with pytest.raises(ValueError, match=str(value)):
        Bitmap([value])


def test_construct_accepts_max_value():
    b = Bitmap([2**32 - 1])
    assert len(b) == 1
    assert 2**32 - 1 in b


def test_construct_rejects_non_integer_values():
    with pytest.raises(TypeError):
        Bitmap(["not an int"])


@pytest.mark.parametrize(
    ("value", "arrow_type"),
    [(-1, pa.int32()), (2**33, pa.int64()), (2**64 - 1, pa.uint64())],
    ids=["negative", "too_large_int64", "too_large_uint64"],
)
def test_construct_from_pyarrow_rejects_out_of_range(value, arrow_type):
    # The offending value must be reported as itself, not silently wrapped
    # (e.g. through a signed cast) into a misleading number.
    arr = pa.array([value], type=arrow_type)
    with pytest.raises(ValueError, match=str(value)):
        Bitmap(arr)


def test_len_iter_contains():
    b = Bitmap([1, 2, 4])
    assert len(b) == 3
    assert 1 in b
    assert 3 not in b
    assert "not an int" not in b
    assert sorted(b) == [1, 2, 4]


def test_iter_is_lazy():
    """`iter()` returns a dedicated streaming iterator (not a `list_iterator`
    over a pre-built list), yielding values one at a time on demand."""
    b = Bitmap(range(1000))
    it = iter(b)
    assert type(it).__name__ == "BitmapIterator"
    assert next(it) == 0
    assert next(it) == 1
    assert list(it) == list(range(2, 1000))


def test_equality():
    assert Bitmap([1, 2, 3]) == Bitmap([3, 2, 1])
    assert Bitmap([1, 2, 3]) == {1, 2, 3}
    assert Bitmap([1, 2, 3]) != Bitmap([1, 2])
    assert Bitmap([1, 2, 3]) != {1, 2}


def test_equality_against_incompatible_type_is_false_not_error():
    # Comparing to a value that isn't a Bitmap or an iterable of ints (e.g. a
    # bare int) is simply unequal, matching normal Python `==` semantics —
    # it must not raise just because the type differs.
    b = Bitmap([1, 2, 3])
    assert (b == 5) is False
    assert (b != 5) is True
    assert (b == "not iterable") is False
    assert (b != "not iterable") is True


def test_is_registered_as_a_set():
    # `Bitmap` is not a `set` subclass, so code that must accept either has to
    # test against the abstract base class.
    b = Bitmap([1])
    assert isinstance(b, collections.abc.Set)
    assert isinstance(b, collections.abc.MutableSet)
    assert not isinstance(b, set)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (lambda a, b: a & b, {2, 3}),
        (lambda a, b: a | b, {1, 2, 3, 4}),
        (lambda a, b: a - b, {1}),
        (lambda a, b: a ^ b, {1, 4}),
    ],
    ids=["and", "or", "sub", "xor"],
)
@pytest.mark.parametrize(
    "other", [Bitmap([2, 3, 4]), {2, 3, 4}, [2, 3, 4]], ids=["bitmap", "set", "list"]
)
def test_set_operators(op, expected, other):
    result = op(Bitmap([1, 2, 3]), other)
    assert isinstance(result, Bitmap)
    assert result == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (lambda a, b: a & b, {2, 3}),
        (lambda a, b: a | b, {1, 2, 3, 4}),
        (lambda a, b: a - b, {4}),
        (lambda a, b: a ^ b, {1, 4}),
    ],
    ids=["rand", "ror", "rsub", "rxor"],
)
def test_reflected_set_operators(op, expected):
    # A plain `set` on the left returns NotImplemented for a `Bitmap` operand,
    # so the reflected form has to carry the operation.
    assert op({2, 3, 4}, Bitmap([1, 2, 3])) == expected


def test_set_operator_with_incompatible_type_raises():
    with pytest.raises(TypeError):
        Bitmap([1]) & 5


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("intersection", {2, 3}),
        ("union", {1, 2, 3, 4}),
        ("difference", {1}),
        ("symmetric_difference", {1, 4}),
    ],
)
def test_named_set_methods(method, expected):
    result = getattr(Bitmap([1, 2, 3]), method)([2, 3, 4])
    assert isinstance(result, Bitmap)
    assert result == expected


def test_subset_superset_and_disjoint():
    b = Bitmap([1, 2])
    assert b.issubset([1, 2, 3])
    assert b <= Bitmap([1, 2])
    assert b < Bitmap([1, 2, 3])
    assert not b < Bitmap([1, 2])
    assert b.issuperset([1])
    assert b >= Bitmap([1, 2])
    assert b > Bitmap([1])
    assert not b > Bitmap([1, 2])
    assert b.isdisjoint([3, 4])
    assert not b.isdisjoint([2])


def test_ordering_against_incompatible_type_raises():
    # Unlike `==`, an ordering comparison against a non-set has no meaning.
    with pytest.raises(TypeError):
        Bitmap([1]) < 5


def test_repr():
    assert repr(Bitmap([1, 2, 3])) == "Bitmap({1, 2, 3})"


def test_repr_truncates_long_bitmaps():
    assert repr(Bitmap(range(25))) == (
        "Bitmap({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, "
        "18, 19, ...}, len=25)"
    )


def test_pickle_round_trip():
    b = Bitmap(range(10_000))
    loaded = pickle.loads(pickle.dumps(b))
    assert loaded == b


def test_add_discard_update():
    b = Bitmap([1, 2, 3])
    b.add(4)
    assert 4 in b
    b.discard(2)
    assert 2 not in b
    b.discard(2)  # discarding an absent value is a no-op
    b.update([10, 11])
    assert {10, 11}.issubset(set(b))


def test_remove_pop_clear():
    b = Bitmap([1, 2, 3])
    b.remove(2)
    assert 2 not in b
    with pytest.raises(KeyError):
        b.remove(2)
    assert b.pop() == 1  # the smallest value
    b.clear()
    assert len(b) == 0
    with pytest.raises(KeyError):
        b.pop()


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__iand__", {2, 3}),
        ("__ior__", {1, 2, 3, 4}),
        ("__isub__", {1}),
        ("__ixor__", {1, 4}),
    ],
)
def test_in_place_set_operators(op, expected):
    b = Bitmap([1, 2, 3])
    shared = Bitmap(b)

    getattr(b, op)([2, 3, 4])

    assert b == expected
    # In-place mutation still copies on write, so the alias is untouched.
    assert shared == {1, 2, 3}


def test_copy_is_independent():
    original = Bitmap([1, 2, 3])
    copy = original.copy()

    copy.add(4)

    assert 4 not in original
    assert copy == {1, 2, 3, 4}


def test_mutation_is_copy_on_write():
    original = Bitmap([1, 2, 3])
    other = Bitmap(original)

    other.add(4)

    assert 4 not in original
    assert 4 in other
    assert original == Bitmap([1, 2, 3])
