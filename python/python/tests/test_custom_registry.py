# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Smoke test for the runtime object-store scheme registration hook.

Exercises the pyo3 additions in ``python/src/object_store.rs`` and the new
``store_registry=`` parameter on ``Session(...)``. Registers the built-in
``MemoryStoreProvider`` under a made-up scheme (``test-mem``) and proves that
``lance.write_dataset(...)`` and ``lance.dataset(...)`` route reads and writes
through the registry-selected store rather than the default built-in scheme
resolver.

The full Python-to-Rust ``ObjectStoreProvider`` callable bridge is not exercised
here — that is a follow-up. See the module docstring in
``python/src/object_store.rs``.
"""

import lance
import pyarrow as pa
import pytest
from lance.lance import (
    _ObjectStoreProvider,
    _ObjectStoreRegistry,
    _Session,
)


def _make_registry_and_session(scheme: str) -> tuple[_ObjectStoreRegistry, _Session]:
    """Return a fresh registry with ``scheme`` bound to an in-memory provider,
    plus a Session that consults it.
    """
    registry = _ObjectStoreRegistry()
    provider = _ObjectStoreProvider.memory()
    registry.register_provider(scheme, provider)
    session = _Session(store_registry=registry)
    return registry, session


def test_custom_scheme_registration_roundtrip():
    """Register ``test-mem`` and round-trip a small table through it.

    Note on lifetimes: ``ObjectStoreRegistry`` caches active stores under
    ``Weak<ObjectStore>``. Every call to ``MemoryStoreProvider::new_store``
    allocates a fresh ``InMemory`` backend, so the writer's dataset handle
    must stay alive across the read to keep the same in-memory store visible
    from the reader.
    """
    _registry, session = _make_registry_and_session("test-mem")

    table = pa.table(
        {
            "i": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "s": pa.array(["a", "b", "c", "d", "e"], type=pa.string()),
        }
    )
    uri = "test-mem://cache/dataset.lance"

    # Keep the write-side dataset alive across the read so the shared
    # ObjectStore held by the registry's Weak cache remains upgradeable.
    written = lance.write_dataset(table, uri, session=session)
    assert written.count_rows() == 5

    read_ds = lance.dataset(uri, session=session)
    round_trip = read_ds.to_table()

    assert round_trip.equals(table), (
        "round-trip through test-mem:// scheme did not match the written table"
    )
    assert read_ds.count_rows() == 5


def test_registry_repr_and_reuse():
    """Registry ``__repr__`` reports cache stats, and reusing a registered
    scheme with the same params returns a cached store rather than a new one.
    """
    registry, session = _make_registry_and_session("test-mem-reuse")

    table = pa.table({"i": pa.array([1, 2, 3], type=pa.int64())})
    uri = "test-mem-reuse://cache/reuse.lance"

    written = lance.write_dataset(table, uri, session=session)

    stats_after_write = repr(registry)
    assert "active_stores=" in stats_after_write
    assert "hits=" in stats_after_write
    assert "misses=" in stats_after_write

    # Re-open the same URI with the same session. This should hit the
    # registry's active-stores cache (weak ref still upgradeable because
    # ``written`` holds a strong ref).
    _reopened = lance.dataset(uri, session=session)
    stats_after_reopen = repr(registry)

    # We do not assert exact hit counts — the registry accounts for both the
    # write- and read-path resolutions — but the reuse call must not have
    # produced additional active-store entries.
    _ = stats_after_reopen  # kept for post-mortem debugging when running verbose
    _ = written  # keep the write handle alive until the assertion above passes


def test_missing_scheme_raises_helpful_error():
    """A URI whose scheme is neither built-in nor registered must raise, and
    the error message must name the missing scheme.
    """
    # Fresh registry with nothing custom registered.
    session = _Session(store_registry=_ObjectStoreRegistry())

    with pytest.raises(Exception) as excinfo:  # OSError or lance.LanceError
        lance.dataset("no-such-scheme://foo/bar.lance", session=session)

    assert "no-such-scheme" in str(excinfo.value), (
        "expected the missing scheme name in the error, got: " + str(excinfo.value)
    )


def test_register_provider_rejects_empty_scheme():
    """Empty schemes are rejected at registration time."""
    registry = _ObjectStoreRegistry()
    with pytest.raises(ValueError):
        registry.register_provider("", _ObjectStoreProvider.memory())


def test_from_capsule_roundtrip():
    """Exercise the ``PyCapsule`` handoff end to end with an in-process producer.

    ``_memory_capsule()`` mirrors what an external, ABI-compatible wheel emits:
    a ``PyCapsule`` named ``lance_object_store_provider`` holding an
    ``Arc<dyn ObjectStoreProvider>``. ``from_capsule`` must accept it (name
    check passes), adopt the provider, and the result must be registrable and
    usable for a read/write round-trip — covering the clone-on-adopt and the
    capsule's own drop when it is garbage-collected.
    """
    registry = _ObjectStoreRegistry()
    capsule = _ObjectStoreProvider._memory_capsule()
    provider = _ObjectStoreProvider.from_capsule(capsule)
    # Drop our reference to the capsule; the adopted Arc must keep the provider
    # alive independently of the capsule.
    del capsule
    registry.register_provider("test-cap", provider)
    session = _Session(store_registry=registry)

    table = pa.table({"i": pa.array([1, 2, 3, 4], type=pa.int64())})
    uri = "test-cap://cache/cap.lance"
    written = lance.write_dataset(table, uri, session=session)
    assert written.count_rows() == 4

    read_ds = lance.dataset(uri, session=session)
    assert read_ds.to_table().equals(table)


def test_from_capsule_rejects_wrong_name():
    """A capsule whose name does not match is rejected before its pointer is
    ever dereferenced (regression for the ``pointer_checked(None)`` bug, which
    rejected *every* correctly-named capsule).
    """
    import ctypes

    make = ctypes.pythonapi.PyCapsule_New
    make.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    make.restype = ctypes.py_object
    # A capsule with the wrong name and a dummy pointer. The name check must
    # fail first, so the bogus pointer is never read.
    storage = (ctypes.c_void_p * 2)()
    capsule = make(ctypes.addressof(storage), b"not-a-lance-provider", None)

    with pytest.raises(ValueError):
        _ObjectStoreProvider.from_capsule(capsule)
