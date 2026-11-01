# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
#
# The following code is originally from https://github.com/pola-rs/polars/blob/ea4389c31b0e87ddf20a85e4c3797b285966edb6/py-polars/polars/dependencies.py
# and is licensed under the MIT license:
#
# License: MIT, Copyright (c) 2020 Ritchie Vink
# https://github.com/pola-rs/polars/blob/main/LICENSE
#
# It has been modified by the Lance developers to fit the needs of the Lance project.


from __future__ import annotations

import re
import sys
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Hashable, cast

_NUMPY_AVAILABLE = True
_PANDAS_AVAILABLE = True
_POLARS_AVAILABLE = True
_TORCH_AVAILABLE = True
_CAGRA_AVAILABLE = True
_RAFT_COMMON_AVAILABLE = True
_HUGGING_FACE_AVAILABLE = True
_PYDANTIC_AVAILABLE = True


class _LazyModule(ModuleType):
    """
    Module that can act both as a lazy-loader and as a proxy.

    Notes
    -----
    We do NOT register this module with `sys.modules` so as not to cause
    confusion in the global environment. This way we have a valid proxy
    module for our own use, but it lives _exclusively_ within lance.

    """

    __lazy__ = True

    _mod_pfx: ClassVar[dict[str, str]] = {
        "numpy": "np.",
        "pandas": "pd.",
        "polars": "pl.",
        "torch": "torch.",
    }

    def __init__(
        self,
        module_name: str,
        *,
        module_available: bool,
    ) -> None:
        """
        Initialise lazy-loading proxy module.

        Parameters
        ----------
        module_name : str
            the name of the module to lazy-load (if available).

        module_available : bool
            indicate if the referenced module is actually available (we will proxy it
            in both cases, but raise a helpful error when invoked if it doesn't exist).

        """
        self._module_available = module_available
        self._module_name = module_name
        self._globals = globals()
        super().__init__(module_name)

    def _import(self) -> ModuleType:
        # import the referenced module, replacing the proxy in this module's globals
        module = import_module(self.__name__)
        self._globals[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, attr: Any) -> Any:
        # have "hasattr('__wrapped__')" return False without triggering import
        # (it's for decorators, not modules, but keeps "make doctest" happy)
        if attr == "__wrapped__":
            raise AttributeError(
                f"{self._module_name!r} object has no attribute {attr!r}"
            )

        # accessing the proxy module's attributes triggers import of the real thing
        if self._module_available:
            # import the module and return the requested attribute
            module = self._import()
            return getattr(module, attr)

        # user has not installed the proxied/lazy module
        elif attr == "__name__":
            return self._module_name
        elif re.match(r"^__\w+__$", attr) and attr != "__version__":
            # allow some minimal introspection on private module
            # attrs to avoid unnecessary error-handling elsewhere
            return None
        else:
            # all other attribute access raises a helpful exception
            pfx = self._mod_pfx.get(self._module_name, "")
            raise ModuleNotFoundError(
                f"{pfx}{attr} requires {self._module_name!r} module to be installed"
            ) from None


def _lazy_import(module_name: str) -> tuple[ModuleType, bool]:
    """
    Lazy import the given module; avoids up-front import costs.

    Parameters
    ----------
    module_name : str
        name of the module to import, eg: "polars".

    Notes
    -----
    If the requested module is not available (eg: has not been installed), a proxy
    module is created in its place, which raises an exception on any attribute
    access. This allows for import and use as normal, without requiring explicit
    guard conditions - if the module is never used, no exception occurs; if it
    is, then a helpful exception is raised.

    Returns
    -------
    tuple of (Module, bool)
        A lazy-loading module and a boolean indicating if the requested/underlying
        module exists (if not, the returned module is a proxy).

    """
    # check if module is LOADED
    if module_name in sys.modules:
        return sys.modules[module_name], True

    # check if module is AVAILABLE
    try:
        module_spec = find_spec(module_name)
        module_available = not (module_spec is None or module_spec.loader is None)
    except ModuleNotFoundError:
        module_available = False

    # create lazy/proxy module that imports the real one on first use
    # (or raises an explanatory ModuleNotFoundError if not available)
    return (
        _LazyModule(
            module_name=module_name,
            module_available=module_available,
        ),
        module_available,
    )


if TYPE_CHECKING:
    import datasets
    import numpy
    import pandas
    import polars
    import torch  # type: ignore[reportMissingImports]
else:
    # heavy/optional third party libs
    numpy, _NUMPY_AVAILABLE = _lazy_import("numpy")
    pandas, _PANDAS_AVAILABLE = _lazy_import("pandas")
    polars, _POLARS_AVAILABLE = _lazy_import("polars")
    torch, _TORCH_AVAILABLE = _lazy_import("torch")
    datasets, _HUGGING_FACE_AVAILABLE = _lazy_import("datasets")
    _, _PYDANTIC_AVAILABLE = _lazy_import("pydantic")


@lru_cache(maxsize=None)
def _might_be(cls: type, type_: str) -> bool:
    # infer whether the given class "might" be associated with the given
    # module (in which case it's reasonable to do a real isinstance check)
    try:
        return any(f"{type_}." in str(o) for o in cls.mro())
    except TypeError:
        return False


def _check_for_numpy(obj: Any, *, check_type: bool = True) -> bool:
    return _NUMPY_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "numpy"
    )


def _check_for_pandas(obj: Any, *, check_type: bool = True) -> bool:
    return _PANDAS_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "pandas"
    )


def _check_for_polars(obj: Any, *, check_type: bool = True) -> bool:
    return _POLARS_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "polars"
    )


def _check_for_torch(obj: Any, *, check_type: bool = True) -> bool:
    return _TORCH_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "torch"
    )


def _check_for_hugging_face(obj: Any, *, check_type: bool = True) -> bool:
    return _HUGGING_FACE_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "datasets"
    )


def _check_for_pydantic(obj: Any, *, check_type: bool = True) -> bool:
    return _PYDANTIC_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "pydantic"
    )


def _is_pydantic_base_model(obj: Any) -> bool:
    if not _PYDANTIC_AVAILABLE:
        return False
    from pydantic import BaseModel

    return isinstance(obj, BaseModel)


def _is_pydantic_base_model_class(obj: Any) -> bool:
    if not _PYDANTIC_AVAILABLE:
        return False
    from pydantic import BaseModel

    return isinstance(obj, type) and issubclass(obj, BaseModel)


def model_to_dict(obj: Any) -> dict[str, Any]:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


def _validate_pydantic_list(data: Any, model_class: type) -> None:
    """Validate that `data` is a list of exact `model_class` instances.

    Rejects non-list iterables (e.g. generators), which would otherwise be
    drained by this validation loop and leave nothing for the caller to
    serialize, and rejects subclass instances, whose extra/overridden fields
    would not match the schema derived from `model_class`.
    """
    if not isinstance(data, list):
        raise TypeError(
            f"Pydantic model data must be provided as a list, got {type(data)!r}"
        )
    for i, item in enumerate(data):
        if type(item) is not model_class:
            raise TypeError(
                f"data[{i}] must be an instance of {model_class!r} exactly "
                f"(subclasses are not accepted), got {type(item)!r} "
                f"(data has {len(data)} items)"
            )


__all__ = [
    # lazy-load third party libs
    "datasets",
    "numpy",
    "pandas",
    "polars",
    "torch",
    # lazy utilities
    "_check_for_hugging_face",
    "_check_for_numpy",
    "_check_for_pandas",
    "_check_for_polars",
    "_check_for_pydantic",
    "_check_for_torch",
    "_is_pydantic_base_model",
    "_is_pydantic_base_model_class",
    "_LazyModule",
    "model_to_dict",
    "_validate_pydantic_list",
    # exported flags/guards
    "_NUMPY_AVAILABLE",
    "_PANDAS_AVAILABLE",
    "_POLARS_AVAILABLE",
    "_PYDANTIC_AVAILABLE",
    "_TORCH_AVAILABLE",
    "_HUGGING_FACE_AVAILABLE",
]
