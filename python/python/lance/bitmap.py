# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from collections.abc import MutableSet

from .lance import Bitmap as Bitmap

# `Bitmap` implements the whole `MutableSet` interface in Rust, but it is a
# distinct type rather than a `set` subclass, so `isinstance(b, set)` is False.
# Registering it here means the check that does work across set-like types,
# `isinstance(b, collections.abc.Set)`, recognizes it. Modules that expose a
# `Bitmap` import it from here rather than from `.lance` so the registration
# always runs.
MutableSet.register(Bitmap)

__all__ = ["Bitmap"]
