# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# ruff: noqa: F821

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional

if TYPE_CHECKING:
    # We don't import directly because of circular import
    from .fragment import FragmentMetadata


@dataclass(frozen=True)
class IndexProgress:
    """Progress event emitted while creating or merging an index.

    Parameters
    ----------
    event : str
        The type of event. One of ``"start"``, ``"progress"``, or ``"complete"``.
    stage : str
        The stage name. Stage names are index-type-specific.
    completed : int, optional
        The amount of work completed so far, if known.
    total : int, optional
        The total amount of work for the stage, if known.
    unit : str
        The unit of work for ``completed`` / ``total``.
    """

    event: Literal["start", "progress", "complete"]
    stage: str
    completed: Optional[int] = None
    total: Optional[int] = None
    unit: str = ""

    @property
    def fraction(self) -> Optional[float]:
        """Return fractional progress if both ``completed`` and ``total`` are known.

        Returns ``None`` when ``total`` is ``None`` or ``0`` (treated as unknown).
        """
        if self.completed is None or self.total in (None, 0):
            return None
        return min(self.completed / self.total, 1.0)


class FragmentWriteProgress(ABC):
    """Progress tracking for Writing a Dataset or Fragment.

    An exception raised by a hook fails the write. The hooks run on the async
    runtime of the write, so Lance calls that block on it, such as
    ``lance.dataset()``, panic when made from a hook.

    Warns
    -----
    This tracking class is experimental and may change in the future.
    """

    def _do_begin(self, fragment_json: str, **kwargs):
        """Called when a new fragment is created"""
        from .fragment import FragmentMetadata

        fragment = FragmentMetadata.from_json(fragment_json)
        return self.begin(fragment, **kwargs)

    @abstractmethod
    def begin(self, fragment: "FragmentMetadata", **kwargs) -> None:
        """Called when a new fragment is about to be written.

        Parameters
        ----------
        fragment : FragmentMetadata
            The fragment that is open to write to. The fragment id might not
            yet be assigned at this point, and its data files might not be
            listed yet.
        kwargs: dict, optional
            Extra keyword arguments to pass to the implementation.

        Returns
        -------
        None
        """
        pass

    def _do_complete(self, fragment_json: str, **kwargs):
        """Called when a fragment is completed"""
        from .fragment import FragmentMetadata

        fragment = FragmentMetadata.from_json(fragment_json)
        return self.complete(fragment, **kwargs)

    @abstractmethod
    def complete(self, fragment: "FragmentMetadata", **kwargs) -> None:
        """Callback when a fragment is completely written.

        Its data file is closed, but a later failure of the write can still
        remove it, so this does not mean the fragment will be committed.

        Parameters
        ----------
        fragment : FragmentMetadata
            The fragment that was written. The fragment id might not yet be
            assigned at this point.
        kwargs: dict, optional
            Extra keyword arguments to pass to the implementation.
        """
        pass


class NoopFragmentWriteProgress(FragmentWriteProgress):
    """No-op implementation of WriteProgressTracker.

    This is the default implementation.
    """

    def begin(self, fragment: "FragmentMetadata", **kargs):
        pass

    def complete(self, fragment: "FragmentMetadata", **kwargs):
        pass


class FileSystemFragmentWriteProgress(FragmentWriteProgress):
    """Progress tracking for Writing a Dataset or Fragment.

    Warns
    -----
    This tracking class is experimental and will change in the future.

    This implementation writes a JSON file to track in-progress state
    to the filesystem for each fragment.

    The files are named after the fragment id. ``write_dataset`` and
    ``write_fragments`` report id 0 for every fragment, as does
    ``LanceFragment.create`` without a ``fragment_id``, so those fragments share
    one pair of files and ``fragment_0.json`` holds only the last fragment begun.
    Writers sharing a directory and a fragment id also share the marker, which can
    then disappear while one of them is still writing or after one has crashed, so
    give such writers their own directories.
    """

    PROGRESS_EXT: str = ".in_progress"

    def __init__(self, base_uri: str, metadata: Optional[Dict[str, str]] = None):
        """Create a FileSystemFragmentWriteProgress tracker.

        Parameters
        ----------
        base_uri : str
            The base directory to write the progress files to. Two files will be created
            under this directory: a Fragment file, and a JSON file to track progress.
        metadata : dict, optional
            Extra metadata for this Progress tracker instance. Can be used to track
            distributed worker where this tracker is running.
        """
        from pyarrow.fs import FileSystem

        fs, path = FileSystem.from_uri(base_uri)
        self._fs = fs
        self._base_path: str = path
        self._metadata = metadata if metadata else {}

    def _in_progress_path(self, fragment: "FragmentMetadata") -> str:
        return os.path.join(
            self._base_path, f"fragment_{fragment.id}{self.PROGRESS_EXT}"
        )

    def _fragment_file(self, fragment: "FragmentMetadata") -> str:
        return os.path.join(self._base_path, f"fragment_{fragment.id}.json")

    def begin(self, fragment: "FragmentMetadata", **kwargs):
        """Called when a new fragment is created.

        Parameters
        ----------
        fragment : FragmentMetadata
            The fragment that is open to write to.
        """

        self._fs.create_dir(self._base_path, recursive=True)

        with self._fs.open_output_stream(self._in_progress_path(fragment)) as out:
            progress_data = {
                "fragment_id": fragment.id,
                "metadata": self._metadata,
            }
            out.write(json.dumps(progress_data).encode("utf-8"))

        with self._fs.open_output_stream(self._fragment_file(fragment)) as out:
            out.write(json.dumps(fragment.to_json()).encode("utf-8"))

    def complete(self, fragment: "FragmentMetadata", **kwargs):
        """Called when a fragment is completed"""
        try:
            self._fs.delete_file(self._in_progress_path(fragment))
        except FileNotFoundError:
            # Another writer sharing this directory removed the marker first.
            pass
