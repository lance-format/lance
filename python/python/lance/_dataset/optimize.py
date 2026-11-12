# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset-wide maintenance operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, TypedDict

from ..lance import Compaction, CompactionMetrics

if TYPE_CHECKING:
    from ..dataset import LanceDataset


class AutoCleanupConfig(TypedDict):
    interval: int
    older_than_seconds: int


class DatasetOptimizer:
    def __init__(self, dataset: LanceDataset):
        self._dataset = dataset

    def compact_files(
        self,
        *,
        target_rows_per_fragment: Optional[int] = None,
        max_rows_per_group: Optional[int] = None,
        max_bytes_per_file: Optional[int] = None,
        min_bytes_per_file: Optional[int] = None,
        materialize_deletions: Optional[bool] = None,
        materialize_deletions_threshold: Optional[float] = None,
        defer_index_remap: Optional[bool] = None,
        num_threads: Optional[int] = None,
        batch_size: Optional[int] = None,
        compaction_mode: Optional[
            Literal["reencode", "try_binary_copy", "force_binary_copy"]
        ] = None,
        binary_copy_read_batch_bytes: Optional[int] = None,
        max_source_fragments: Optional[int] = None,
        max_source_rows: Optional[int] = None,
        max_source_bytes: Optional[int] = None,
        source_budget_mode: Optional[Literal["hard", "soft"]] = None,
        excluded_fragment_ids: Optional[list[int]] = None,
        data_storage_version: Optional[str] = None,
    ) -> CompactionMetrics:
        """Compacts small files in the dataset, reducing total number of files.

        This does a few things:
         * Removes deleted rows from fragments
         * Removes dropped columns from fragments
         * Merges small fragments into larger ones

        This method preserves the insertion order of the dataset. This may mean
        it leaves small fragments in the dataset if they are not adjacent to
        other fragments that need compaction. For example, if you have fragments
        with row counts 5 million, 100, and 5 million, the middle fragment will
        not be compacted because the fragments it is adjacent to do not need
        compaction.

        Default values for these options can be stored in the dataset manifest
        config using keys prefixed with ``lance.compaction.``. For example,
        setting the config key ``lance.compaction.target_rows_per_fragment`` to
        ``"500000"`` will use 500,000 as the default target rows per fragment.
        Explicitly provided parameters take precedence over manifest config
        values, which in turn take precedence over hardcoded defaults.

        Supported config keys: ``lance.compaction.target_rows_per_fragment``,
        ``lance.compaction.max_rows_per_group``,
        ``lance.compaction.max_bytes_per_file``,
        ``lance.compaction.min_bytes_per_file``,
        ``lance.compaction.materialize_deletions``,
        ``lance.compaction.materialize_deletions_threshold``,
        ``lance.compaction.defer_index_remap``,
        ``lance.compaction.batch_size``,
        ``lance.compaction.compaction_mode``,
        ``lance.compaction.binary_copy_read_batch_bytes``,
        ``lance.compaction.max_source_fragments``,
        ``lance.compaction.max_source_rows``,
        ``lance.compaction.max_source_bytes``,
        ``lance.compaction.source_budget_mode``,
        ``lance.compaction.data_storage_version``.

        Parameters
        ----------
        target_rows_per_fragment: int, optional
            The target number of rows per fragment. This is the number of rows
            that will be in each fragment after compaction. If not specified,
            uses the manifest config value, or 1024*1024.
        max_rows_per_group: int, optional
            Max number of rows per group. This does not affect which fragments
            need compaction, but does affect how they are re-written if selected.
            If not specified, uses the manifest config value, or 1024.

            This setting only affects datasets using the legacy storage format.
            The newer format does not require row groups.
        max_bytes_per_file: Optional[int], default None
            Maximum bytes per output file. See
            :attr:`lance.optimize.CompactionOptions.max_bytes_per_file`
            for defaults.
        min_bytes_per_file: int, optional
            Files at or above this size are excluded from small-file compaction.
            When omitted or None, defaults to the dataset config value, or 95% of
            :attr:`lance.optimize.CompactionOptions.max_bytes_per_file`.
            Must be greater than zero and no greater than that limit.
            Files needing cleanup may still be compacted.
            For example, ``compact_files(min_bytes_per_file=512 * 1024 * 1024)``.
        materialize_deletions: bool, optional
            Whether to compact fragments with soft deleted rows so they are no
            longer present in the file. If not specified, uses the manifest
            config value, or True.
        materialize_deletions_threshold: float, optional
            The fraction of original rows that are soft deleted in a fragment
            before the fragment is a candidate for compaction. If not specified,
            uses the manifest config value, or 0.1.
        defer_index_remap: bool, optional
            Whether to defer index remapping during compaction. If not specified,
            uses the manifest config value, or False.
        num_threads: int, optional
            The number of threads to use when performing compaction. If not
            specified, defaults to the number of cores on the machine.
        batch_size: int, optional
            The batch size to use when scanning input fragments.  You may want
            to reduce this if you are running out of memory during compaction.

            The default will use the same default from ``scanner``.
        compaction_mode: str, optional
            The compaction mode. Valid values:

            - ``"reencode"``: Decode and re-encode data (default).
            - ``"try_binary_copy"``: Try binary copy if fragments are
              compatible, fall back to reencode otherwise.
            - ``"force_binary_copy"``: Use binary copy or fail if fragments
              are not compatible.
        binary_copy_read_batch_bytes: int, optional
            The batch size in bytes for reading during binary copy operations.
            Controls how much data is read at once when performing binary copy.
            Defaults to 16MB.
        max_source_fragments: int, optional
            Maximum number of source fragments to compact in a single run.
            Compaction tasks are included until adding the next task would
            exceed this limit, allowing compaction to proceed incrementally.
            Fragments are processed oldest first. If not specified, uses the
            manifest config value, or applies no limit.
        max_source_rows: int, optional
            Maximum number of source rows to compact in a single run. Rows are
            counted as live rows (physical rows minus soft-deleted rows).
            Tasks are included until adding the next task would exceed this
            limit.
        max_source_bytes: int, optional
            Maximum number of source bytes to compact in a single run,
            measured as the total size of the source fragments' data and
            overlay files. Tasks are included until adding the next task
            would exceed this limit. Blob v2 payloads live in separate
            blob files and are not counted, so this is not a cap on total
            compaction I/O for datasets with blob columns.
        source_budget_mode: {"hard", "soft"}, optional
            Controls how all configured ``max_source_*`` budgets are applied
            to cumulative totals across the tasks selected for one plan.
            ``"hard"`` rejects a task when adding it would exceed any
            cumulative budget and is the default. ``"soft"`` always admits
            the first indivisible task. If that task exceeds a budget, the
            plan stops there; otherwise later tasks use the same cumulative
            checks as ``"hard"``. If a byte budget is set but the first task
            has a source file without a recorded size, ``"soft"`` admits that
            task and stops; ``"hard"`` reports an error.
        excluded_fragment_ids: list[int], optional
            Fragment IDs to exclude from compaction planning. Excluded
            fragments remain unchanged and act as boundaries, so fragments
            on opposite sides are not combined into the same compaction task.
            Duplicate and unknown IDs are ignored.
        data_storage_version: str, optional
            Output data file version, such as "2.2", "stable", or "next".
            Uses the compaction config target when set, otherwise the dataset's
            default write version. Does not change that default or the versions
            of unselected files. V1/V2 cross-family targets are rejected.

        Returns
        -------
        CompactionMetrics
            Metrics about the compaction process

        See Also
        --------
        lance.optimize.Compaction
        """
        opts = {
            k: v
            for k, v in dict(
                target_rows_per_fragment=target_rows_per_fragment,
                max_rows_per_group=max_rows_per_group,
                max_bytes_per_file=max_bytes_per_file,
                min_bytes_per_file=min_bytes_per_file,
                materialize_deletions=materialize_deletions,
                materialize_deletions_threshold=materialize_deletions_threshold,
                defer_index_remap=defer_index_remap,
                num_threads=num_threads,
                batch_size=batch_size,
                compaction_mode=compaction_mode,
                binary_copy_read_batch_bytes=binary_copy_read_batch_bytes,
                max_source_fragments=max_source_fragments,
                max_source_rows=max_source_rows,
                max_source_bytes=max_source_bytes,
                source_budget_mode=source_budget_mode,
                excluded_fragment_ids=excluded_fragment_ids,
                data_storage_version=data_storage_version,
            ).items()
            if v is not None
        }
        return Compaction.execute(self._dataset, opts)

    def inline_deletions(self) -> int:
        """Move small deletion vectors from ``_deletions/`` files into the manifest.

        A reader fetches every deletion file it needs with one request each, so
        on a table where most fragments carry a few deleted rows the first
        ``take`` of a process costs one request per fragment. Deletion vectors
        stored in the manifest arrive with it instead.

        The budget comes from the table config key
        ``lance.deletions.inline_max_bytes``: a deletion vector whose bitmap
        encoding is at most that many bytes is stored inline, both by this
        call and by later deletes. Set it with :meth:`LanceDataset.update_config`
        first. Readers must support the ``FLAG_INLINE_DELETIONS`` feature flag
        once a fragment stores its deletions inline, and the move is one way:
        setting the budget back to ``0`` stops new inline writes but leaves the
        moved vectors in the manifest, so upgrade every reader of the table
        before calling this.

        Returns
        -------
        int
            The number of fragments whose deletion vector moved into the
            manifest. Nothing is committed when it is zero.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> ds = lance.write_dataset(pa.table({"a": range(100)}), "memory://inline")
        >>> ds.delete("a < 10")
        {'num_deleted_rows': 10}
        >>> ds.update_config({"lance.deletions.inline_max_bytes": "65536"})
        {'lance.deletions.inline_max_bytes': '65536'}
        >>> ds.optimize.inline_deletions()
        1
        >>> ds.count_rows()
        90
        """
        return self._dataset._ds.inline_deletions()

    def optimize_indices(self, **kwargs) -> None:
        """Optimizes index performance.

        As new data arrives it is not added to existing indexes automatically.
        When searching we need to perform an indexed search of the old data plus
        an expensive unindexed search on the new data.  As the amount of new
        unindexed data grows this can have an impact on search latency.
        This function will add the new data to existing indexes, restoring the
        performance. By default, this function does not retrain the index, it only
        assigns the new data to existing partitions. This means an update is much
        quicker than retraining the entire index but may have less accuracy
        (especially if the new data exhibits new patterns, concepts, or trends)

        Parameters
        ----------
        num_indices_to_merge: optional, int, default None
            The number of indices to merge.
            If set to 0, new delta index will be created.
        index_names: List[str], default None
            The names of the indices to optimize.
            If None, all indices will be optimized.
        retrain: bool, default False
            Whether to retrain the whole index.
            If true, the index will be retrained based on the current data,
            `num_indices_to_merge` will be ignored,
            and all indices will be merged into one.

            This is useful when the data distribution has changed significantly,
            and we want to retrain the index to improve the search quality.
            This rebuilds the index from the source data and may be expensive.
        max_rows_per_task: optional, int, default None
            Upper bound on the rows one build task covers. Setting this or
            `max_rows_per_segment` splits the build of every index whose
            segments can be merged into several tasks; a split vector build
            never rebalances the partitions (set `retrain` to rebuild).
        max_rows_per_segment: optional, int, default None
            Upper bound on the rows one committed segment holds.

            The limits are packing targets: an indivisible fragment or segment
            larger than a target is processed on its own.

        See :py:class:`lance.optimize.OptimizeIndices` to run the same
        optimization as a plan whose tasks execute on several workers.
        """
        self._dataset._ds.optimize_indices(**kwargs)

    def enable_auto_cleanup(self, auto_cleanup_config: AutoCleanupConfig, **kwargs):
        """Enable autocleaning for an existing dataset.

        Parameters
        ----------
        auto_cleanup_config: AutoCleanupConfig
            Config options for automatic cleanup of the dataset.
            If set, dataset's old versions will be automatically
            cleaned up according to this parameter.
        """
        self._dataset._ds.update_config(
            {
                "lance.auto_cleanup.interval": str(auto_cleanup_config["interval"]),
                "lance.auto_cleanup.older_than": f"{auto_cleanup_config['older_than_seconds']}s",  # noqa E501
            }
        )

    def disable_auto_cleanup(self, **kwargs):
        """Disable every automatic cleanup activation form."""
        config = self._dataset.config()
        keys = [
            "lance.auto_cleanup.interval",
            "lance.auto_cleanup.older_than",
            "lance.auto_cleanup.policy_interval",
            "lance.auto_cleanup.policy",
            *(key for key in config if key.startswith("lance.auto_cleanup.keep_one.")),
        ]
        self._dataset._ds.delete_config_keys(keys)
