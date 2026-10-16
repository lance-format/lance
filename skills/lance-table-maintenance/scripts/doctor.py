#!/usr/bin/env python3
"""Read-only health check for a Lance table. Prints one JSON report.

Reads only the manifest of the latest version: no data file or index file
is opened and no older manifest is read, so the cost grows with the size of
that manifest, not with the amount of data. The exception is a table written
by an old version of Lance whose fragments carry no row count: counting the
rows and planning a compaction then read every data file, and the check gets
slow on such a table with many fragments. It stays read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from backends.base import BUDGET_KEYS, CONFIG_PREFIX
from backends.local import (
    classify,
    log,
    log_to_stderr,
    redact,
    storage_options,
    storage_options_file,
)

DEFAULT_TARGET_ROWS = 1024 * 1024


def check(ds, uri: str, storage: dict | None) -> dict:
    import lance
    from lance.optimize import Compaction

    report = {"table": uri, "pylance_version": lance.__version__, "errors": {}}

    def section(name: str, compute) -> None:
        """One failing section must not take the whole report down."""
        started = time.monotonic()
        try:
            report.update(compute())
        except Exception as e:
            report["errors"][name] = redact(str(e), storage)
        log.info("%s took %.2fs", name, time.monotonic() - started)

    config = ds.config() if hasattr(ds, "config") else {}

    def fragments() -> dict:
        target = int(config.get(CONFIG_PREFIX + "target_rows_per_fragment", 0))
        target = target or DEFAULT_TARGET_ROWS
        stats = ds.stats.dataset_stats(max_rows_per_group=target)
        return {
            "num_fragments": stats["num_fragments"],
            "num_deleted_rows": stats["num_deleted_rows"],
            "small_fragment_threshold_rows": target,
            "num_small_fragments": stats["num_small_files"],
        }

    def compaction() -> dict:
        # The table's own targets apply; its per-run budgets do not, because
        # they say how much one run does, not whether there is work to do.
        keys = [k for k in BUDGET_KEYS if CONFIG_PREFIX + k in config]
        budget = {k: config[CONFIG_PREFIX + k] for k in keys}
        tasks = Compaction.plan(ds, dict.fromkeys(keys)).num_tasks()
        return {"compaction_tasks": tasks, "compaction_budget": budget}

    def versions() -> dict:
        # versions() would read every manifest; version_refs() only lists them.
        return {"num_versions": len(ds.version_refs())}

    def tags() -> dict:
        found = ds.tags.list()
        return {"tags": {name: tag["version"] for name, tag in found.items()}}

    def indices() -> dict:
        # index_stats() can commit a write on tables written by old versions.
        rows = report.get("num_rows")
        found = []
        for index in ds.describe_indices():
            if index.name.startswith("__lance_"):
                continue
            indexed = index.num_rows_indexed
            entry = {
                "name": index.name,
                "index_type": index.index_type,
                "num_segments": len(index.segments),
                "num_rows_indexed": indexed,
                "num_unindexed_rows": None if rows is None else rows - indexed,
            }
            found.append(entry)
        return {"indices": found}

    section("version", lambda: {"version": ds.version})
    section("num_rows", lambda: {"num_rows": ds.count_rows()})
    section("fragments", fragments)
    section("compaction", compaction)
    if hasattr(ds, "version_refs"):
        section("versions", versions)
    else:
        report["errors"]["versions"] = "this pylance cannot count versions cheaply"
    section("tags", tags)
    section(
        "auto_cleanup",
        lambda: {
            "auto_cleanup": {
                k: v for k, v in config.items() if k.startswith("lance.auto_cleanup.")
            }
        },
    )
    if hasattr(ds, "describe_indices"):
        section("indices", indices)
    else:
        report["errors"]["indices"] = "this pylance cannot describe indices cheaply"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("uri", help="Table URI or path")
    args = parser.parse_args()
    log_to_stderr()

    # Keeps the check read-only on tables written by old versions.
    # Must be set before lance is imported.
    os.environ["LANCE_AUTO_MIGRATION"] = "false"
    import lance

    storage = None
    started = time.monotonic()
    try:
        storage = storage_options(storage_options_file())
        ds = lance.dataset(args.uri, storage_options=storage)
    except Exception as e:
        error = {"code": classify(e), "message": redact(str(e), storage)}
        print(json.dumps({"error": error}))
        sys.exit(1)
    # Reads the manifest, which is most of the cost on an object store.
    log.info("open took %.2fs", time.monotonic() - started)
    print(json.dumps(check(ds, args.uri, storage), default=str))


if __name__ == "__main__":
    main()
