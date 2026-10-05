# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Custom benchmark infrastructure for tracking IO and memory stats.

This module provides an `io_memory_benchmark` marker and fixture that tracks:
- Peak memory usage
- Total allocations
- Read IOPS and bytes
- Write IOPS and bytes

Usage:
    @pytest.mark.io_memory_benchmark()
    def test_something(benchmark):
        def workload(dataset):
            dataset.to_table()
        benchmark(workload, dataset)
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import pytest

# Try to import memtest, but don't fail if not available
try:
    import memtest

    MEMTEST_AVAILABLE = memtest.is_preloaded()
except ImportError:
    MEMTEST_AVAILABLE = False


@dataclass
class BenchmarkStats:
    """Statistics collected during a benchmark run."""

    # Memory stats (only populated if memtest is preloaded)
    peak_bytes: int = 0
    total_allocations: int = 0

    # IO stats
    read_iops: int = 0
    read_bytes: int = 0
    write_iops: int = 0
    write_bytes: int = 0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    name: str
    stats: BenchmarkStats


# Global storage for benchmark results
_benchmark_results: List[BenchmarkResult] = []


def _format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _format_count(count: int) -> str:
    """Format a large count with commas."""
    for unit in ["", "K"]:
        if abs(count) < 1000.0:
            return f"{count:.1f} {unit}"
        count /= 1000.0
    return f"{count:.1f} M"


class IOMemoryBenchmark:
    """Benchmark fixture that tracks IO and memory during execution."""

    def __init__(self, test_name: str):
        self._test_name = test_name
        self._stats = BenchmarkStats()

    def __call__(
        self,
        func: Callable,
        dataset: Any,
        warmup: bool = True,
        setup: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """
        Run a benchmark function with IO and memory tracking.

        Parameters
        ----------
        func : Callable
            The function to benchmark. Should accept a dataset as first argument.
        dataset : lance.LanceDataset
            The dataset to pass to the function.
        warmup : bool, default True
            Whether to run a warmup iteration before measuring.
        setup : Callable, optional
            Called before the warmup run and again before the measured run.
            Required for workloads that mutate the dataset, so each run starts
            from the same state. Setup runs before IO stats are reset, so its
            own IO is never attributed to the benchmark. If it returns a value,
            that value is passed to ``func`` instead of ``dataset`` -- use this
            when the reset produces a new dataset handle.

        Returns
        -------
        Any
            The return value of the benchmark function.
        """

        def prepare() -> Any:
            if setup is None:
                return dataset
            replacement = setup()
            return dataset if replacement is None else replacement

        # Warmup run (not measured)
        if warmup:
            func(prepare())

        target = prepare()

        # Reset IO stats before the measured run. Stats are tracked on the
        # ObjectStore, which is shared by every handle from the same session,
        # so resetting/reading through `dataset` also captures work that `func`
        # performs through a handle returned by `setup`.
        dataset.io_stats_incremental()

        # Run with memory tracking if available
        if MEMTEST_AVAILABLE:
            memtest.reset_stats()
            result = func(target)
            mem_stats = memtest.get_stats()
            self._stats.peak_bytes = mem_stats["peak_bytes"]
            self._stats.total_allocations = mem_stats["total_allocations"]
        else:
            result = func(target)

        # Capture IO stats
        io_stats = dataset.io_stats_incremental()
        self._stats.read_iops = io_stats.read_iops
        self._stats.read_bytes = io_stats.read_bytes
        self._stats.write_iops = io_stats.write_iops
        self._stats.write_bytes = io_stats.written_bytes

        return result

    def get_stats(self) -> BenchmarkStats:
        """Get the collected statistics."""
        return self._stats


@pytest.fixture
def io_mem_benchmark(request):
    """
    Fixture that provides IO and memory benchmarking.

    Only active for tests marked with @pytest.mark.io_memory_benchmark().
    For other tests, returns a no-op benchmark that just calls the function.

    Usage:
        @pytest.mark.io_memory_benchmark()
        def test_something(io_mem_benchmark):
            def workload(dataset):
                dataset.to_table()
            io_mem_benchmark(workload, dataset)
    """
    marker = request.node.get_closest_marker("io_memory_benchmark")

    if marker is None:
        # Not an io_memory_benchmark test, return a simple passthrough
        class PassthroughBenchmark:
            def __call__(self, func, dataset, warmup=True, setup=None):
                if setup is not None:
                    replacement = setup()
                    if replacement is not None:
                        dataset = replacement
                return func(dataset)

        yield PassthroughBenchmark()
        return

    test_name = request.node.name
    tracker = IOMemoryBenchmark(test_name)

    yield tracker

    # Store results after test completes
    stats = tracker.get_stats()
    _benchmark_results.append(BenchmarkResult(name=test_name, stats=stats))


def pytest_configure(config):
    """Register the io_memory_benchmark marker."""
    config.addinivalue_line(
        "markers",
        "io_memory_benchmark(): Mark test as an IO/memory benchmark",
    )


def pytest_addoption(parser):
    """Add command-line options for benchmark output."""
    group = parser.getgroup("io_memory_benchmark", "IO/memory benchmark options")
    group.addoption(
        "--benchmark-stats-json",
        action="store",
        default=None,
        metavar="PATH",
        help="Output path for benchmark stats JSON in Bencher Metric Format (BMF)",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print benchmark statistics summary at the end of the test run."""
    if not _benchmark_results:
        return

    terminalreporter.write_sep("=", "IO/Memory Benchmark Statistics")

    # Calculate column widths
    name_width = max(len(r.name) for r in _benchmark_results)
    name_width = max(name_width, len("Test"))

    # Header
    if MEMTEST_AVAILABLE:
        terminalreporter.write_line(
            f"{'Test':<{name_width}}  {'Peak Mem':>10}  {'Allocs':>10}  "
            f"{'Read IOPS':>10}  {'Read Bytes':>12}  "
            f"{'Write IOPS':>10}  {'Write Bytes':>12}"
        )
        terminalreporter.write_line("-" * (name_width + 76))
    else:
        terminalreporter.write_line(
            f"{'Test':<{name_width}}  "
            f"{'Read IOPS':>10}  {'Read Bytes':>12}  "
            f"{'Write IOPS':>10}  {'Write Bytes':>12}"
        )
        terminalreporter.write_line("-" * (name_width + 52))

    # Results sorted by read bytes (descending)
    sorted_results = sorted(
        _benchmark_results, key=lambda r: r.stats.read_bytes, reverse=True
    )

    for result in sorted_results:
        s = result.stats
        if MEMTEST_AVAILABLE:
            terminalreporter.write_line(
                f"{result.name:<{name_width}}  "
                f"{_format_bytes(s.peak_bytes):>10}  "
                f"{_format_count(s.total_allocations):>10}  "
                f"{s.read_iops:>10,}  "
                f"{_format_bytes(s.read_bytes):>12}  "
                f"{s.write_iops:>10,}  "
                f"{_format_bytes(s.write_bytes):>12}"
            )
        else:
            terminalreporter.write_line(
                f"{result.name:<{name_width}}  "
                f"{s.read_iops:>10,}  "
                f"{_format_bytes(s.read_bytes):>12}  "
                f"{s.write_iops:>10,}  "
                f"{_format_bytes(s.write_bytes):>12}"
            )

    if not MEMTEST_AVAILABLE:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            "Note: Memory tracking not available. "
            "Run with LD_PRELOAD=$(lance-memtest) to enable."
        )

    terminalreporter.write_line("")


def pytest_sessionfinish(session, exitstatus):
    """Write benchmark results to JSON file if --benchmark-stats-json was specified."""
    if not _benchmark_results:
        return

    output_path = session.config.getoption("--benchmark-stats-json")
    if not output_path:
        return

    # Convert to Bencher Metric Format (BMF)
    bmf_output = {}
    for result in _benchmark_results:
        s = result.stats
        bmf_output[result.name] = {
            "read_iops": {"value": s.read_iops},
            "read_bytes": {"value": s.read_bytes},
            "write_iops": {"value": s.write_iops},
            "write_bytes": {"value": s.write_bytes},
        }
        if MEMTEST_AVAILABLE:
            bmf_output[result.name]["peak_memory_bytes"] = {"value": s.peak_bytes}
            bmf_output[result.name]["total_allocations"] = {
                "value": s.total_allocations
            }

    with open(output_path, "w") as f:
        json.dump(bmf_output, f, indent=2)
