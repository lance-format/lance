# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import subprocess
import sys

import lance
import pyarrow as pa
import pytest

# The metrics recorder is process-global and installed once, so the whole
# bridge is exercised in a single test to avoid cross-test global-state coupling.


def _metrics_by_name(reader):
    data = reader.get_metrics_data()
    result = {}
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                result[metric.name] = metric
    return result


def test_instrument_lance_metrics_exports_object_store_metrics(tmp_path):
    pytest.importorskip("opentelemetry.sdk.metrics")
    from lance.otel import instrument_lance_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    assert instrument_lance_metrics(provider)

    # The catalog is populated once the recorder is installed.
    from lance.lance import lance_metrics_catalog

    catalog = {desc.name: desc for desc in lance_metrics_catalog()}
    assert "lance_object_store_requests_total" in catalog
    assert catalog["lance_object_store_request_duration_seconds"].kind == "histogram"
    # Gauges and the retryable counter are described too, so they surface in the
    # export even when a plain write doesn't happen to emit them.
    assert catalog["lance_object_store_retryable_responses_total"].kind == "counter"
    assert catalog["lance_object_store_in_flight_requests"].kind == "gauge"
    # Memory metrics are described whether or not LANCE_MEMORY_OBSERVE is set;
    # they only carry values while it is.
    for name in (
        "lance_memory_rss_bytes",
        "lance_memory_rss_peak_bytes",
        "lance_memory_cgroup_bytes",
        "lance_memory_cgroup_limit_bytes",
    ):
        assert catalog[name].kind == "gauge"
    assert catalog["lance_memory_stage_peak_bytes"].kind == "histogram"
    assert catalog["lance_memory_operation_peak_bytes"].kind == "histogram"

    # Generate object store activity on the local filesystem (scheme "file").
    table = pa.table({"id": pa.array(range(256))})
    dataset = lance.write_dataset(table, str(tmp_path / "ds.lance"))
    assert dataset.to_table().num_rows == 256

    metrics = _metrics_by_name(reader)

    requests = metrics["lance_object_store_requests_total"]
    points = list(requests.data.data_points)
    assert points, "expected at least one request data point"
    # The `base` label carries the store scheme ("file") by default.
    assert all("base" in p.attributes and "operation" in p.attributes for p in points)
    assert sum(p.value for p in points) > 0

    # Histograms are decomposed into bucket / count / sum observable counters.
    bucket = metrics["lance_object_store_request_duration_seconds_bucket"]
    bucket_points = list(bucket.data.data_points)
    assert bucket_points
    assert all("le" in p.attributes for p in bucket_points)
    # The implicit +Inf bucket must be present and is the cumulative maximum.
    assert any(p.attributes["le"] == "+Inf" for p in bucket_points)

    count = metrics["lance_object_store_request_duration_seconds_count"]
    assert sum(p.value for p in count.data.data_points) > 0

    # The `_sum` instrument must also be wired and report positive latency.
    duration_sum = metrics["lance_object_store_request_duration_seconds_sum"]
    assert sum(p.value for p in duration_sum.data.data_points) > 0


_OBSERVED_BUILD = """
import sys

import lance
import pyarrow as pa
from lance.otel import instrument_lance_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

reader = InMemoryMetricReader()
assert instrument_lance_metrics(MeterProvider(metric_readers=[reader]))
dataset = lance.write_dataset(pa.table({"id": pa.array(range(256))}), sys.argv[1])
dataset.create_scalar_index("id", "BTREE")

points = {}
for resource_metrics in reader.get_metrics_data().resource_metrics:
    for scope_metrics in resource_metrics.scope_metrics:
        for metric in scope_metrics.metrics:
            points[metric.name] = list(metric.data.data_points)

assert points["lance_memory_rss_bytes"][0].value > 0
build = {"operation": "create_index", "index_type": "btree"}
builds = [
    point.value
    for point in points["lance_memory_operation_peak_bytes_count"]
    if dict(point.attributes) == build
]
assert builds == [1], builds
# Peaks fall into byte buckets, the smallest of which is 16 MiB.
bounds = {
    point.attributes["le"]
    for point in points["lance_memory_operation_peak_bytes_bucket"]
}
assert "16777216" in bounds, bounds
"""


@pytest.mark.skipif(
    sys.platform != "linux", reason="process memory is only read on Linux"
)
def test_instrument_lance_metrics_exports_memory_while_observed(tmp_path):
    pytest.importorskip("opentelemetry.sdk.metrics")
    # LANCE_MEMORY_OBSERVE is read once per process, so the observed build
    # runs in a process of its own.
    result = subprocess.run(
        [sys.executable, "-c", _OBSERVED_BUILD, str(tmp_path / "ds.lance")],
        capture_output=True,
        text=True,
        env={**os.environ, "LANCE_MEMORY_OBSERVE": "1"},
    )
    assert result.returncode == 0, result.stderr


_INVALID_SWITCH_BUILD = """
import sys

import lance
import pyarrow as pa

dataset = lance.write_dataset(pa.table({"id": pa.array(range(16))}), sys.argv[1])
try:
    dataset.create_scalar_index("id", "BTREE")
except ValueError as error:
    assert 'LANCE_MEMORY_OBSERVE="maybe" is invalid' in str(error), error
else:
    raise AssertionError("the build ignored an invalid LANCE_MEMORY_OBSERVE")
"""


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_invalid_memory_switch_fails_index_builds(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _INVALID_SWITCH_BUILD, str(tmp_path / "ds.lance")],
        capture_output=True,
        text=True,
        env={**os.environ, "LANCE_MEMORY_OBSERVE": "maybe"},
    )
    assert result.returncode == 0, result.stderr


def test_snapshot_empty_before_install_is_safe():
    # snapshot is callable regardless of installation state and never raises.
    from lance.lance import snapshot_lance_metrics

    assert isinstance(snapshot_lance_metrics(), list)


def test_instrument_warns_when_recorder_unavailable(monkeypatch):
    # A foreign `metrics` recorder already installed -> register returns False;
    # instrument_lance_metrics must warn and return False without instrumenting.
    pytest.importorskip("opentelemetry.sdk.metrics")
    import lance.otel as otel
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    monkeypatch.setattr(otel, "register_lance_metrics_recorder", lambda: False)

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    with pytest.warns(UserWarning, match="recorder"):
        assert otel.instrument_lance_metrics(provider) is False
