# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from lance.log import ENV_NAME_PYLANCE_LOGGING_LEVEL, LOGGER, get_log_level, set_logger


@pytest.fixture(autouse=True)
def teardown_logger():
    yield
    while LOGGER.handlers:
        LOGGER.handlers.pop()


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("DEBUG", "DEBUG"),
        ("INFO", "INFO"),
        ("WARNING", "WARNING"),
        ("DEBUG,INFO", "DEBUG"),
        ("", "INFO"),
        ("lance-core=debug,WARNING", "WARNING"),
        ("DEBUG,lance-core=WARNING", "DEBUG"),
    ],
)
def test_get_log_level(env_value, expected):
    with mock.patch.dict(os.environ, {ENV_NAME_PYLANCE_LOGGING_LEVEL: env_value}):
        assert get_log_level() == expected


def test_default_logger_level():
    assert LOGGER.level == logging.INFO


def test_set_logger_with_defaults(tmp_path):
    log_file = tmp_path / "test.log"
    set_logger(file_path=str(log_file))
    assert LOGGER.level == logging.INFO
    assert len(LOGGER.handlers) == 1
    assert isinstance(LOGGER.handlers[0], logging.FileHandler)
    assert LOGGER.handlers[0].baseFilename == str(log_file)


def test_set_logger_with_custom_level(tmp_path):
    log_file = tmp_path / "test.log"
    set_logger(file_path=str(log_file), level=logging.DEBUG)
    assert LOGGER.level == logging.DEBUG


def test_set_logger_with_custom_format(tmp_path):
    log_file = tmp_path / "test.log"
    custom_format = "%(levelname)s: %(message)s"
    set_logger(file_path=str(log_file), format_string=custom_format)
    print(LOGGER.handlers[0].formatter._fmt)
    assert LOGGER.handlers[0].formatter._fmt == custom_format


def test_set_logger_with_custom_handler(tmp_path):
    custom_handler = logging.StreamHandler()
    set_logger(log_handler=custom_handler)
    assert LOGGER.handlers[0] == custom_handler


def test_logger_output(tmp_path, caplog):
    log_file = tmp_path / "test.log"
    set_logger(file_path=str(log_file))
    with caplog.at_level(logging.INFO):
        LOGGER.info("Test log message")
    assert "Test log message" in caplog.text


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_lance_log_file(tmp_path):
    log_file = tmp_path / "lance_rust.log"

    # Run a simple Lance operation with file logging enabled
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lance; import pyarrow as pa; "
            "lance.write_dataset(pa.table({'x': range(10)}), 'memory://test')",
        ],
        capture_output=True,
        env={
            "LANCE_LOG": "debug",
            "LANCE_LOG_FILE": str(log_file),
        },
    )

    assert result.returncode == 0, f"Command failed: {result.stderr.decode()}"

    assert log_file.exists(), "Log file was not created"

    log_content = log_file.read_text()
    assert len(log_content.strip()) > 0, "Log file is empty"

    # Check that stderr is empty or minimal (logs should go to file, not stderr)
    stderr_content = result.stderr.decode().strip()
    # Allow for some minimal output but no actual log messages
    assert "DEBUG" not in stderr_content, (
        "Debug logs should not appear in stderr when file logging is enabled"
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_lance_log_file_with_directory_creation(tmp_path):
    log_dir = tmp_path / "logs" / "nested"
    log_file = log_dir / "lance_rust.log"

    # Run a simple Lance operation with file logging to a nested path
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lance; import pyarrow as pa; "
            "lance.write_dataset(pa.table({'x': range(10)}), 'memory://test')",
        ],
        capture_output=True,
        env={
            "LANCE_LOG": "info",
            "LANCE_LOG_FILE": str(log_file),
        },
    )

    assert result.returncode == 0, f"Command failed: {result.stderr.decode()}"

    # Check that the directory and log file were created
    assert log_dir.exists(), "Log directory was not created"
    assert log_file.exists(), "Log file was not created"

    log_content = log_file.read_text()
    assert len(log_content.strip()) > 0, "Log file is empty"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_lance_log_filters_trace_event_targets(tmp_path):
    log_file = tmp_path / "lance_rust.log"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lance; import pyarrow as pa; "
            "lance.write_dataset(pa.table({'x': range(10)}), 'memory://test')",
        ],
        capture_output=True,
        env={
            "LANCE_LOG": "warn,lance::events::dataset_events=info",
            "LANCE_LOG_FILE": str(log_file),
        },
    )

    assert result.returncode == 0, f"Command failed: {result.stderr.decode()}"

    log_content = log_file.read_text()
    assert "lance::events::dataset_events" in log_content
    assert 'target="lance::dataset_events"' in log_content
    assert "lance::events::file_audit" not in log_content
    assert "lance::file_audit" not in log_content


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="memory observation reads /proc, so it works on Linux only",
)
def test_lance_log_records_memory_events(tmp_path):
    log_file = tmp_path / "lance_rust.log"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lance; import pyarrow as pa; "
            "ds = lance.write_dataset(pa.table({'x': range(10)}), 'memory://test'); "
            "ds.create_scalar_index('x', 'BTREE')",
        ],
        capture_output=True,
        env={
            "LANCE_MEMORY_OBSERVE": "1",
            "LANCE_LOG": "warn,lance::events::memory=info",
            "LANCE_LOG_FILE": str(log_file),
        },
    )

    assert result.returncode == 0, f"Command failed: {result.stderr.decode()}"

    log_content = log_file.read_text()
    memory_lines = [
        line for line in log_content.splitlines() if "lance::events::memory" in line
    ]
    for event in [
        "operation_start",
        "stage_start",
        "stage_complete",
        "operation_complete",
    ]:
        assert any(f'event="{event}"' in line for line in memory_lines), log_content
    assert all("operation_id=" in line for line in memory_lines), log_content
    assert all('index_type="btree"' in line for line in memory_lines), log_content
    assert "lance::events::dataset_events" not in log_content


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_lance_log_file_invalid_path():
    invalid_path = "/invalid/path/that/cannot/be/created/lance.log"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lance; import pyarrow as pa; "
            "lance.write_dataset(pa.table({'x': range(10)}), 'memory://test')",
        ],
        capture_output=True,
        env={
            "LANCE_LOG": "info",
            "LANCE_LOG_FILE": invalid_path,
        },
    )

    # The command should still succeed (fallback to stderr)
    assert result.returncode == 0, (
        f"Command should succeed even with invalid log path: {result.stderr.decode()}"
    )
    assert not Path(invalid_path).exists(), "Log file should not be created"

    # Should contain an error message about the invalid path
    stderr_content = result.stderr.decode()
    assert len(stderr_content) > 0


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_timestamp_precision():
    def get_sample_log_line(precision: str):
        output = subprocess.run(
            [
                sys.executable,
                "-c",
                "import lance; import pyarrow as pa;\
                     lance.write_dataset(pa.table({'x': range(100)}), 'memory://test')",
            ],
            capture_output=True,
            check=True,
            env={
                "LANCE_LOG": "debug",
                "LANCE_LOG_TS_PRECISION": precision,
            },
        )
        return output.stderr.decode("utf-8").splitlines()[0]

    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z.*", get_sample_log_line("ns")
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z.*", get_sample_log_line("us")
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z.*", get_sample_log_line("ms")
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z.*", get_sample_log_line("s")
    )
