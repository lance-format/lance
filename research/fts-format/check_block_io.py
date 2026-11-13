# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""CI-only correctness checks for the disposable byte-access prototype."""

import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile


def checksum(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


binary = sys.argv[1]
with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    blocks = [bytes(range(17)), bytes(index % 251 for index in range(4086)),
              bytes(index % 239 for index in range(4897))]
    payload = b"".join(blocks)
    offsets = [0, 17, 4103, 9000]
    span = 12288
    terms = 3
    (root / "fixture.blocks").write_bytes(payload)
    (root / "fixture.offsets.json").write_text(json.dumps(offsets))
    data = root / "data"
    original = (payload + bytes(span - len(payload))) * terms
    data.write_bytes(original)
    directory = root / "directories"
    packed = struct.pack("<4Q", *offsets)
    directory.write_bytes((packed + bytes(4096 - len(packed))) * terms)

    def run(selected, window, with_directory=False, expected_error=None):
        environment = dict(os.environ)
        environment.pop("FTS_IO_DIRECTORY_FILE", None)
        if with_directory:
            environment["FTS_IO_DIRECTORY_FILE"] = str(directory)
        command = [binary, str(data), str(root / "fixture.offsets.json"), str(terms),
                   "17", str(selected), str(window), "2", "0", "0", "0"]
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        if expected_error:
            assert result.returncode != 0, result.stdout
            assert expected_error in result.stderr, result.stderr
            return None
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    all_expected = (sum(map(checksum, blocks)) * 17) & ((1 << 64) - 1)
    for with_directory in [False, True]:
        full = run(3, 0, with_directory)
        selective = run(3, 4096, with_directory)
        assert full["checksum"] == all_expected
        assert selective["checksum"] == all_expected
        assert full["payload_read_bytes"] == 17 * span
        assert full["metadata_read_bytes"] == (17 * 4096 if with_directory else 0)
        partial_full = run(1, 0, with_directory)
        partial_selective = run(1, 4096, with_directory)
        assert partial_full["checksum"] == partial_selective["checksum"]
        assert partial_full["logical_selected_bytes"] == partial_selective["logical_selected_bytes"]
        assert partial_selective["payload_read_bytes"] < partial_full["payload_read_bytes"]

    bad_directory = struct.pack("<4Q", 1, 17, 4103, 9000)
    directory.write_bytes((bad_directory + bytes(4096 - len(bad_directory))) * terms)
    run(3, 4096, True, "directory content mismatch")
    data.write_bytes(original[:-1])
    run(3, 4096, False, "data length mismatch")
print("byte-access integrity contracts passed")
