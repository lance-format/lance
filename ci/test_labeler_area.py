"""Regression tests for format-spec path classification."""

from pathlib import Path

import yaml


ROOT = Path(__file__).parent.parent
LABELER_CONFIG = ROOT / ".github" / "labeler-area.yml"
EXECUTION_PROTO_PATHS = {
    "protos/ann.proto",
    "protos/filtered_read.proto",
    "protos/table_identifier.proto",
}


def paths_for(label):
    config = yaml.safe_load(LABELER_CONFIG.read_text())
    return set(config[label][0]["changed-files"][0]["any-glob-to-any-file"])


def test_format_labels_use_the_same_paths():
    assert paths_for("A-format") == paths_for("format-change")


def test_only_persisted_protos_are_format_changes():
    detected_proto_paths = {
        path for path in paths_for("format-change") if path.startswith("protos/")
    }
    all_proto_paths = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "protos").glob("*.proto")
    }

    assert detected_proto_paths == all_proto_paths - EXECUTION_PROTO_PATHS
