import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("failures", "attempts", "exit_code"), [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 3, 1)]
)
def test_maven_setup_retries_before_publishing(tmp_path, failures, attempts, exit_code):
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/java-publish.yml"
    )
    steps = yaml.safe_load(workflow.read_text())["jobs"]["publish"]["steps"]
    setup = next(step for step in steps if step.get("name") == "Set up Maven")

    wrapper = tmp_path / "mvnw"
    wrapper.write_text(
        f"""#!/bin/sh
echo "$*" >> calls
test "$(wc -l < calls)" -gt {failures}
"""
    )
    wrapper.chmod(0o755)

    # Skip the delay between attempts in this test.
    result = subprocess.run(
        ["bash", "-e", "-c", "sleep() { :; }\n" + setup["run"]], cwd=tmp_path
    )
    assert result.returncode == exit_code
    assert (tmp_path / "calls").read_text().splitlines() == ["--version"] * attempts
