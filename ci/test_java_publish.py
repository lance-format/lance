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


def test_maven_cache_contract_across_java_workflows():
    repo_root = Path(__file__).resolve().parents[1]
    java_workflow = yaml.safe_load(
        (repo_root / ".github/workflows/java.yml").read_text()
    )
    publish_workflow = yaml.safe_load(
        (repo_root / ".github/workflows/java-publish.yml").read_text()
    )

    for job_name, steps in [
        ("build-jni", java_workflow["jobs"]["build-jni"]["steps"]),
        ("build-and-test-java", java_workflow["jobs"]["build-and-test-java"]["steps"]),
        ("publish", publish_workflow["jobs"]["publish"]["steps"]),
    ]:
        setup_java = next(
            step
            for step in steps
            if step.get("uses", "").startswith("actions/setup-java@")
        )
        assert "cache" not in setup_java.get("with", {}), (
            f"{job_name} should not use setup-java's built-in cache"
        )

    jni_cache = next(
        step
        for step in java_workflow["jobs"]["build-jni"]["steps"]
        if step.get("name") == "Cache JNI Maven dependencies"
    )
    test_cache = next(
        step
        for step in java_workflow["jobs"]["build-and-test-java"]["steps"]
        if step.get("name") == "Cache Java test dependencies"
    )
    publish_cache = next(
        step
        for step in publish_workflow["jobs"]["publish"]["steps"]
        if step.get("name") == "Restore Maven dependencies"
    )

    jni_prefix = jni_cache["env"]["MAVEN_CACHE_PREFIX"]
    publish_prefix = publish_cache["env"]["MAVEN_CACHE_PREFIX"]
    assert jni_cache["with"]["restore-keys"] == "${{ env.MAVEN_CACHE_PREFIX }}"
    assert jni_prefix != publish_prefix
    assert publish_cache["uses"].startswith("actions/cache/restore@")
    assert publish_cache["with"]["key"] == test_cache["with"]["key"]
    assert publish_cache["with"]["restore-keys"] == "${{ env.MAVEN_CACHE_PREFIX }}"
    assert (
        test_cache["env"]["MAVEN_CACHE_PREFIX"]
        == "maven-java-tests-${{ runner.os }}-${{ runner.arch }}-${{ matrix.java-version == '25' && 'without-spotless' || 'with-spotless' }}-"
    )
    assert (
        publish_prefix
        == "maven-java-tests-${{ runner.os }}-${{ runner.arch }}-with-spotless-"
    )

    for step_name in ("Dry run", "Publish with Java 11"):
        step = next(
            s
            for s in publish_workflow["jobs"]["publish"]["steps"]
            if s.get("name") == step_name
        )
        run_script = step["run"]
        assert "--legacy-local-repository" not in run_script
        assert " -llr" not in run_script
        assert "-Daether.priority.SimpleLocalRepositoryManagerFactory=100" in run_script
        assert "-Daether.connector.http.retryHandler.count=6" in run_script
