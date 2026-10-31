"""Runs the quickstart snippets embedded in docs/src/quickstart/ end to end."""

import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from snippet_runner import run_sql

QUICKSTART = Path(__file__).resolve().parent.parent / "quickstart"
OUTPUTS = QUICKSTART / "outputs"
RUNNER = Path(__file__).resolve().parent / "snippet_runner.py"
INSTALL_SCRIPTS = ["lancedb-install.sh", "spark-install.sh"]


def spark_env():
    env = dict(os.environ)
    # Spark launches Python workers with `python3` from PATH unless told
    # otherwise, which may not be the interpreter running these tests.
    env["PYSPARK_PYTHON"] = sys.executable
    return env


def spark_submit_args():
    # With the default local[*] master, writes split into one task per core and
    # each task writes its own fragment, so outputs such as fragments_indexed
    # would depend on the machine. A fixed thread count keeps them stable.
    args = ["--master", "local[2]"]
    # Point Spark's dependency resolution at the directory CI caches.
    ivy_dir = os.environ.get("SPARK_IVY_DIR")
    if ivy_dir:
        args += ["--conf", f"spark.jars.ivy={ivy_dir}"]
    return args


def check_outputs(snippet: Path, recorded_dir: Path, update: bool):
    """Compare a snippet's recorded outputs with the copies the docs embed."""
    pattern = f"{snippet.name}.*.txt"
    recorded = {path.name: path.read_text() for path in recorded_dir.glob(pattern)}
    if update:
        for path in OUTPUTS.glob(pattern):
            path.unlink()
        OUTPUTS.mkdir(exist_ok=True)
        for name, text in recorded.items():
            (OUTPUTS / name).write_text(text)
        return
    expected = {path.name: path.read_text() for path in OUTPUTS.glob(pattern)}
    hint = "rerun with `pytest --update-outputs` if the change is intended"
    assert sorted(recorded) == sorted(expected), f"sections with output changed; {hint}"
    for name in expected:
        assert recorded[name] == expected[name], f"{name} changed; {hint}"


@pytest.fixture
def update_outputs(request):
    return request.config.getoption("--update-outputs")


def install_requirements(install_script: str) -> list[Requirement]:
    args = shlex.split((QUICKSTART / install_script).read_text())
    assert args[:2] == ["pip", "install"]
    return [Requirement(spec) for spec in args[2:]]


@pytest.mark.parametrize("install_script", INSTALL_SCRIPTS)
def test_install_command_matches_locked_versions(install_script):
    # The pages show these commands to readers, so they must accept the
    # versions the snippets are actually tested against.
    for requirement in install_requirements(install_script):
        installed = version(requirement.name)
        assert requirement.specifier.contains(installed), (
            f"{install_script} asks for {str(requirement)!r} but the snippets are "
            f"tested with {requirement.name}=={installed}"
        )


def test_install_commands_cover_tested_dependencies():
    # A package the snippets are tested with but no page tells readers to
    # install (e.g. pandas for `to_pandas()`) breaks the quickstart on a fresh
    # environment while passing here.
    pyproject = tomllib.loads((QUICKSTART.parent / "pyproject.toml").read_text())
    tested = {
        canonicalize_name(Requirement(spec).name)
        for spec in pyproject["project"]["dependencies"]
    }
    documented = {
        canonicalize_name(requirement.name)
        for script in INSTALL_SCRIPTS
        for requirement in install_requirements(script)
    }
    assert tested == documented


def test_lancedb_quickstart(tmp_path, update_outputs):
    snippet = QUICKSTART / "lancedb_quickstart.py"
    subprocess.run([sys.executable, RUNNER, snippet, tmp_path], check=True)
    check_outputs(snippet, tmp_path, update_outputs)


# Spark creates a Derby metastore in its working directory, so the Spark tests
# run from tmp_path to keep it out of the checkout.
def test_spark_quickstart_pyspark(tmp_path, update_outputs):
    snippet = QUICKSTART / "spark_quickstart.py"
    env = spark_env()
    env["PYSPARK_SUBMIT_ARGS"] = shlex.join([*spark_submit_args(), "pyspark-shell"])
    subprocess.run(
        [sys.executable, RUNNER, snippet, tmp_path / "outputs"],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    check_outputs(snippet, tmp_path / "outputs", update_outputs)


def test_spark_quickstart_sql(tmp_path, update_outputs):
    snippet = QUICKSTART / "spark_quickstart.sql"
    shutil.rmtree("/tmp/lance-spark-quickstart", ignore_errors=True)
    command = shlex.split(
        (QUICKSTART / "spark-sql.sh").read_text().replace("\\\n", " ")
    )
    run_sql(
        [*command, *spark_submit_args()],
        snippet,
        tmp_path / "outputs",
        env=spark_env(),
        cwd=tmp_path,
    )
    check_outputs(snippet, tmp_path / "outputs", update_outputs)
