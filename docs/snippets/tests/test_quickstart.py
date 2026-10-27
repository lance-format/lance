"""Runs the quickstart snippets embedded in docs/src/quickstart/ end to end."""

import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

QUICKSTART = Path(__file__).resolve().parent.parent / "quickstart"


def spark_env():
    env = dict(os.environ)
    # Spark launches Python workers with `python3` from PATH unless told
    # otherwise, which may not be the interpreter running these tests.
    env["PYSPARK_PYTHON"] = sys.executable
    return env


def ivy_conf():
    """Point Spark's dependency resolution at the directory CI caches."""
    ivy_dir = os.environ.get("SPARK_IVY_DIR")
    return ["--conf", f"spark.jars.ivy={ivy_dir}"] if ivy_dir else []


@pytest.mark.parametrize("install_script", ["lancedb-install.sh", "spark-install.sh"])
def test_install_command_matches_locked_versions(install_script):
    # The pages show these commands to readers, so they must accept the
    # versions the snippets are actually tested against.
    args = shlex.split((QUICKSTART / install_script).read_text())
    assert args[:2] == ["pip", "install"]
    for spec in args[2:]:
        requirement = Requirement(spec)
        installed = version(requirement.name)
        assert requirement.specifier.contains(installed), (
            f"{install_script} asks for {spec!r} but the snippets are tested "
            f"with {requirement.name}=={installed}"
        )


def test_lancedb_quickstart():
    subprocess.run([sys.executable, QUICKSTART / "lancedb_quickstart.py"], check=True)


# Spark creates a Derby metastore in its working directory, so the Spark tests
# run from tmp_path to keep it out of the checkout.
def test_spark_quickstart_pyspark(tmp_path):
    env = spark_env()
    env["PYSPARK_SUBMIT_ARGS"] = shlex.join([*ivy_conf(), "pyspark-shell"])
    subprocess.run(
        [sys.executable, QUICKSTART / "spark_quickstart.py"],
        check=True,
        env=env,
        cwd=tmp_path,
    )


def test_spark_quickstart_sql(tmp_path):
    shutil.rmtree("/tmp/lance-spark-quickstart", ignore_errors=True)
    command = shlex.split(
        (QUICKSTART / "spark-sql.sh").read_text().replace("\\\n", " ")
    )
    command += [*ivy_conf(), "-f", str(QUICKSTART / "spark_quickstart.sql")]
    result = subprocess.run(
        command,
        check=True,
        env=spark_env(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    rows = result.stdout.splitlines()
    assert rows[-4:] == [
        "1\tred running shoes\t79.0",
        "2\tblue trail running shoes\t120.0",
        "3\tleather hiking boots\t150.0",
        "6\ttrail socks\t15.0",
    ]
