#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Install isolated pylance interpreters and run the IVF_RQ matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def env_protoc() -> str:
    found = shutil.which("protoc")
    if found is None:
        raise RuntimeError("protoc is required to build pylance from source")
    return found

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def ensure_wheel_venv(root: Path, version: str) -> Path:
    venv = root / f"venv-{version}"
    python = venv / "bin" / "python"
    if not python.exists():
        run(["uv", "venv", str(venv)])
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                f"pylance=={version}",
                "pyarrow",
                "numpy",
                "huggingface_hub",
            ]
        )
    return python


def ensure_local_venv(root: Path) -> Path:
    venv = root / "venv-main"
    python = venv / "bin" / "python"
    marker = venv / ".built-from-source"
    if python.exists() and marker.exists():
        return python
    if python.exists():
        try:
            subprocess.check_call([str(python), "-c", "import lance"])
        except subprocess.CalledProcessError:
            pass
        else:
            marker.write_text("ok\n")
            return python
    run(["uv", "venv", str(venv)])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "maturin",
            "pyarrow",
            "numpy",
            "huggingface_hub",
        ]
    )
    # Fat LTO on pylance OOMs a 16 GiB box. Keep opt-level=3 without LTO.
    env = os.environ.copy()
    env["CARGO_PROFILE_RELEASE_LTO"] = "false"
    env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "8"
    env["PROTOC"] = env_protoc()
    wheel_dir = root / "wheels"
    wheel_dir.mkdir(exist_ok=True)
    subprocess.check_call(
        [
            str(venv / "bin" / "maturin"),
            "build",
            "--release",
            "-m",
            str(REPO / "python" / "Cargo.toml"),
            "--out",
            str(wheel_dir),
        ],
        env=env,
    )
    wheels = sorted(wheel_dir.glob("pylance-*.whl"))
    if not wheels:
        raise RuntimeError(f"maturin built no pylance wheel in {wheel_dir}")
    run(["uv", "pip", "install", "--python", str(python), "--force-reinstall", str(wheels[-1])])
    marker.write_text("ok\n")
    return python


def bench_run_cmd(
    *,
    python: Path,
    data_dir: Path,
    label: str,
    bits: int,
    query_count: int,
    out: Path,
) -> list[str]:
    return [
        str(python),
        str(HERE / "bench.py"),
        "run",
        "--data-dir",
        str(data_dir),
        "--label",
        label,
        "--bits",
        str(bits),
        "--query-count",
        str(query_count),
        "--out",
        str(out),
    ]


def latest_label() -> str:
    # Ignore bench-only commits so the chart names the engine revision.
    sha = subprocess.check_output(
        ["git", "log", "-1", "--format=%h", "--", ":!benchmarks"],
        cwd=REPO,
        text=True,
    ).strip()
    return f"{sha} (main)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/lance-vibe-msmarco"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lance-vibe-venvs"))
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-main-build", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        run(
            [
                sys.executable,
                str(HERE / "bench.py"),
                "download",
                "--data-dir",
                str(args.data_dir),
            ]
        )

    jobs = [
        ("v11.0.0", ensure_wheel_venv(args.work_dir, "11.0.0")),
        ("v12.0.0", ensure_wheel_venv(args.work_dir, "12.0.0")),
    ]
    if not args.skip_main_build:
        jobs.append((latest_label(), ensure_local_venv(args.work_dir)))

    # Each version writes its own IVF_RQ files. Sharing a writer across
    # readers mixes one version's index layout with another version's search
    # path, which is not that version's performance.
    results = []
    for label, python in jobs:
        for bits in (1, 5):
            slug = label.replace(" ", "_").replace("(", "").replace(")", "")
            out = args.results_dir / f"{slug}-rq{bits}.json"
            run(
                bench_run_cmd(
                    python=python,
                    data_dir=args.data_dir,
                    label=label,
                    bits=bits,
                    query_count=args.query_count,
                    out=out,
                )
            )
            results.append(str(out))

    manifest = args.results_dir / "_manifest.json"
    manifest.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"wrote {manifest}")
    plot = HERE / "plot.py"
    if plot.exists() and results:
        run(
            [
                sys.executable,
                str(plot),
                "--results-dir",
                str(args.results_dir),
                "--out",
                str(args.results_dir / "ivf_rq_latency.png"),
            ]
        )


if __name__ == "__main__":
    main()
