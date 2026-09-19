#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Install isolated pylance interpreters and run the IVF_RQ matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    run(
        [
            str(venv / "bin" / "maturin"),
            "develop",
            "--release",
            "-m",
            str(REPO / "python" / "Cargo.toml"),
        ]
    )
    marker.write_text("ok\n")
    return python


def latest_label() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short=9", "HEAD"], cwd=REPO, text=True
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

    # Build each bit-width once with the oldest runtime so every version
    # queries the same on-disk index. Current can read released IVF_RQ files;
    # the reverse is not true for the latest writer.
    results = []
    for bits in (1, 5):
        for i, (label, python) in enumerate(jobs):
            slug = label.replace(" ", "_").replace("(", "").replace(")", "")
            out = args.results_dir / f"{slug}-rq{bits}.json"
            cmd = [
                str(python),
                str(HERE / "bench.py"),
                "run",
                "--data-dir",
                str(args.data_dir),
                "--label",
                label,
                "--bits",
                str(bits),
                "--query-count",
                str(args.query_count),
                "--out",
                str(out),
            ]
            if i > 0:
                cmd.append("--skip-index")
            run(cmd)
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
