#!/usr/bin/env python3
"""Single entry point for Lance table maintenance jobs.

Validates the job, loads an execution backend, calls it, and prints exactly
one JSON object on stdout. See references/backend-interface.md.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import sys
from urllib.parse import urlsplit

from backends.base import OPTIONS, STATES, BackendError

BACKENDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backends")
# "." and ".." would name a directory outside the one that holds the jobs.
JOB_ID = re.compile(r"(?!\.\.?\Z)[A-Za-z0-9._-]{1,128}\Z")
JOB_FIELDS = ("table", "operation", "options")
METHODS = ("submit", "status", "cancel")


def invalid(message: str) -> BackendError:
    return BackendError("invalid_input", message)


def check_type(name: str, value, expected: type) -> None:
    # bool is a subclass of int, so it has to be ruled out explicitly.
    if expected is bool:
        ok = isinstance(value, bool)
    elif expected is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif expected is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        try:
            # json accepts NaN and Infinity.
            ok = ok and math.isfinite(value)
        except OverflowError:  # an integer too large for a float
            ok = False
    else:
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
    if not ok:
        raise invalid(f"option {name} has the wrong type")


def load_job(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            job = json.load(f)
    except (OSError, ValueError) as e:
        raise invalid(f"cannot read job file {path}: {e}") from e
    if not isinstance(job, dict):
        raise invalid("the job must be a JSON object")
    for field in job:
        if field not in JOB_FIELDS:
            raise invalid(f"unknown field: {field}")

    operation = job.get("operation")
    if not isinstance(operation, str) or operation not in OPTIONS:
        raise invalid(f"unknown operation: {operation}")
    options = job.get("options", {})
    if not isinstance(options, dict):
        raise invalid("options must be a JSON object")
    for name, value in options.items():
        if name not in OPTIONS[operation]:
            raise invalid(f"unknown option: {name}")
        check_type(name, value, OPTIONS[operation][name])
    if operation == "cleanup":
        if "older_than_seconds" not in options and "retain_versions" not in options:
            raise invalid("cleanup needs older_than_seconds or retain_versions")
        if options.get("retain_versions", 1) < 1:
            raise invalid("retain_versions must be at least 1")
        if options.get("older_than_seconds", 0) < 0:
            raise invalid("older_than_seconds must not be negative")

    table = job.get("table")
    if not isinstance(table, str) or not table:
        raise invalid("table must be a non-empty string")
    if "://" in table:
        try:
            parts = urlsplit(table)
        except ValueError as e:
            raise invalid(f"table is not a valid URI: {e}") from e
        if parts.username or parts.password:
            raise invalid("table must not contain credentials")
    else:
        table = os.path.abspath(os.path.expanduser(table))
    return {"table": table, "operation": operation, "options": options}


def list_backends() -> list[str]:
    names = [f[:-3] for f in os.listdir(BACKENDS_DIR) if f.endswith(".py")]
    return sorted(n for n in names if n not in ("__init__", "base"))


def load_backend(name: str):
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)
    elif name in list_backends():
        module_name, class_name = f"backends.{name}", "Backend"
    else:
        raise invalid(f"unknown backend: {name}. Available: {list_backends()}")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    # ValueError and TypeError: a name that starts or ends with a dot.
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        raise invalid(f"cannot load backend {name}: {e}") from e
    missing = [m for m in METHODS if not callable(getattr(cls, m, None))]
    if missing:
        raise invalid(f"backend {name} has no {', '.join(missing)}")
    return cls()


def check_status(status) -> dict:
    if (
        not isinstance(status, dict)
        or status.get("state") not in STATES
        or not isinstance(status.get("job_id"), str)
        or not JOB_ID.match(status["job_id"])
    ):
        raise BackendError("internal", f"backend returned an invalid status: {status}")
    return status


@contextlib.contextmanager
def stdout_to_stderr():
    """Send everything written to stdout to stderr, including what child
    processes and native code write, so that stdout carries only our JSON."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def run(args: argparse.Namespace) -> dict:
    if args.command == "backends":
        return {"backends": list_backends()}
    if args.command in ("preview", "submit"):
        target = load_job(args.target)
    elif JOB_ID.match(args.target):
        target = args.target
    else:
        raise invalid(f"invalid job id: {args.target}")

    with stdout_to_stderr():
        backend = load_backend(args.backend)
        method = getattr(backend, args.command, None)
        if not callable(method):
            raise BackendError("unsupported", f"{args.backend} has no {args.command}")
        result = method(target)

    if args.command == "preview":
        if not isinstance(result, dict):
            raise BackendError("internal", "backend returned an invalid preview")
        return result
    return {**check_status(result), "backend": args.backend}


class Parser(argparse.ArgumentParser):
    def error(self, message: str):
        raise invalid(message)


def main() -> None:
    parser = Parser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    commands.add_parser("backends", help="list the backends shipped with this skill")
    for name, target in (
        ("preview", "job.json"),
        ("submit", "job.json"),
        ("status", "job_id"),
        ("cancel", "job_id"),
    ):
        command = commands.add_parser(name)
        command.add_argument("target", metavar=target)
        command.add_argument("--backend", default="local")

    exit_code = 0
    try:
        output = run(parser.parse_args())
    except Exception as e:
        code = getattr(e, "code", None)
        if not isinstance(code, str) or not code:
            code = "internal"
        output = {"error": {"code": code, "message": str(e) or type(e).__name__}}
        exit_code = 1
    print(json.dumps(output, default=str))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
