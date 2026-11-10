"""Run a snippet file section by section and record what each section displays.

Docs pages embed sections of the snippet files delimited by
`--8<-- [start:name]` / `--8<-- [end:name]` markers, followed by the recorded
output of each section. The tests compare freshly recorded output with the
checked-in copies so the pages cannot drift from what the code really prints.

Python files run like a notebook: every section shares one namespace, and the
value of a section's final expression is displayed along with anything it
prints. Run this module directly on a Python snippet file so that each file
gets a fresh interpreter:

    python snippet_runner.py <snippet.py> <output dir>
"""

import ast
import contextlib
import io
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = re.compile(r"--8<-- \[(start|end):([\w-]+)\]")


@dataclass
class Chunk:
    # None for code outside any section, which runs but is not recorded.
    section: str | None
    first_line: int
    text: str


def split_sections(source: str) -> list[Chunk]:
    chunks = []
    section, first_line, lines = None, 1, []
    for number, line in enumerate(source.splitlines(keepends=True), start=1):
        marker = MARKER.search(line)
        if marker is None:
            lines.append(line)
            continue
        kind, name = marker.groups()
        if (kind == "start") != (section is None) or (
            kind == "end" and name != section
        ):
            raise ValueError(f"line {number}: unexpected {kind} marker for {name!r}")
        chunks.append(Chunk(section, first_line, "".join(lines)))
        section = name if kind == "start" else None
        first_line, lines = number + 1, []
    if section is not None:
        raise ValueError(f"section {section!r} is never closed")
    chunks.append(Chunk(None, first_line, "".join(lines)))
    return [chunk for chunk in chunks if chunk.text.strip()]


def write_outputs(outputs: dict[str, str], prefix: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for section, text in outputs.items():
        if text.strip():
            (output_dir / f"{prefix}.{section}.txt").write_text(text.strip("\n") + "\n")


def run_python_chunk(chunk: Chunk, filename: str, namespace: dict) -> None:
    tree = ast.parse(chunk.text, filename)
    ast.increment_lineno(tree, chunk.first_line - 1)
    final_expression = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final_expression = tree.body.pop().value
    exec(compile(tree, filename, "exec"), namespace)
    if final_expression is not None:
        value = eval(
            compile(ast.Expression(final_expression), filename, "eval"), namespace
        )
        if value is not None:
            print(repr(value))


def run_python(path: Path, output_dir: Path) -> None:
    import pandas as pd

    # Show whole frames rather than pandas' terminal-width truncation, and round
    # floats so platform-specific last digits (e.g. BM25 scores) don't leak in.
    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 4)

    namespace = {"__name__": "__main__", "__file__": str(path)}
    outputs = {}
    for chunk in split_sections(path.read_text()):
        if chunk.section is None:
            run_python_chunk(chunk, str(path), namespace)
            continue
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            run_python_chunk(chunk, str(path), namespace)
        outputs[chunk.section] = captured.getvalue()
    write_outputs(outputs, path.name, output_dir)


def run_sql(command: list[str], path: Path, output_dir: Path, **run_kwargs) -> None:
    """Run a SQL snippet file through a CLI such as spark-sql.

    The CLI prints every result set to stdout with nothing to tell sections or
    statements apart, so a marker query is interleaved after each statement.
    """
    script = []
    for chunk in split_sections(path.read_text()):
        script.append(f"SELECT '@@section:{chunk.section or ''}' AS snippet_marker;\n")
        for statement in re.split(r";\s*\n", chunk.text.strip().rstrip(";")):
            code_lines = [
                line
                for line in statement.splitlines()
                if not line.lstrip().startswith("--")
            ]
            if "".join(code_lines).strip():
                script.append(
                    f"{statement};\nSELECT '@@statement' AS snippet_marker;\n"
                )
    script_path = output_dir / f"{path.name}.marked"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text("".join(script))

    result = subprocess.run(
        [*command, "-f", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        **run_kwargs,
    )
    script_path.unlink()

    outputs: dict[str, list[str]] = {}
    section, statement_lines = None, []
    for line in result.stdout.splitlines():
        if line == "snippet_marker":
            continue
        if line.startswith("@@section:"):
            section = line.removeprefix("@@section:") or None
        elif line == "@@statement":
            # DDL and DML print a header with no rows (e.g. "Response code").
            if section is not None and len(statement_lines) > 1:
                outputs.setdefault(section, []).append("\n".join(statement_lines))
            statement_lines = []
        else:
            statement_lines.append(line)
    write_outputs(
        {section: "\n\n".join(results) for section, results in outputs.items()},
        path.name,
        output_dir,
    )


if __name__ == "__main__":
    run_python(Path(sys.argv[1]), Path(sys.argv[2]))
