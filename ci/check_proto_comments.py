#!/usr/bin/env python3
"""Require `/* ... */` for multi-line comments in .proto files.

Multi-line `//` runs cannot be folded by editors, which makes the long design
notes in these schemas hard to skim. The `/* text` + ` * text` layout keeps every
content column exactly where the `//` form had it, so protoc extracts identical
leading_comments and the generated Rust, Python and Java doc comments do not
change.

The SPDX header is exempt; license-header-check requires it verbatim.
"""

import argparse
import re
import sys
from pathlib import Path

DEFAULT_ROOTS = (".",)
SKIP_DIRS = {".git", ".venv", "node_modules", "target"}

COMMENT_LINE = re.compile(r"^(\s*)(///?)(.*)$")
SPDX_LINE = re.compile(r"^//\s*SPDX-")


def violations(lines):
    """Yield (start, indent, contents) for each multi-line `//` comment run."""
    i = 0
    while i < len(lines) and SPDX_LINE.match(lines[i]):
        i += 1
    while i < len(lines):
        match = COMMENT_LINE.match(lines[i])
        if match is None:
            i += 1
            continue
        indent = match.group(1)
        contents = []
        j = i
        while j < len(lines):
            line = COMMENT_LINE.match(lines[j])
            if line is None or line.group(1) != indent:
                break
            contents.append(line.group(3))
            j += 1
        if len(contents) > 1:
            yield i, indent, contents
        i = j


def rewrite(text):
    lines = text.split("\n")
    out = []
    i = 0
    for start, indent, contents in violations(lines):
        out.extend(lines[i:start])
        out.append((indent + "/*" + contents[0]).rstrip())
        out.extend((indent + " *" + content).rstrip() for content in contents[1:])
        out.append(indent + " */")
        i = start + len(contents)
    out.extend(lines[i:])
    return "\n".join(out)


def collect(paths):
    files = []
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            files.append(path)
            continue
        found = path.rglob("*.proto")
        files.extend(sorted(p for p in found if SKIP_DIRS.isdisjoint(p.parts)))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=DEFAULT_ROOTS, help="files or directories to check"
    )
    parser.add_argument(
        "--fix", action="store_true", help="rewrite offending comments in place"
    )
    args = parser.parse_args(argv)

    failed = False
    for file in collect(args.paths):
        original = file.read_text()
        offenders = list(violations(original.split("\n")))
        if not offenders:
            continue
        failed = True
        if args.fix:
            file.write_text(rewrite(original))
            print(f"{file}: rewrote {len(offenders)} multi-line comment(s)")
        else:
            for start, _, contents in offenders:
                print(
                    f"{file}:{start + 1}: multi-line comment must use /* ... */ ({len(contents)} lines)"
                )

    if failed and not args.fix:
        print(
            "\nRun `python ci/check_proto_comments.py --fix` to convert them.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
