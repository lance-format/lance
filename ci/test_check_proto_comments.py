"""Unit tests for the .proto multi-line comment style check.

Run with: pytest ci/test_check_proto_comments.py
"""

import pytest

from check_proto_comments import rewrite

HEADER = "// SPDX-License-Identifier: Apache-2.0\n// SPDX-FileCopyrightText: Copyright The Lance Authors\n"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(HEADER, id="license_header"),
        pytest.param("// A single line comment.\nmessage M {}\n", id="single_line"),
        pytest.param("/// A single doc comment.\nmessage M {}\n", id="single_doc_line"),
        pytest.param(
            "message M {\n  int32 a = 1; // Trailing.\n  int32 b = 2; // Also trailing.\n}\n",
            id="trailing",
        ),
        pytest.param(
            "/* Already a block\n * comment.\n */\nmessage M {}\n", id="already_block"
        ),
    ],
)
def test_leaves_compliant_comments_alone(source):
    assert rewrite(source) == source


def test_rewrites_multi_line_run_as_block():
    source = "// First line.\n// Second line.\nmessage M {}\n"
    assert rewrite(source) == "/* First line.\n * Second line.\n */\nmessage M {}\n"


def test_preserves_content_columns_and_indent():
    source = "message M {\n  // Offsets are laid out as:\n  //   i == 0: 0\n  //   i >  0: prev\n  int32 a = 1;\n}\n"
    assert rewrite(source) == (
        "message M {\n  /* Offsets are laid out as:\n   *   i == 0: 0\n   *   i >  0: prev\n   */\n  int32 a = 1;\n}\n"
    )


def test_blank_comment_lines_carry_no_trailing_whitespace():
    source = "// First paragraph.\n//\n// Second paragraph.\nmessage M {}\n"
    assert (
        rewrite(source)
        == "/* First paragraph.\n *\n * Second paragraph.\n */\nmessage M {}\n"
    )


def test_rewrites_doc_comment_run():
    source = "/// First line.\n/// Second line.\nmessage M {}\n"
    assert rewrite(source) == "/* First line.\n * Second line.\n */\nmessage M {}\n"


def test_splits_runs_at_differing_indent():
    source = "// Outer.\n// Outer continued.\nmessage M {\n  // Inner.\n  // Inner continued.\n  int32 a = 1;\n}\n"
    assert rewrite(source) == (
        "/* Outer.\n * Outer continued.\n */\nmessage M {\n  /* Inner.\n   * Inner continued.\n   */\n  int32 a = 1;\n}\n"
    )


def test_keeps_license_header_when_file_starts_with_it():
    source = HEADER + "\n// Doc line one.\n// Doc line two.\nmessage M {}\n"
    assert (
        rewrite(source)
        == HEADER + "\n/* Doc line one.\n * Doc line two.\n */\nmessage M {}\n"
    )


def test_is_idempotent():
    once = rewrite("// First line.\n// Second line.\nmessage M {}\n")
    assert rewrite(once) == once


def test_converts_a_comment_run_that_abuts_the_license_header():
    source = HEADER + "// Doc line one.\n// Doc line two.\nmessage M {}\n"
    assert (
        rewrite(source)
        == HEADER + "/* Doc line one.\n * Doc line two.\n */\nmessage M {}\n"
    )
