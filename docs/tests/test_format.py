"""Regression tests for rendered format specifications."""

import re
import unittest
from html import unescape
from pathlib import Path
from tempfile import TemporaryDirectory

from mkdocs.commands.build import build
from mkdocs.config import load_config

DOCS_DIR = Path(__file__).resolve().parents[1]


class ProtobufTests(unittest.TestCase):
    def test_range_messages_keep_their_own_fields(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "src"
            sources.mkdir()
            for name, path, heading in (
                ("encoding", "file/encoding.md", "### Range\n"),
                (
                    "row_ids",
                    "table/row_id_lineage.md",
                    "##### Range (Contiguous Values)\n",
                ),
            ):
                page = (DOCS_DIR / "src/format" / path).read_text(encoding="utf-8")
                section = page.split(heading, 1)[1].split("\n#", 1)[0]
                (sources / f"{name}.md").write_text(
                    "# Range\n" + section, encoding="utf-8"
                )

            config = load_config(
                config_file=str(DOCS_DIR / "mkdocs.yml"),
                docs_dir=str(sources),
                site_dir=str(root / "site"),
                nav=[{"Encoding": "encoding.md"}, {"Row IDs": "row_ids.md"}],
                strict=True,
            )
            build(config)
            for name, expected, unexpected in (
                (
                    "encoding",
                    (
                        "uint64 uncompressed_bits_per_value = 1;",
                        "uint64 start = 2;",
                        "uint64 step = 3;",
                    ),
                    "uint64 end = 2;",
                ),
                (
                    "row_ids",
                    ("uint64 start = 1;", "uint64 end = 2;"),
                    "uncompressed_bits_per_value",
                ),
            ):
                with self.subTest(page=name):
                    html = (root / "site" / name / "index.html").read_text(
                        encoding="utf-8"
                    )
                    code = re.search(r"<pre\b.*?</pre>", html, re.DOTALL)
                    self.assertIsNotNone(code)
                    rendered = unescape(re.sub("<[^>]+>", "", code.group()))
                    for field in expected:
                        self.assertIn(field, rendered)
                    self.assertNotIn(unexpected, rendered)
