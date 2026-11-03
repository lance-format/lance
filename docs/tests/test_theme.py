"""Regression tests for the rendered MkDocs theme."""

import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from mkdocs.commands.build import build
from mkdocs.config import load_config

DOCS_DIR = Path(__file__).resolve().parents[1]


class SidenavTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "index.md",
                "spark/index.md",
                "spark/ddl/create.md",
                "trino/ddl/create.md",
                "standalone.md",
            ):
                source = root / "src" / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("# Test page\n", encoding="utf-8")

            config = load_config(
                config_file=StringIO("""
nav:
  - Integrations:
      - Overview: index.md
      - Apache Spark:
          - Welcome: spark/index.md
          - DDL:
              - CREATE NAMESPACE: spark/ddl/create.md
          - Project: https://github.com/lance-format/lance-spark
      - Trino:
          - Operations:
              - DDL:
                  - CREATE SCHEMA: trino/ddl/create.md
  - Standalone: standalone.md
"""),
                site_name="Theme test",
                docs_dir=str(root / "src"),
                site_dir=str(root / "site"),
                theme={"name": None, "custom_dir": str(DOCS_DIR / "theme")},
                plugins=[],
            )
            build(config)
            cls.sidebars = {}
            for name in ("spark/ddl/create/index.html", "standalone/index.html"):
                html = (root / "site" / name).read_text(encoding="utf-8")
                sidebar = html.split('<aside class="ld-sidenav"', 1)[1].split(
                    "</aside>", 1
                )[0]
                cls.sidebars[name] = ElementTree.fromstring(
                    "<aside" + sidebar + "</aside>"
                )

    def test_nested_sections_keep_their_parents(self):
        sidebar = self.sidebars["spark/ddl/create/index.html"]
        groups = sidebar.findall("./div[@class='ld-sidenav__group']")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].find("div").text, "Integrations")
        spark, trino = groups[0].findall("./div[@class='ld-sidenav__group']")
        self.assertEqual(spark.find("div").text, "Apache Spark")
        self.assertEqual(trino.find("div").text, "Trino")
        spark_ddl = spark.find("./div[@class='ld-sidenav__group']")
        self.assertEqual(spark_ddl.find("div").text, "DDL")
        self.assertEqual(spark_ddl.find("a").text, "CREATE NAMESPACE")
        operations = trino.find("./div[@class='ld-sidenav__group']")
        self.assertEqual(operations.find("div").text, "Operations")
        trino_ddl = operations.find("./div[@class='ld-sidenav__group']")
        self.assertEqual(trino_ddl.find("div").text, "DDL")
        self.assertEqual(trino_ddl.find("a").text, "CREATE SCHEMA")

    def test_links_keep_order_urls_and_active_state(self):
        sidebar = self.sidebars["spark/ddl/create/index.html"]
        spark = sidebar.find("./div/div[@class='ld-sidenav__group']")
        self.assertEqual([child.tag for child in spark], ["div", "a", "div", "a"])
        welcome, project = spark.findall("a")
        self.assertEqual(welcome.attrib, {"href": "../../"})
        self.assertEqual(
            project.attrib,
            {
                "href": "https://github.com/lance-format/lance-spark",
                "target": "_blank",
                "rel": "noopener",
            },
        )
        active = sidebar.findall(".//a[@class='active']")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].text, "CREATE NAMESPACE")
        self.assertEqual(active[0].get("href"), "./")

    def test_top_level_page(self):
        sidebar = self.sidebars["standalone/index.html"]
        self.assertEqual(sidebar.find("./div/div").text, "Standalone")
        self.assertEqual(
            sidebar.find("./div/a").attrib, {"class": "active", "href": "./"}
        )
