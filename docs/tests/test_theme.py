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
                "guide/read_and_write.md",
                "guide/spark/index.md",
                "guide/spark/ddl/create.md",
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
  - User Guide:
      - PyLance:
          - Core Operations:
              - Read and Write: guide/read_and_write.md
      - LanceDB: https://docs.lancedb.com/
      - Apache Spark:
          - Welcome: guide/spark/index.md
          - Operations:
              - DDL:
                  - CREATE NAMESPACE: guide/spark/ddl/create.md
  - Integrations:
      - External Docs: https://docs.lancedb.com/
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
            cls.topnavs = {}
            for name in (
                "guide/read_and_write/index.html",
                "guide/spark/ddl/create/index.html",
                "spark/ddl/create/index.html",
                "standalone/index.html",
            ):
                html = (root / "site" / name).read_text(encoding="utf-8")
                topnav = html.split('<nav class="ld-topnav"', 1)[1].split("</nav>", 1)[
                    0
                ]
                cls.topnavs[name] = ElementTree.fromstring("<nav" + topnav + "</nav>")
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
        spark, trino = groups[0].findall("./details[@class='ld-sidenav__sub']")
        self.assertEqual(spark.attrib.get("open"), "open")
        self.assertIsNone(trino.attrib.get("open"))
        self.assertEqual(spark.find("./summary/span").text, "Apache Spark")
        self.assertEqual(trino.find("./summary/span").text, "Trino")
        spark_ddl = spark.find("./div/details[@class='ld-sidenav__sub']")
        self.assertEqual(spark_ddl.attrib.get("open"), "open")
        self.assertEqual(spark_ddl.find("./summary/span").text, "DDL")
        self.assertEqual(spark_ddl.find("./div/a").text, "CREATE NAMESPACE")
        operations = trino.find("./div/details[@class='ld-sidenav__sub']")
        self.assertIsNone(operations.attrib.get("open"))
        self.assertEqual(operations.find("./summary/span").text, "Operations")
        trino_ddl = operations.find("./div/details[@class='ld-sidenav__sub']")
        self.assertEqual(trino_ddl.find("./summary/span").text, "DDL")
        self.assertEqual(trino_ddl.find("./div/a").text, "CREATE SCHEMA")

    def test_links_keep_order_urls_and_active_state(self):
        sidebar = self.sidebars["spark/ddl/create/index.html"]
        spark_items = sidebar.find("./div/details[@class='ld-sidenav__sub']/div")
        self.assertEqual([child.tag for child in spark_items], ["a", "details", "a"])
        welcome, project = spark_items.findall("a")
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
        tabs = self.topnavs["standalone/index.html"].findall("a")
        self.assertEqual(tabs[0].text, "User Guide")
        self.assertEqual(tabs[0].get("href"), "../guide/read_and_write/")
        self.assertEqual(tabs[1].text, "Integrations")
        self.assertEqual(tabs[1].get("href"), "..")

    def test_track_switcher_prioritizes_pylance_and_lancedb_and_scopes_tree(self):
        pylance_sidebar = self.sidebars["guide/read_and_write/index.html"]
        tracks = pylance_sidebar.findall("./div/div[@class='ld-sidenav__tracks']/a")
        self.assertEqual(
            [a.find("span").text for a in tracks],
            ["PyLance", "LanceDB", "Apache Spark"],
        )
        self.assertIn("active", tracks[0].attrib.get("class", ""))
        self.assertEqual(tracks[1].attrib.get("href"), "https://docs.lancedb.com/")
        groups = pylance_sidebar.findall("./div[@class='ld-sidenav__group']")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].find("div").text, "Core Operations")
        self.assertEqual(groups[0].find("a").text, "Read and Write")

        spark_sidebar = self.sidebars["guide/spark/ddl/create/index.html"]
        spark_tracks = spark_sidebar.findall("./div/div[@class='ld-sidenav__tracks']/a")
        self.assertIn("active", spark_tracks[2].attrib.get("class", ""))
        spark_groups = spark_sidebar.findall("./div[@class='ld-sidenav__group']")
        self.assertEqual(len(spark_groups), 2)
        self.assertIsNone(spark_groups[0].find("./div[@class='ld-sidenav__label']"))
        self.assertEqual(spark_groups[0].find("a").text, "Welcome")
        self.assertEqual(spark_groups[1].find("div").text, "Operations")
        ddl = spark_groups[1].find("./details[@class='ld-sidenav__sub']")
        self.assertEqual(ddl.attrib.get("open"), "open")
        self.assertEqual(ddl.find("./summary/span").text, "DDL")
