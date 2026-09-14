"""MkDocs hook: render the PMC roster table from `pmc.yaml` at build time.

`docs/src/community/pmc.yaml` is the source of truth for the PMC roster (it also
drives the format-spec vote gate). The roster page contains the placeholder
`<!-- PMC_ROSTER_TABLE -->`; this hook expands it into a Markdown table when the
docs are built, so the table never has to be maintained by hand.

Registered via `hooks:` in `mkdocs.yml`.
"""

import pathlib

import yaml

PLACEHOLDER = "<!-- PMC_ROSTER_TABLE -->"

COLUMNS = [
    ("Name", "name"),
    ("GitHub Handle", "handle"),
    ("Affiliation", "affiliation"),
    ("Ecosystem Roles", "ecosystem_roles"),
]


def _render_table(members):
    headers = [title for title, _ in COLUMNS]
    rows = [[str(m.get(key, "") or "") for _, key in COLUMNS] for m in members]
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(len(COLUMNS))
    ]

    def row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [row(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines.extend(row(r) for r in rows)
    return "\n".join(lines)


def on_page_markdown(markdown, page, config, files):
    if PLACEHOLDER not in markdown:
        return markdown
    roster_path = pathlib.Path(config["docs_dir"]) / "community" / "pmc.yaml"
    roster = yaml.safe_load(roster_path.read_text())
    return markdown.replace(PLACEHOLDER, _render_table(roster["members"]))
