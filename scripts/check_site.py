#!/usr/bin/env python
"""Strict checks on the rendered Quarto site (site/_site).

- every internal link (href/src) resolves to a file in the output directory
- every in-page anchor (#id) exists in the target page
- every module page has the eight required sections as level-2 headings
- every module page's Overview opens with "**Why this module exists.**", and every
  `### Step N — title` opens with "**What this computes.**" (checked in the source)
- no leftover MkDocs syntax (`!!!`, `--8<--`) in the source pages

Exit status 1 on any failure. Run after `quarto render site`.
"""
from __future__ import annotations

import pathlib
import re
import sys
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE_SRC = ROOT / "site"
OUT = SITE_SRC / "_site"
REQUIRED = ["Overview", "Learning objectives", "Background", "Steps", "Checkpoint", "Challenge", "Going deeper", "Troubleshooting"]
OVERVIEW_OPENER = "**Why this module exists.**"
STEP_OPENER = "**What this computes.**"
STEP_HEADING = re.compile(r"^### Step \d+ — \S")


def _next_paragraph(lines: list[str], i: int) -> str:
    """The first non-blank line after line i."""
    return next((line for line in lines[i + 1 :] if line.strip()), "")


def template_errors(src: pathlib.Path) -> list[str]:
    """Openers the module-page template requires (AGENTS.md, Site)."""
    rel = src.relative_to(ROOT)
    lines = src.read_text().splitlines()
    errors = []
    for i, line in enumerate(lines):
        if line.strip() == "## Overview" and not _next_paragraph(lines, i).startswith(OVERVIEW_OPENER):
            errors.append(f"{rel}:{i + 1}: Overview does not open with {OVERVIEW_OPENER}")
        if line.startswith("### Step"):
            if not STEP_HEADING.match(line):
                errors.append(f"{rel}:{i + 1}: step heading is not '### Step N — title'")
            if not _next_paragraph(lines, i).startswith(STEP_OPENER):
                errors.append(f"{rel}:{i + 1}: step does not open with {STEP_OPENER}")
    if "## Overview" not in lines:
        errors.append(f"{rel}: no Overview section")
    return errors


class Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.h2: list[str] = []
        self._in_h2 = False
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag in ("script", "link", "img") and (a.get("src") or a.get("href")):
            self.links.append(a.get("src") or a.get("href"))
        if tag == "h2":
            self._in_h2, self._buf = True, ""

    def handle_endtag(self, tag):
        if tag == "h2" and self._in_h2:
            self.h2.append(self._buf.strip())
            self._in_h2 = False

    def handle_data(self, data):
        if self._in_h2:
            self._buf += data


def main() -> int:
    errors: list[str] = []
    if not OUT.exists():
        print("site/_site missing; run `quarto render site` first")
        return 1
    pages = {p: Collector() for p in OUT.rglob("*.html")}
    for p, c in pages.items():
        c.feed(p.read_text(errors="ignore"))
    for p, c in pages.items():
        rel = p.relative_to(OUT)
        for href in c.links:
            if re.match(r"^(https?:|mailto:|data:|javascript:)", href):
                continue
            target, _, frag = href.partition("#")
            if target == "":
                tpath = p
            else:
                tpath = (p.parent / target).resolve()
                if tpath.is_dir():
                    tpath = tpath / "index.html"
                if not tpath.exists():
                    errors.append(f"{rel}: broken link {href}")
                    continue
            if frag and tpath.suffix == ".html":
                ids = pages[tpath].ids if tpath in pages else set()
                if frag not in ids:
                    errors.append(f"{rel}: missing anchor #{frag} in {tpath.relative_to(OUT)}")
        if re.match(r"day\d/m\d", str(rel)) and rel.name != "index.html":
            missing = [r for r in REQUIRED if not any(h.startswith(r) for h in c.h2)]
            if missing:
                errors.append(f"{rel}: missing sections {missing}")
    for src in sorted(SITE_SRC.glob("day*/m*.qmd")):
        errors += template_errors(src)
    for src in SITE_SRC.rglob("*.qmd"):
        text = src.read_text()
        if re.search(r"^\s*!!!\s", text, re.M) or "--8<--" in text:
            errors.append(f"{src.relative_to(ROOT)}: leftover MkDocs syntax")
    if errors:
        print("\n".join(errors))
        print(f"{len(errors)} problem(s)")
        return 1
    n_mod = sum(1 for p in pages if re.match(r"day\d/m\d", str(p.relative_to(OUT))))
    print(f"OK: {len(pages)} pages, {n_mod} module pages with all eight sections and the template openers, all internal links and anchors resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
