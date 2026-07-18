import re
from pathlib import Path

from dashboard_artifacts import check_static_dashboard


ROOT = Path(__file__).resolve().parents[1]
AUTHORED = ROOT / "webapp" / "static"
GENERATED = ROOT / "docs"
PAGES = ("index.html", "dataset.html", "evaluation.html", "model.html")
TITLE = "NOTINO - predikce"


def _description_strip(html: str, page: str) -> str:
    strips = re.findall(
        r'<header class="[^"]*\bdescription-strip\b[^"]*"[^>]*>.*?</header>',
        html,
        flags=re.DOTALL,
    )
    assert len(strips) == 1, f"{page} must contain exactly one description strip"

    shared_hero = re.search(
        r'<header class="hero">.*?</header>',
        html,
        flags=re.DOTALL,
    )
    assert shared_hero, f"{page} is missing the shared hero/navigation"
    assert html[shared_hero.end():].lstrip().startswith(
        '<header class="description-strip'
    ), f"{page} description strip must immediately follow the shared hero"
    return strips[0]


def test_authored_and_generated_pages_share_description_contract():
    for directory in (AUTHORED, GENERATED):
        assert tuple(sorted(path.name for path in directory.glob("*.html"))) == tuple(
            sorted(PAGES)
        )
        for page in PAGES:
            html = (directory / page).read_text(encoding="utf-8")
            assert re.findall(r"<title>(.*?)</title>", html, flags=re.DOTALL) == [
                TITLE
            ]
            strip = _description_strip(html, f"{directory.name}/{page}")
            assert 'class="model-badge"' in strip
            assert re.search(r"<h1(?:\s[^>]*)?>.+?</h1>", strip, flags=re.DOTALL)
            assert re.search(
                r'<p class="blurb"(?:\s[^>]*)?>.*?</p>',
                strip,
                flags=re.DOTALL,
            )

    assert check_static_dashboard(ROOT) == []


def test_description_geometry_has_one_shared_owner():
    css = (AUTHORED / "styles.css").read_text(encoding="utf-8")
    block_match = re.search(r"\.description-strip\s*\{([^}]*)\}", css, re.DOTALL)
    assert block_match
    block = block_match.group(1)
    for declaration in (
        "box-sizing: border-box;",
        "width: 100%;",
        "max-width: none;",
        "min-height: var(--description-strip-min-height);",
        "margin: 0;",
        "padding: var(--description-strip-padding-block) var(--page-padding-inline);",
        "border-bottom: var(--description-strip-border-width) solid var(--mc);",
    ):
        assert declaration in block

    variables = {
        "--page-padding-inline": "56px",
        "--description-strip-padding-block": "40px",
        "--description-strip-border-width": "6px",
        "--description-strip-min-height": "300px",
    }
    for name, value in variables.items():
        assert css.count(f"{name}: {value};") == 1
    assert "@media (max-width: 900px)" in css
    assert ":root { --page-padding-inline: 24px; }" in css
    assert "scrollbar-gutter: stable;" in css

    for selector in (".overview-hero", ".dataset-hero", ".evaluation-hero"):
        match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", css, re.DOTALL)
        assert match
        declarations = [
            declaration.strip()
            for declaration in match.group(1).split(";")
            if declaration.strip()
        ]
        assert len(declarations) == 1
        assert declarations[0].startswith("--mc:")


def test_browser_title_cannot_be_mutated_by_javascript():
    for directory in (AUTHORED, GENERATED):
        for script in directory.glob("*.js"):
            source = script.read_text(encoding="utf-8")
            assert "document.title" not in source, script
            assert "page-title" not in source, script
