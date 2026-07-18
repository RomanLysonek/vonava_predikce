import re
from pathlib import Path

from dashboard_artifacts import check_static_dashboard


ROOT = Path(__file__).resolve().parents[1]
AUTHORED = ROOT / "webapp" / "static"
GENERATED = ROOT / "docs"
PAGES = ("index.html", "dataset.html", "evaluation.html", "model.html")
TITLE = "NOTINO - predikce"
PROMO_CHILDREN = (
    '<a class="promo-dataset-link" data-dataset-link href="{dataset_href}" '
    'title="The data structure and the modeling decisions it forced">'
    "30 Product Time Series</a>",
    '<span id="promo-strategy">7-Day Forecast</span>',
    '<span id="promo-model-count">Models Compared</span>',
    '<a class="promo-evaluation-link" data-evaluation-link '
    'href="{evaluation_href}" title="Rolling forecast origins; not the same '
    'thing as recursive inference">Walk-Forward Validated</a>',
)


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


def _promo_bar(html: str, page: str) -> str:
    bars = re.findall(
        r'<div class="promo-bar">\s*(.*?)\s*</div>',
        html,
        flags=re.DOTALL,
    )
    assert len(bars) == 1, f"{page} must contain exactly one promo bar"
    return "\n".join(line.strip() for line in bars[0].splitlines())


def test_authored_and_generated_pages_share_exact_promo_markup():
    hrefs = {
        AUTHORED: ("/dataset", "/evaluation"),
        GENERATED: ("./dataset.html", "./evaluation.html"),
    }
    for directory, (dataset_href, evaluation_href) in hrefs.items():
        expected = "\n".join(
            child.format(
                dataset_href=dataset_href,
                evaluation_href=evaluation_href,
            )
            for child in PROMO_CHILDREN
        )
        bars = {
            page: _promo_bar(
                (directory / page).read_text(encoding="utf-8"),
                f"{directory.name}/{page}",
            )
            for page in PAGES
        }
        assert set(bars.values()) == {expected}


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


def test_promo_geometry_is_exact_and_responsive():
    css = (AUTHORED / "styles.css").read_text(encoding="utf-8")
    desktop = """\
.promo-bar {
  position: relative;
  z-index: 1;
  box-sizing: border-box;
  width: 100%;
  min-height: 40px;
  margin: 0;
  padding: 8px var(--page-padding-inline);
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  align-items: center;
  column-gap: 24px;
  background: #fff;
  border-bottom: 1px solid var(--hairline);
  font-size: 10px;
  line-height: 1.2;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text);
}"""
    children = """\
.promo-bar > * {
  min-width: 0;
  white-space: nowrap;
}"""
    tablet = """\
@media (max-width: 700px) {
  .promo-bar {
    min-height: 57px;
    padding: 8px 24px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    column-gap: 24px;
    row-gap: 8px;
  }"""
    mobile = """\
@media (max-width: 480px) {
  .promo-bar {
    min-height: 89px;
    grid-template-columns: minmax(0, 1fr);
    row-gap: 8px;
  }"""
    for contract in (desktop, children, tablet, mobile):
        assert contract in css
    assert "display: flex;" not in desktop
    assert "width: 60%;" not in css
    assert "font-size: clamp(8.5px, 0.72vw, 10.5px);" not in css


def test_browser_title_cannot_be_mutated_by_javascript():
    for directory in (AUTHORED, GENERATED):
        for script in directory.glob("*.js"):
            source = script.read_text(encoding="utf-8")
            assert "document.title" not in source, script
            assert "page-title" not in source, script
