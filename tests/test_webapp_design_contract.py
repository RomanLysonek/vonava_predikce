import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard_artifacts import check_static_dashboard


ROOT = Path(__file__).resolve().parents[1]
AUTHORED = ROOT / "webapp" / "static"
GENERATED = ROOT / "docs"
PAGES = (
    "index.html",
    "dataset.html",
    "evaluation.html",
    "model.html",
    "whole-story.html",
)
TITLE = "NOTINO - predikce"
CHROME_CANDIDATES = (
    "google-chrome",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
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


def _chrome_binary() -> str | None:
    for candidate in CHROME_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.is_file():
            return str(path)
    return None


def _render_promo_geometry(
    tmp_path: Path,
    chrome: str,
    viewport_width: int,
) -> dict:
    shutil.copy2(AUTHORED / "styles.css", tmp_path / "styles.css")
    probe = tmp_path / f"promo-{viewport_width}.html"
    probe.write_text(
        """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <div class="promo-bar">
    <a>30 Product Time Series</a>
    <span>7-Day Direct multi-horizon Forecast</span>
    <span>7 Models Compared</span>
    <a>Walk-Forward Validated</a>
  </div>
  <pre id="result"></pre>
  <script>
    function measurePromo() {
      const bar = document.querySelector(".promo-bar");
      const children = [...bar.children];
      const textRects = children.map((child) => {
        const range = document.createRange();
        range.selectNodeContents(child);
        const rect = range.getBoundingClientRect();
        return {
          left: rect.left,
          right: rect.right,
          top: rect.top,
          bottom: rect.bottom,
        };
      });
      const overlaps = [];
      for (let left = 0; left < textRects.length; left += 1) {
        for (let right = left + 1; right < textRects.length; right += 1) {
          const a = textRects[left];
          const b = textRects[right];
          const sameRow = a.top < b.bottom && b.top < a.bottom;
          const intersects = a.left < b.right && b.left < a.right;
          if (sameRow && intersects) overlaps.push([left + 1, right + 1]);
        }
      }
      document.getElementById("result").textContent = JSON.stringify({
        viewportWidth: window.innerWidth,
        columns: getComputedStyle(bar).gridTemplateColumns.split(" ").length,
        alignments: children.map((child) => getComputedStyle(child).textAlign),
        overlaps,
      });
    }
    window.addEventListener("load", () => {
      Promise.race([
        document.fonts.ready,
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]).then(() => setTimeout(measurePromo, 0));
    });
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--force-device-scale-factor=1",
            f"--window-size={viewport_width},300",
            "--virtual-time-budget=3000",
            "--dump-dom",
            probe.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r'<pre id="result">(\{.*?\})</pre>', completed.stdout)
    assert match, completed.stdout
    return json.loads(match.group(1))


def _render_mobile_promo_alignment(tmp_path: Path, chrome: str) -> dict:
    frame = tmp_path / "promo-mobile-frame.html"
    frame.write_text(
        """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <div class="promo-bar">
    <a>30 Product Time Series</a>
    <span>7-Day Direct multi-horizon Forecast</span>
    <span>7 Models Compared</span>
    <a>Walk-Forward Validated</a>
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )
    probe = tmp_path / "promo-mobile-probe.html"
    probe.write_text(
        """<!DOCTYPE html>
<html>
<body>
  <pre id="result"></pre>
  <script>
    function measureMobile(frame) {
      const view = frame.contentWindow;
      const children = [...view.document.querySelector(".promo-bar").children];
      document.getElementById("result").textContent = JSON.stringify({
        mediaMatches: view.matchMedia("(max-width: 480px)").matches,
        alignments: children.map((child) => view.getComputedStyle(child).textAlign),
      });
    }
  </script>
  <iframe id="frame" src="./promo-mobile-frame.html" onload="measureMobile(this)"
          style="width:480px;height:180px;border:0"></iframe>
</body>
</html>
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--force-device-scale-factor=1",
            "--window-size=600,300",
            "--virtual-time-budget=3000",
            "--dump-dom",
            probe.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r'<pre id="result">(\{.*?\})</pre>', completed.stdout)
    assert match, completed.stdout
    return json.loads(match.group(1))


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

    for selector in (
        ".overview-hero",
        ".dataset-hero",
        ".evaluation-hero",
        ".wholestory-hero",
    ):
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
@media (max-width: 840px) {
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
    assert ".promo-bar > :nth-child(n)" in css
    assert "display: flex;" not in desktop
    assert "width: 60%;" not in css
    assert "font-size: clamp(8.5px, 0.72vw, 10.5px);" not in css


def test_runtime_promo_labels_have_safe_rendered_geometry(tmp_path):
    chrome = _chrome_binary()
    if not chrome:
        pytest.skip("Chrome/Chromium is required for rendered promo geometry")

    geometry = {
        width: _render_promo_geometry(tmp_path, chrome, width)
        for width in (701, 776, 800, 801, 840, 841, 900, 901)
    }
    for width, measurement in geometry.items():
        assert measurement["viewportWidth"] == width
        assert measurement["overlaps"] == [], (width, measurement)

    for width in (701, 776, 800, 801, 840):
        assert geometry[width]["columns"] == 2
    for width in (841, 900, 901):
        assert geometry[width]["columns"] == 4

    mobile = _render_mobile_promo_alignment(tmp_path, chrome)
    assert mobile["mediaMatches"] is True
    assert mobile["alignments"] == ["left"] * 4


def test_browser_title_cannot_be_mutated_by_javascript():
    for directory in (AUTHORED, GENERATED):
        for script in directory.glob("*.js"):
            source = script.read_text(encoding="utf-8")
            assert "document.title" not in source, script
            assert "page-title" not in source, script
