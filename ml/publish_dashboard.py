"""Publish or verify the generated static GitHub Pages dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path

from dashboard_artifacts import check_static_dashboard, publish_static_dashboard


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed generated output differs from authored sources",
    )
    args = parser.parse_args(argv)
    if args.check:
        errors = check_static_dashboard(ROOT)
        if errors:
            raise SystemExit(
                "Static dashboard is stale:\n- " + "\n- ".join(errors)
            )
        print("Static dashboard matches generated output.")
        return
    publish_static_dashboard(
        ROOT,
        ROOT / "outputs" / "results.json",
        ROOT / "outputs" / "run_manifest.json",
    )
    print("Published webapp/static to docs/.")


if __name__ == "__main__":
    main()
