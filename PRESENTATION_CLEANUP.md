# Presentation cleanup

This document records the second, presentation-only cleanup applied to the
original English repository. The cleanup does **not** change model code, input
data, stored predictions, metric values, dashboard behavior or the final model
decision.

## Removed working material

- IDE configuration under `.idea/`;
- local run logs matching `pipeline_*.log` and the working note `progress.txt`;
- the exploratory notebook `playing_around.ipynb`;
- the obsolete alternative output branch `outputs_c1_half365/`;
- fold checkpoints and final-audit checkpoints;
- per-candidate OOF files and checkpoints from the C1, C2 and C3/C4 screens;
- Python, pytest and notebook caches;
- empty optional summaries;
- duplicate strategy-specific submission copies, while preserving the canonical
  submission and the explicit cross-model ensemble alternative.

These files are either machine-local or reproducible from the retained source,
recommendation JSON and commands in `RETRAINING_AND_CLI_GUIDE.md`.

## Retained evidence

- both input Parquet files;
- all source code and tests;
- the canonical NeuralNet/direct submission;
- the frozen cross-model ensemble weights and alternative submission;
- complete main-run OOF predictions and final forecasts;
- the disjoint final-audit OOF and summaries;
- result tables and recommendations from every staged screening phase;
- the local FastAPI dashboard and the static GitHub Pages build.

## Presentation-facing corrections

`README.md` now starts with the confirmed final state rather than the archived
pre-C0 benchmark. It explicitly documents:

- NeuralNet/direct as the canonical submission;
- test-aligned WAPE as the frozen selection objective;
- the selected C3/C4 target and loss configuration;
- the frozen ensemble weights;
- the untouched final-audit comparison that kept NeuralNet canonical;
- optional staged research runners grouped under `ml/experiments/`, leaving
  `ml/pipeline.py` as the unambiguous final training entry point.

## Verification target

The cleaned package should satisfy all of the following:

```bash
uv sync --frozen --group dev
uv run pytest tests/ -m "not integration" -q
node tests/webapp_smoke_test.js
python -m compileall -q ml webapp tests
```

The frontend source in `webapp/static/` and the GitHub Pages copy in `docs/`
should remain byte-for-byte identical for their shared files.
