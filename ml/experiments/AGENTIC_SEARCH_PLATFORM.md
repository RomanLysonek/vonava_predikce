# Agentic Champion/Challenger Search — a platform for improving the forecasting models

This documents the reusable method + tooling built to systematically search for
pipeline configurations that beat the current champion, *without* fooling
ourselves with noise or test-set overfitting.

Two artifacts make up the platform:
- `ml/experiments/champion_challenger_search.py` — the harness (run / record /
  report / propose, plus a CSV ledger).
- This methodology note — the loop the harness encodes and the guardrails.

--------------------------------------------------------------------------------
## 1. The problem

The pipeline's submitted primary is **NeuralNet** with test-aligned WAPE
**0.276149**; the secondary **Ensemble** is **0.253249**. Goal: find a config
that beats the NeuralNet primary on test-aligned WAPE by a *trustworthy* margin,
and generalize the procedure.

## 2. Why a naive search is wrong

Two traps, both of which we measured directly:

1. **Run-to-run nondeterminism.** Re-running the *exact champion config* (the
   E0 "control") did **not** reproduce the committed numbers:
   - NN CV-WAPE:   0.273459 (committed) vs 0.266147 (control)  -> ~0.007 swing
   - NN test-WAPE: 0.276149 (committed) vs 0.275611 (control)  -> ~0.0005 swing
   NN training on Apple-Silicon MPS is nondeterministic even with fixed seeds.
   Lesson: **CV-WAPE is noisy (+/-~0.007); test-aligned WAPE is stable
   (+/-~0.0005).** A single number is not the truth — you need a control.

2. **Test-set overfitting.** If you generate dozens of challengers and keep the
   one with the best test-aligned score, you are p-hacking the test set.

## 3. The loop (what the platform enforces)

1. **Anchor to the champion**, read from the pipeline's own `results.json`
   (validation CV-WAPE + held-out test-aligned WAPE, per model).
2. **Run a CONTROL** — the champion config unchanged — to measure the noise
   floor. All challengers are compared to the control, not the committed number.
3. **One lever at a time** (clean attribution): each challenger = champion BASE
   with a single flag changed. Combine only proven levers afterwards.
4. **Isolated execution**: every run gets its own git worktree + checkpoint dir
   and its `results.json` is snapshotted, so runs never clobber each other.
5. **Noise-aware, test-primary verdict**:
   - decision metric = held-out **test-aligned WAPE** (stable), improvement must
     exceed `TEST_NOISE` (0.0015);
   - **CV-WAPE corroborates** — a test win that CV *contradicts* is flagged
     `MIXED -> confirm`, not trusted blindly;
   - any candidate winner must **survive a repeat run** (test-aligned is stable,
     so a real win reproduces).
6. **Ledger everything** (CSV) and let `propose` suggest the next challengers.

## 4. Lever map (pipeline.py knobs the search can turn)

| lever | champion | alternatives |
|---|---|---|
| `--nn-loss` | mse | huber, combined, logcosh |
| `--nn-target-mode` | residual | log1p |
| `--baseline-variant` | weighted_4321 | weighted_8421, lag7, weekday_median |
| `--trend-features` | off | on |
| `--c2-feature-groups` | all 5 | subsets |
| `--nn-combined-mse-weight`, channel features, ensemble members | ... | ... |

## 5. Results (throughput: full-history runs ~37-40 min solo; 3-way parallel in
git worktrees ~= 15 min/run effective, a ~2.5x win)

Deltas are vs the **control** (fair, noise-aware baseline). Test noise ~0.0015.

| challenger | lever | NN CV | NN test | d_test | verdict |
|---|---|---|---|---|---|
| E0-control | (same as champion) | 0.266147 | 0.275611 | 0 | noise floor |
| E1-huber | nn-loss=huber | 0.269970 | 0.276721 | +0.0011 | flat/worse |
| E2-logcosh | nn-loss=logcosh | 0.259605 | 0.275734 | +0.0001 | CV win, test FLAT |
| E3-log1p | nn-target=log1p | 0.271299 | 0.299469 | +0.0239 | REJECT (much worse) |
| E6-baselag7 | baseline=lag7 | 0.265533 | 0.278740 | +0.0031 | REJECT |
| E7-trendon | trend=on | 0.269617 | 0.277903 | +0.0023 | REJECT |
| **E5-base8421** | **baseline=weighted_8421** | 0.283390 | **0.271347** | **-0.0043** | **CANDIDATE WIN (confirm)** |

Key finding so far: **loss/target NN tweaks do not move the held-out test metric**
beyond noise (logcosh improves the *noisy* CV but is test-flat). The lever that
*does* move test-aligned is the **residual baseline**: `weighted_8421` (a more
recency-tilted same-weekday baseline) improves the held-out (future, winter-
weighted) test for both NeuralNet (-0.0043) and the Ensemble (-0.0020), while
*worsening* interior CV (+0.017). That CV/test divergence is not a red flag here
but a *regime signal*: recency helps the future, not the interior folds — the
same intuition the earlier trailing-window/recency experiment tried too bluntly.

## 6. Confirmation + stacking (Batches 3-4) — the ROBUST WIN

The `weighted_8421` test win **reproduces** (E5 0.271347, E5b 0.272483; both
~0.0043 below champion). Its lone weakness was CV. Stacking a **robust NN loss**
on top *repairs* the CV regression while keeping the test win:

| config | NN CV | NN test | Ens CV | Ens test | vs champion |
|---|---|---|---|---|---|
| champion (weighted_4321, mse) | 0.273459 | 0.276149 | 0.247078 | 0.253249 | — |
| E5 weighted_8421 | 0.283390 | **0.271347** | 0.247628 | **0.251449** | best test, CV worse |
| **E8 weighted_8421 + logcosh** | **0.260547** | **0.273091** | **0.244835** | **0.252207** | **beats on ALL 4** |
| **E9 weighted_8421 + combined** | **0.248228** | **0.273581** | **0.240337** | **0.252254** | **beats on ALL 4, best CV** |

**Headline:** After confirmation repeats, **E8 = `weighted_8421 + logcosh`** is
the robust winner. It reproduces tightly across two independent runs and beats
the champion on **all four** metrics:

| metric | champion | E8 (mean of 2 runs) | delta | rel |
|---|---|---|---|---|
| NeuralNet **test** | 0.276149 | **0.27326** [0.27309, 0.27342] | **-0.00289** | **-1.05%** |
| NeuralNet CV | 0.273459 | 0.25467 | -0.01879 | -6.87% |
| Ensemble test | 0.253249 | 0.25211 | -0.00114 | -0.45% |
| Ensemble CV | 0.247078 | 0.24258 | -0.00449 | -1.82% |

Confirmation nuances (why E8 and not the others):
- **E5 `weighted_8421` alone** has the *best* NN-test (0.27192, -1.5%) but its CV
  is much worse (~0.283) — a fragile, CV-contradicted win. Good for max-test if
  you accept the CV regression, but not the robust pick.
- **E9 `weighted_8421 + combined`** has the *best CV* (0.251) but its NN-test is
  noisy across repeats (0.27358 / 0.27594 — one run nearly ties the champion),
  so its test edge is not reliable.
- **E8 `weighted_8421 + logcosh`** is the sweet spot: consistent NN-test win
  (+/-0.0002 between runs) **and** a solid CV improvement **and** ensemble gains,
  all reproduced. The recency-tilted residual baseline supplies the test win; the
  robust logcosh loss supplies the CV recovery and run-to-run stability.
- **Exploration E12** (add `--channel-history-features on` to E9) was **rejected**
  — it worsened NN-test to 0.2788. Not every extra lever helps; the platform's
  test gate catches this.

## 7. How to use the platform going forward
```
# baseline numbers from any results.json
python ml/experiments/champion_challenger_search.py baseline --results outputs/results.json
# run a challenger end-to-end (BASE + one override) in an isolated worktree
python ml/experiments/champion_challenger_search.py run --name E-try \
    --repo /path/to/worktree --overrides '{"--baseline-variant":"weighted_8421"}'
# ranked ledger + next-step proposals
python ml/experiments/champion_challenger_search.py report
python ml/experiments/champion_challenger_search.py propose
```

## 8. Honest caveats
- The win is on ONE held-out test period; treat test-aligned as a confirmation
  gate, not something to grind against indefinitely. Prefer levers with a causal
  story (recency->future) over pure metric-chasing.
- NN nondeterminism means small deltas need a repeat run; the platform bakes
  this in via the control + confirm step.
