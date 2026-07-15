# Dataset-profile to modeling audit

## Scope

This audit compares the supplied dataset-profile findings with the
current C0-C6 implementation and the persisted final configuration in
`outputs/results.json`.

## Verdict

The dataset profile materially propagated into the modeling system. None of
its ten central modeling consequences was simply forgotten:

- **Eight were adopted directly** in the final data, feature, target,
  evaluation or benchmark contract.
- **Two were implemented and screened but deliberately not retained**:
  exponential recency weighting and channel-share auxiliary modeling.
- Several semantic uncertainties remain impossible to eliminate from the
  supplied fields alone; they are documented rather than hidden.

The final system therefore reflects evidence from both the raw-data profile
and rolling-origin experiments, rather than treating the profile's plausible
recommendations as mandatory conclusions.

## Finding-to-implementation matrix

| Dataset-profile finding | Modeling response | Final status | Evidence in repository |
|---|---|---|---|
| The data is a panel of 30 related daily series, not IID rows | Global models pool all products while preserving `ProductId` through embeddings, native categories or one-hot encoding | Implemented | `ml/framework.py`; model definitions under `ml/models/` |
| 3,547 absent rows are staggered launches and 29 are internal active-history gaps | Reindex each product to a complete daily calendar only between its own first and last row; inserted gaps retain unknown quantity/availability; separate first-row and first-available clocks | Implemented | `reindex_daily_calendar`, `product_reference_dates`, lifecycle features in `ml/framework.py` |
| Pre-launch, unavailable, missing and genuine zero-demand states are semantically different | Explicit observed/available/unavailable/gap state and lifecycle features; no blind zero filling | Implemented | `_availability_state`, `_window_state_features`, C2 lifecycle group |
| `ProductAvailable=False` is a censored-sales signal, not a guaranteed zero; future availability is absent from test | Unavailable targets are excluded from primary model fitting and conditional-demand scoring; availability-aware lags exclude them; future `ProductAvailable` is not a model covariate; realized-sales scoring remains diagnostic | Implemented | `select_trainable_panel_rows`; conditional/common evaluation in `ml/pipeline.py` |
| Campaign subtype IDs are labels, not an ordinal scale | NN embeddings, tree native categorical domains and Ridge one-hot encoding | Implemented | `CAMPAIGN_CATEGORIES`, `TREE_CATEGORICAL_COLUMNS`, model preprocessors |
| Sale flag, coupon subtype and numerical discount have distinct semantics; positive discount can coexist with subtype `-1` | Preserve all fields separately; add campaign-active, app-only, subtype-match, discount-without-campaign and app-discount-advantage features | Implemented and selected | C2 campaign feature group in `prepare_features`; all C2 groups selected |
| Base price is not the customer offer under channel-specific discounts | Derive effective web/app prices and relative effective-price features against recent product history | Implemented and selected | C2 price group in `prepare_features` and `build_direct_panel` |
| Strong day-of-week, annual and event seasonality; test covers one full week | Cyclic calendar features, horizon embedding/feature, availability-aware 7/14/21/28 same-weekday baseline, annual references and explicit Black Friday/Christmas/Valentine/Mother's Day proximity | Implemented and selected where supported | Calendar/event features and target-relative seasonal lags in `ml/framework.py` |
| Promotional extremes are plausible business events, not automatic data errors | No blanket target winsorization; event context, holiday/event validation strata, top-volume diagnostics and prediction-safety guards are used instead | Implemented | C2 event group, validation strata, C3 cap diagnostics, C6 top-decile/error panels |
| Products share market-wide shocks | Add leakage-safe aggregate demand state and future-known cross-sectional campaign/discount intensity; train global pooled models | Implemented and selected | C2 market group in `build_origin_state_features` and `prepare_features` |
| Demand and channel mix are nonstationary; January 2026 is materially lower than prior years | Screen 365/730-day windows, exponential half-lives, trend features and multiple recent seasonal baselines; emphasize January-like periods in the frozen selection objective and keep a recent benchmark | Addressed experimentally; no decay retained | C1 screening artifacts; final config uses all history, no half-life, weighted 4:3:2:1 baseline, trend off |
| App share migrated from roughly 5% to 45% and app-only campaigns are common | Implement channel-history features and an auxiliary app-share head while retaining total quantity as the primary target | Implemented, screened, rejected | C4 code and `outputs/c34_screening/recommendation.json`; final config disables channel head/history |
| Quantities are right-skewed and overdispersed | Predict log counts or baseline-relative log residuals; screen Huber, MSE, mixed, Log-Cosh and Tweedie formulations instead of assuming a simple Poisson model | Implemented and selected per model | Final: NN MSE residual, XGBoost residual, LightGBM log1p |
| Only 30 products and substantial product-scale heterogeneity | Use WAPE as the primary global metric, retain per-product diagnostics and compare on a common population | Implemented | `compute_metrics`, B4/C6 artifacts and dashboard |
| A seven-day test should be compared with recent seasonal methods | Include lag-7 seasonal naive, availability-aware 28-day moving average and 4:3:2:1 same-weekday baseline, plus Ridge/tree comparators | Implemented | `ml/models/naive_baselines.py`, Dynamic Ridge, XGBoost, LightGBM |
| Test covariates are familiar but target level is drifting | Avoid novelty-oriented handling; focus validation on temporal transfer, January-like weighting and recent confirmation | Implemented | Test-aligned WAPE, 12 development origins, 4 recent benchmark origins, 3 frozen audit origins |

## Why two profile recommendations were not retained literally

### 1. “Weight recent data more heavily”

This was treated as a hypothesis, not a rule. C1 tested:

- all history, 730-day and 365-day training windows;
- no decay and 365/180/90-day half-lives;
- multiple same-weekday baselines;
- explicit trend features.

A 365-day half-life initially improved the recent benchmark, but after the C2
semantic feature representation was added, no decay and a 365-day half-life
were effectively tied. The final confirmed configuration retained all eligible
history with no exponential decay because it did not lose the frozen
January-weighted development objective. Recency still enters through the
4:3:2:1 seasonal baseline, recent lags/rolling state, recent benchmark and
January-like validation weighting.

### 2. “Forecast total demand plus app share”

C4 implemented the proposal fully enough to test it:

- lag-0/lag-7 app share;
- volume-weighted 7/28-day channel shares;
- app/web rolling levels;
- recent-versus-long share movement;
- an auxiliary app-share head sharing the NN representation;
- recursive channel-composition propagation.

The channel-aware candidates worsened the selected total-demand objective, so
the auxiliary head and channel-history representation were rejected. This is
a successful negative experiment, not an omitted recommendation.

## Remaining limitations and residual risks

1. **Unavailable sales are not true latent-demand labels.** Conditional
   training/scoring avoids obvious stockout distortion, but inventory amount,
   stockout timing and lost sales are unavailable. The system cannot identify
   fully unconstrained demand causally.

2. **Campaign subtype 5 is too rare for a stable dedicated effect.** Global
   pooling and categorical regularization reduce variance, but no reliable
   subtype-specific response should be claimed. It is absent from the supplied
   test week, lowering immediate risk.

3. **Price anomalies were not corrected without business confirmation.** The
   system uses relative/effective prices and lifecycle context but preserves
   the supplied values. This is preferable to arbitrary winsorization, though
   a data-owner review could improve semantics.

4. **No product/category/brand/traffic/inventory metadata exists.** The model
   can interpolate among the same 30 products, but generalization to unseen
   products or causal campaign interpretation remains limited.

5. **The app/web decomposition remains diagnostically interesting.** It was
   correctly rejected for this submission, but could become useful with more
   channel-specific covariates, a longer recent regime or a loss explicitly
   calibrated to both total and composition quality.

## Dashboard addition

The new `Data story` tab provides a shorter presentation version of this audit:

- six high-impact data facts;
- the C0-C5 decision trail rendered from `outputs/results.json`;
- finding-to-response mappings;
- rejected shortcuts and experiments;
- known limitations;
- links to the model-specific and evaluation-methodology tabs.
