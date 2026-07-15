function datasetFeatureGroups(groups) {
  const labels = {
    price: "price and channel-specific effective offer",
    campaign: "campaign semantics and channel asymmetry",
    lifecycle: "lifecycle, calendar-gap and availability state",
    market: "cross-product market context",
    event: "retail-event distance and windows",
  };
  return (groups || []).map((group) => labels[group] || group);
}

function datasetHistoryCopy(cfg) {
  const windowText = cfg.training_window_days == null
    ? "all eligible history"
    : `the latest ${cfg.training_window_days} days`;
  const weightText = cfg.recency_half_life_days == null
    ? "no exponential recency decay"
    : `${cfg.recency_half_life_days}-day recency half-life`;
  return `${windowText}, ${weightText}`;
}

function datasetEnsembleWeights(data) {
  const strategy = canonicalStrategy(data);
  const weights = data.ensemble?.strategies?.[strategy]?.weights || {};
  const entries = Object.entries(weights);
  if (!entries.length) return "No accepted ensemble in this artifact";
  return entries
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([model, weight]) => `${model} ${ratePct(weight, 0)}`)
    .join(" · ");
}

function renderDatasetCurrentDecision(data) {
  const target = "QuantityApp + QuantityWeb";
  const note = document.getElementById("dataset-current-note");
  if (!note) return;
  note.innerHTML = `<strong>Current published solution:</strong> forecast total demand (${target}) with ${canonicalModel(data)} / ${strategyLabel(canonicalStrategy(data))}. The app-share head and recency decay were screened rather than assumed; neither was retained by the frozen development objective.`;
}

function renderDatasetDecisionTrail(data) {
  const cfg = data.config || {};
  const groups = datasetFeatureGroups(cfg.c2_feature_groups);
  const auditRows = data.final_audit_test_aligned_scores || [];
  const auditNn = auditRows.find((row) => row.model === "NeuralNet");
  const auditEnsemble = auditRows.find((row) => row.model === "Ensemble");
  const stages = [
    {
      badge: "C0 · Data contract",
      title: "Repair the panel before modeling",
      body: "Daily reindexing, explicit gap/unavailable/pre-launch states, availability-aware lags, common scoring rows and leakage guards established what one training observation actually means.",
      detail: "Missing rows were never silently turned into sales zeros",
    },
    {
      badge: "C1 · Nonstationarity",
      title: "Screen recency instead of guessing",
      body: `Training windows, exponential half-lives, baseline variants and trend features were compared. The retained configuration uses ${datasetHistoryCopy(cfg)}, the ${cfg.baseline_variant || "weighted_4321"} same-weekday baseline and ${cfg.enable_trend_features ? "trend features" : "no explicit trend group"}.`,
      detail: "Recent decline was addressed experimentally, not by an arbitrary cutoff",
    },
    {
      badge: "C2 · Semantics",
      title: "Translate business anomalies into features",
      body: `The selected feature groups are ${groups.join(", ") || "the configured semantic set"}. They encode effective prices, app-only campaigns, discount-without-campaign cases, lifecycle state, market intensity and retail events.`,
      detail: `${groups.length || 0} semantic groups retained`,
    },
    {
      badge: "C3/C4 · Targets",
      title: "Choose objectives and test channel structure",
      body: `The selected NeuralNet predicts a ${cfg.nn_target_mode || "residual"} target with ${String(cfg.nn_loss || "configured").toUpperCase()} loss. XGBoost uses ${cfg.xgboost_target_mode || cfg.tree_target_mode || "log1p"}; LightGBM uses ${cfg.lightgbm_target_mode || cfg.tree_target_mode || "log1p"}. Channel-history and app-share auxiliary candidates were tested and rejected.`,
      detail: "Total demand stayed canonical because channel-aware candidates worsened it",
    },
    {
      badge: "Strategy · 7 days",
      title: `${strategyLabel(canonicalStrategy(data))} selected`,
      body: "Direct and recursive engines were implemented under the same origin cutoff. Direct won the development selection and avoids feeding uncertain day-one predictions into later horizons for the submitted week.",
      detail: "Walk-forward evaluation remains separate from forecast generation",
    },
    {
      badge: "C5 · Freeze and audit",
      title: "Blend where useful, submit what survived",
      body: `The secondary convex blend is ${datasetEnsembleWeights(data)}. It improved development and recent confirmation, but the untouched aligned audit ${auditNn && auditEnsemble ? `favored NeuralNet ${ratePct(auditNn.test_aligned_score, 2)} vs. ensemble ${ratePct(auditEnsemble.test_aligned_score, 2)}` : "did not replace the canonical NeuralNet"}.`,
      detail: "Weights were frozen before the one-shot audit",
    },
  ];
  document.getElementById("dataset-decision-trail").innerHTML = stages.map((stage) => `
    <article class="dataset-decision-card">
      <span class="dataset-decision-badge">${stage.badge}</span>
      <h3>${stage.title}</h3>
      <p>${stage.body}</p>
      <strong>${stage.detail}</strong>
    </article>
  `).join("");
}

function renderDatasetResponses(data) {
  const cfg = data.config || {};
  const rows = [
    ["Staggered launches and isolated gaps", "Reindex each product to a daily calendar, preserve inserted gaps as unknown, and carry separate first-row and first-available lifecycle clocks. Rolling demand uses only observed-and-available quantities."],
    ["Stock-constrained realized sales", "Exclude unavailable targets from the primary supervised and conditional-scoring population. Keep realized-sales scoring as a diagnostic and do not use future ProductAvailable, which is absent from test."],
    ["Full weekly cycle and annual events", "Use day-of-week/month/year cyclic encodings, target-relative same-weekday lags, nullable annual references and deterministic Black Friday, Christmas, Valentine and Mother’s Day proximity features."],
    ["Changing demand level", `Screen all/730/365-day windows, none/365/180/90-day half-lives and explicit trend features. The selected full run retained ${datasetHistoryCopy(cfg)} because stronger decay did not improve the frozen aligned objective.`],
    ["Price and campaign semantics", "Treat campaign subtype as categorical, keep IsSaleOrPromo separate, derive web/app effective prices, app discount advantage, app-only campaigns and discount-without-campaign indicators. All five C2 semantic groups survived screening."],
    ["Related products and synchronized shocks", "Fit global pooled models across all products, retain ProductId identity through embeddings/categories, and add leakage-safe market demand plus future-known cross-sectional campaign/discount intensity."],
    ["Right-skewed overdispersed quantities", "Model in log or baseline-relative log space rather than assuming a simple Poisson count process. Huber, MSE, Log-Cosh, mixed loss and Tweedie alternatives were screened; the final targets are model-specific."],
    ["Channel migration", "Test leakage-safe app-share history and an auxiliary app-share head while keeping total demand as the submitted target. The channel-aware variants worsened total-demand accuracy, so they remain documented rejected experiments."],
    ["Event-driven extreme quantities", "Retain plausible promotional peaks, add retail-event features, use holiday/event validation strata, and expose top-volume and largest-error diagnostics instead of winsorizing the business signal away."],
    ["Need for honest floors", "Compare against lag-7 seasonal persistence, a flat availability-aware 28-day mean, Dynamic Ridge, XGBoost and LightGBM. Complexity is credited only when it beats simple recent-seasonal behavior on the same rows."],
    ["Seven coupled forecast horizons", "Implement both direct stacked-horizon and genuine recursive one-step engines. Select the contract from development OOF and keep the seven target dates aligned to one shared observed origin."],
    ["Small model-selection sample", "Separate development decisions, recent confirmation and a frozen one-shot audit. Fit ensemble weights only on development OOF and retain NeuralNet as canonical when the final aligned audit marginally favors it."],
  ];
  document.getElementById("dataset-response-list").innerHTML = rows.map(([title, description]) => `
    <div class="definition-item dataset-response-item"><strong>${title}</strong><span>${description}</span></div>
  `).join("");
}

function wireDatasetOverviewLink() {
  const link = document.querySelector?.("[data-overview-link]");
  if (link) link.href = overviewHref();
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "dataset");
    updateStrategyCopy(data, canonicalStrategy(data));
    renderDatasetCurrentDecision(data);
    renderDatasetDecisionTrail(data);
    renderDatasetResponses(data);
    wireDatasetOverviewLink();
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load dataset-story metadata</h2></div>
        <p style="color:var(--bad)">${err.message}</p>
      </div>`;
  }
}

main();
