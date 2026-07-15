function evaluationSummaryRow(data, source, model = canonicalModel(data)) {
  const rows = source === "development"
    ? (data.dev_summary_all || data.dev_summary || [])
    : source === "final_audit"
      ? (data.final_audit_summary || [])
      : (data.benchmark_summary_all || data.benchmark_summary || []);
  const strategy = canonicalStrategy(data);
  return rows.find((row) => (
    row.model === model
    && (!row.strategy || row.strategy === strategy)
    && row.evaluation_regime === "conditional"
    && row.comparison_population === "common"
    && row.aggregation === "global"
  )) || null;
}

function rowCountCopy(row) {
  if (!row) return "Persisted row count unavailable";
  const scored = Number(row.n_scored ?? row.n);
  const expected = Number(row.n_expected);
  if (!Number.isFinite(scored)) return "Persisted row count unavailable";
  return Number.isFinite(expected)
    ? `${scored.toLocaleString()} scored of ${expected.toLocaleString()} expected rows`
    : `${scored.toLocaleString()} scored rows`;
}

function renderCurrentEvaluation(data) {
  const strategy = canonicalStrategy(data);
  const model = canonicalModel(data);
  const current = document.getElementById("evaluation-current-note");
  if (!current) return;
  current.innerHTML = `<strong>Current published decision:</strong> ${model} / ${strategyLabel(strategy)}. The repository supports recursive inference for eligible models, but walk-forward validation is the outer evaluation loop; it does not mean recursive inference, and recursive generation is not the canonical submission in this artifact. <a href="${datasetHref()}">The dataset story explains why this evaluation contract was necessary.</a>`;
}

function renderEvaluationStages(data) {
  const cfg = data.config || {};
  const dev = evaluationSummaryRow(data, "development");
  const benchmark = evaluationSummaryRow(data, "benchmark");
  const audit = evaluationSummaryRow(data, "final_audit");
  const submissionDates = [...new Set((data.submission || []).map((row) => row.DateKey))].sort();
  const testRange = submissionDates.length
    ? `${submissionDates[0]} to ${submissionDates[submissionDates.length - 1]}`
    : "the supplied seven-day test block";
  const testStart = submissionDates[0] || "the first supplied test date";
  const stages = [
    {
      badge: "1 · Development",
      title: `${cfg.n_dev_origins || dev?.n_folds || 12} rolling origins`,
      role: "Decision split",
      body: "Seasonally scattered historical origins used for feature, target, loss, direct-vs-recursive, model and ensemble decisions. Only this split may tune the forecasting system.",
      detail: rowCountCopy(dev),
    },
    {
      badge: "2 · Recent benchmark",
      title: `${cfg.n_cv_folds || benchmark?.n_folds || 4} non-overlapping origins`,
      role: "Confirmation split",
      body: "The latest complete seven-day blocks before the real test period. They check whether the development choice still behaves sensibly under recent demand conditions, but do not change it.",
      detail: rowCountCopy(benchmark),
    },
    {
      badge: "3 · Frozen final audit",
      title: `${(cfg.final_audit_origins || []).length || audit?.n_folds || 3} untouched origins`,
      role: "One-shot audit",
      body: "Disjoint origins kept out of normal experiments. They are executed only after C5 ensemble weights and the canonical decision are frozen, preventing repeated holdout optimization.",
      detail: rowCountCopy(audit),
    },
    {
      badge: "4 · Actual submission",
      title: `${cfg.num_products || 30} products × ${cfg.horizon || 7} days`,
      role: "Unlabelled forecast",
      body: `The selected final model is trained on all eligible history available before ${testStart}, then forecasts ${testRange} and produces the canonical ${Number(cfg.num_products || 30) * Number(cfg.horizon || 7)}-row grid. Test actuals are never used anywhere in evaluation or selection.`,
      detail: `${(data.submission || []).length.toLocaleString()} submitted rows`,
    },
  ];
  document.getElementById("evaluation-stage-grid").innerHTML = stages.map((stage) => `
    <article class="evaluation-stage-card">
      <span class="evaluation-stage-badge">${stage.badge}</span>
      <h2>${stage.title}</h2>
      <strong>${stage.role}</strong>
      <p>${stage.body}</p>
      <span class="evaluation-stage-detail">${stage.detail}</span>
    </article>
  `).join("");
}

function renderStrategyMethods(data) {
  const strategy = canonicalStrategy(data);
  const exported = Object.keys(data.forecasts_by_strategy || {});
  const recursiveModels = (data.models || []).filter((model) => (model.strategies || []).includes("recursive"));
  const directModels = (data.models || []).filter((model) => (model.strategies || []).includes("direct"));
  const rows = [
    [
      "Direct multi-horizon",
      `At each origin, a stacked (ForecastOrigin × Horizon × ProductId) panel contains seven rows per product. All horizons use the same observed information cutoff; no forecast is fed into a later horizon. ${directModels.length} displayed models support this contract, and it is the current canonical strategy.`,
    ],
    [
      "Recursive one-step",
      `A one-day-ahead model is applied seven times. After each step, the generated quantity is appended as synthetic available history and lag features are rebuilt. This exposes compounding error. ${recursiveModels.length} displayed models support this contract; Dynamic Ridge is intentionally excluded.`,
    ],
    [
      "Paired comparison",
      "When both strategies are run, direct and recursive predictions are inner-joined on the same origin, product, target date and horizon before comparison. The strategy winner is selected from development OOF only.",
    ],
    [
      "Current artifact",
      `The published results contain ${exported.map(strategyLabel).join(" and ") || strategyLabel(strategy)} final forecasts, with ${strategyLabel(strategy)} selected. Strategy support in a model tab describes implemented capability, not necessarily an exported forecast in this particular run.`,
    ],
  ];
  document.getElementById("strategy-method-list").innerHTML = rows.map(([title, description]) => `
    <div class="definition-item evaluation-step-item"><strong>${title}</strong><span>${description}</span></div>
  `).join("");
}

function renderSelectionObjective(data) {
  const cfg = data.config || {};
  const weights = cfg.validation_stratum_weights || {};
  const protocol = cfg.selection_protocol === "test-aligned"
    ? "Frozen test-aligned WAPE"
    : "Global conditional/common WAPE";
  const score = (data.test_aligned_scores || []).find((row) => (
    row.model === canonicalModel(data) && row.strategy === canonicalStrategy(data)
  ));
  const rows = [
    ["Primary objective", `${protocol} on development out-of-fold predictions. WAPE is computed separately for each validation-period type, then combined with fixed weights.`],
    ["January/February proxy", `${ratePct(weights.winter_test_like ?? 0.6, 0)} weight. All seven target dates fall in January or February, making this the closest seasonal proxy for the supplied January test week.`],
    ["Regular periods", `${ratePct(weights.regular ?? 0.25, 0)} weight. Development windows outside the winter-proxy and holiday/event stress definitions.`],
    ["Holiday / retail-event stress", `${ratePct(weights.holiday_event ?? 0.15, 0)} weight. Windows containing December or late-November retail-event dates, retained to prevent a January-focused objective from ignoring peak-demand robustness.`],
    ["Selected result", `${canonicalModel(data)} / ${strategyLabel(canonicalStrategy(data))}${score && Number.isFinite(Number(score.test_aligned_score)) ? ` achieved ${ratePct(score.test_aligned_score, 2)} development test-aligned WAPE` : " won the frozen development objective"}. The recent benchmark confirmed the decision; it did not choose it.`],
  ];
  document.getElementById("selection-objective-list").innerHTML = rows.map(([title, description]) => `
    <div class="definition-item evaluation-step-item"><strong>${title}</strong><span>${description}</span></div>
  `).join("");
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "evaluation");
    updateStrategyCopy(data, canonicalStrategy(data));
    renderCurrentEvaluation(data);
    renderEvaluationStages(data);
    renderStrategyMethods(data);
    renderSelectionObjective(data);
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load evaluation metadata</h2></div>
        <p style="color:var(--bad)">${err.message}</p>
      </div>`;
  }
}

main();
