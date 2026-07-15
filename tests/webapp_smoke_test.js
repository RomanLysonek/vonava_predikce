"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const staticDir = path.join(root, "webapp", "static");

function checkSyntax() {
  for (const name of ["common.js", "app.js", "model.js", "evaluation.js", "dataset.js"]) {
    const result = spawnSync(process.execPath, ["--check", path.join(staticDir, name)], {
      encoding: "utf8",
    });
    assert.strictEqual(result.status, 0, `${name} syntax check failed:\n${result.stderr}`);
  }
}

function checkStrategyHelpers() {
  const context = { window: {}, console };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);

  const directOnly = {
    forecasts_by_strategy: { direct: {} },
    selection: { canonical_strategy: "direct" },
  };
  assert.strictEqual(context.availableStrategies(directOnly).join(","), "direct");
  assert.strictEqual(context.canonicalStrategy(directOnly), "direct");

  const recursiveOnly = {
    forecasts_by_strategy: { recursive: {} },
    selection: { canonical_strategy: "recursive" },
  };
  assert.strictEqual(context.canonicalStrategy(recursiveOnly), "recursive");

  const both = {
    forecasts_by_strategy: { direct: {}, recursive: {} },
    selection: { canonical_strategy: "recursive" },
    benchmark_summary_all: [
      { model: "NeuralNet", strategy: "direct", evaluation_regime: "conditional", comparison_population: "common", aggregation: "global", MAE: 1 },
      { model: "NeuralNet", strategy: "recursive", evaluation_regime: "conditional", comparison_population: "common", aggregation: "global", MAE: 2 },
    ],
  };
  assert.strictEqual(context.availableStrategies(both).length, 2);
  const rows = context.summaryRows(both, { strategy: "recursive", regime: "conditional" });
  assert.strictEqual(rows.length, 1);
  assert.strictEqual(rows[0].MAE, 2);

  const withDirectOnlyRidge = {
    ...both,
    models: [{ key: "DynamicRidge", slug: "dynamicridge", strategies: ["direct"] }],
    forecasts_by_strategy: {
      direct: { DynamicRidge: {} },
      recursive: { NeuralNet: {} },
    },
  };
  assert.strictEqual(context.availableStrategies(withDirectOnlyRidge, "DynamicRidge").join(","), "direct");
}

function checkProductExplorerControls() {
  const instances = [];
  class ChartStub {
    constructor(element, config) {
      this.element = element;
      this.data = config.data;
      this.options = config.options;
      instances.push(this);
    }
    destroy() {}
  }

  let legendRefreshes = 0;
  const legendButton = {
    dataset: { modelKey: "NeuralNet" },
    addEventListener(type, handler) {
      if (type === "click") this.click = handler;
    },
  };
  const legendContainer = {
    innerHTML: "",
    querySelectorAll: () => [legendButton],
  };
  const elements = {
    "chart-product": {},
    "product-history-toggle": { checked: true },
    "product-model-legend": legendContainer,
  };
  const context = {
    window: {},
    console,
    Chart: ChartStub,
    document: { getElementById: (id) => elements[id] || {} },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
  const appSource = fs.readFileSync(path.join(staticDir, "app.js"), "utf8");
  vm.runInContext(appSource.slice(0, appSource.lastIndexOf("main();")), context);

  const data = {
    models: [{ key: "NeuralNet", label: "NeuralNet", color: "#111111" }],
    history: { "1": { dates: ["2026-01-01", "2026-01-02"], quantity: [1, 2] } },
    forecasts_by_strategy: {
      direct: {
        NeuralNet: { "1": { dates: ["2026-01-03", "2026-01-04"], quantity: [3, 4] } },
      },
    },
  };

  context.setAllProductModels(data, true);
  context.setProductHistoryVisible(true);
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances.length, 1);
  assert.strictEqual(instances[0].data.labels.join(","), "2026-01-01,2026-01-02,2026-01-03,2026-01-04");
  assert.strictEqual(instances[0].options.plugins.legend.display, false);

  context.setProductHistoryVisible(false);
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances[1].data.labels.join(","), "2026-01-03,2026-01-04");
  assert.strictEqual(instances[1].data.datasets[0].data.join(","), "3,4");

  context.renderProductModelLegend(data, "direct", () => { legendRefreshes += 1; });
  assert.ok(legendContainer.innerHTML.includes("NeuralNet"));
  assert.ok(!legendContainer.innerHTML.includes("History"));
  legendButton.click();
  assert.strictEqual(legendRefreshes, 1);
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances[2].data.datasets[0].hidden, true);

  context.setAllProductModels(data, true);
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances[3].data.datasets[0].hidden, false);
}


function checkSingleModelProductExplorer() {
  const instances = [];
  class ChartStub {
    constructor(element, config) {
      this.element = element;
      this.data = config.data;
      this.options = config.options;
      instances.push(this);
    }
    destroy() {}
  }

  const context = {
    window: { location: { search: "", pathname: "/model/neuralnet" } },
    console,
    Chart: ChartStub,
    document: {
      documentElement: { style: { setProperty() {} } },
      getElementById: () => ({}),
    },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
  const modelSource = fs.readFileSync(path.join(staticDir, "model.js"), "utf8");
  vm.runInContext(modelSource.slice(0, modelSource.lastIndexOf("main();")), context);

  const data = {
    history: { "1": { dates: ["2026-01-01", "2026-01-02"], quantity: [1, 2] } },
    forecasts_by_strategy: {
      direct: {
        NeuralNet: { "1": { dates: ["2026-01-03", "2026-01-04"], quantity: [3, 4] } },
      },
    },
  };
  const model = { key: "NeuralNet", label: "NeuralNet", color: "#111111" };

  context.setModelProductHistoryVisible(true);
  context.renderProductChart(data, model, "1", "direct");
  assert.strictEqual(instances[0].data.labels.join(","), "2026-01-01,2026-01-02,2026-01-03,2026-01-04");
  assert.strictEqual(instances[0].data.datasets.length, 2);

  context.setModelProductHistoryVisible(false);
  context.renderProductChart(data, model, "1", "direct");
  assert.strictEqual(instances[1].data.labels.join(","), "2026-01-03,2026-01-04");
  assert.strictEqual(instances[1].data.datasets.length, 1);
  assert.strictEqual(instances[1].data.datasets[0].data.join(","), "3,4");
  assert.strictEqual(instances[1].options.scales.x.ticks.maxTicksLimit, 7);

  assert.strictEqual(instances[1].options.plugins.legend.display, false);
}


function checkModelMethodDescriptions() {
  const elements = {
    "nn-selection-panel": { hidden: true },
    "nn-selection-grid": { innerHTML: "" },
  };
  const context = {
    window: { location: { search: "", pathname: "/model/neuralnet" } },
    console,
    document: {
      documentElement: { style: { setProperty() {} } },
      getElementById: (id) => elements[id] || null,
    },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
  const modelSource = fs.readFileSync(path.join(staticDir, "model.js"), "utf8");
  vm.runInContext(modelSource.slice(0, modelSource.lastIndexOf("main();")), context);

  const data = {
    config: {
      seeds: [42, 123, 777],
      final_epochs: 60,
      nn_batch_size: 512,
      nn_loss: "mse",
      nn_target_mode: "residual",
      enable_trend_features: false,
      c2_feature_groups: ["price", "campaign", "lifecycle", "market", "event"],
      training_window_days: null,
      recency_half_life_days: null,
    },
    ensemble: {
      strategies: {
        direct: { weights: { NeuralNet: 0.36, XGBoost: 0.25, LightGBM: 0.39 } },
      },
    },
    final_audit_test_aligned_scores: [
      { model: "NeuralNet", test_aligned_score: 0.278328 },
      { model: "Ensemble", test_aligned_score: 0.279737 },
    ],
    ablation_showcase: [
      { tier: "C3/C4", stage: "channel_aux", candidate: "channel_control", model: "NeuralNet", test_aligned_WAPE: 0.262069 },
      { tier: "C3/C4", stage: "channel_aux", candidate: "channel_history_aux_0p2", model: "NeuralNet", test_aligned_WAPE: 0.274209 },
      { tier: "C3/C4", stage: "channel_aux", candidate: "channel_history_only", model: "NeuralNet", test_aligned_WAPE: 0.285132 },
    ],
  };
  for (const key of [
    "NeuralNet", "Ensemble", "XGBoost", "LightGBM",
    "DynamicRidge", "SeasonalNaive", "MovingAvg28",
  ]) {
    const method = context.modelMethodology(data, { key, blurb: key }, "direct");
    const expectedSections = key === "NeuralNet" ? 7 : 5;
    assert.strictEqual(method.items.length, expectedSections, `${key} should have the expected explanation sections`);
    assert.ok(method.items.every((item) => item[1].length > 80), `${key} descriptions should be substantive`);
  }
  const movingAverage = context.modelMethodology(data, { key: "MovingAvg28" }, "direct");
  assert.ok(movingAverage.items.some((item) => item[1].includes("latest 28 calendar rows")));
  assert.ok(movingAverage.items.some((item) => item[1].includes("stays perfectly flat")));
  assert.ok(movingAverage.items.some((item) => item[0] === "Evaluation contract"));

  context.renderNeuralNetSelection(data, { key: "NeuralNet" }, "direct");
  assert.strictEqual(elements["nn-selection-panel"].hidden, false);
  assert.ok(elements["nn-selection-grid"].innerHTML.includes("App-vs-web split experiment"));
  assert.ok(elements["nn-selection-grid"].innerHTML.includes("26.21%"));
  assert.ok(elements["nn-selection-grid"].innerHTML.includes("27.42%"));
  assert.ok(elements["nn-selection-grid"].innerHTML.includes("Rejected"));
  assert.ok(elements["nn-selection-grid"].innerHTML.includes("Why this model is submitted"));

  context.renderNeuralNetSelection(data, { key: "XGBoost" }, "direct");
  assert.strictEqual(elements["nn-selection-panel"].hidden, true);
}

function checkEvaluationMethodology() {
  const elements = {
    "evaluation-current-note": { innerHTML: "" },
    "evaluation-stage-grid": { innerHTML: "" },
    "strategy-method-list": { innerHTML: "" },
    "selection-objective-list": { innerHTML: "" },
    "site-nav": { innerHTML: "" },
    "promo-strategy": { textContent: "" },
    "promo-model-count": { textContent: "" },
    "footer-method-text": { textContent: "" },
  };
  const context = {
    window: { STATIC_DASHBOARD: true },
    console,
    document: {
      getElementById: (id) => elements[id] || null,
      querySelectorAll: () => [],
    },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
  const evaluationSource = fs.readFileSync(path.join(staticDir, "evaluation.js"), "utf8");
  vm.runInContext(evaluationSource.slice(0, evaluationSource.lastIndexOf("main();")), context);

  const data = {
    config: {
      horizon: 7,
      num_products: 30,
      n_dev_origins: 12,
      n_cv_folds: 4,
      final_audit_origins: ["2024-01-17", "2024-05-15", "2024-11-14"],
      selection_protocol: "test-aligned",
      validation_stratum_weights: {
        winter_test_like: 0.6,
        regular: 0.25,
        holiday_event: 0.15,
      },
    },
    models: [
      { key: "NeuralNet", slug: "neuralnet", label: "NeuralNet", color: "#111", strategies: ["direct", "recursive"] },
      { key: "DynamicRidge", slug: "dynamicridge", label: "Dynamic Ridge", color: "#222", strategies: ["direct"] },
    ],
    forecasts_by_strategy: { direct: {} },
    selection: { canonical_model: "NeuralNet", canonical_strategy: "direct" },
    submission: Array.from({ length: 210 }, (_, index) => ({
      ProductId: Math.floor(index / 7) + 1,
      DateKey: `2026-01-${String(12 + (index % 7)).padStart(2, "0")}`,
      Quantity: 1,
    })),
    dev_summary_all: [{
      model: "NeuralNet", strategy: "direct", evaluation_regime: "conditional",
      comparison_population: "common", aggregation: "global", n_scored: 2390, n_expected: 2393,
    }],
    benchmark_summary_all: [{
      model: "NeuralNet", strategy: "direct", evaluation_regime: "conditional",
      comparison_population: "common", aggregation: "global", n_scored: 840, n_expected: 840,
    }],
    final_audit_summary: [{
      model: "NeuralNet", strategy: "direct", evaluation_regime: "conditional",
      comparison_population: "common", aggregation: "global", n_scored: 630, n_expected: 630,
    }],
    test_aligned_scores: [{ model: "NeuralNet", strategy: "direct", test_aligned_score: 0.278 }],
  };

  context.renderCurrentEvaluation(data);
  context.renderEvaluationStages(data);
  context.renderStrategyMethods(data);
  context.renderSelectionObjective(data);
  context.renderNav(data, "evaluation");

  assert.ok(elements["evaluation-current-note"].innerHTML.includes("walk-forward validation is the outer evaluation loop"));
  assert.ok(elements["evaluation-stage-grid"].innerHTML.includes("12 rolling origins"));
  assert.ok(elements["evaluation-stage-grid"].innerHTML.includes("210 submitted rows"));
  assert.ok(elements["strategy-method-list"].innerHTML.includes("Direct multi-horizon"));
  assert.ok(elements["strategy-method-list"].innerHTML.includes("Recursive one-step"));
  assert.ok(elements["selection-objective-list"].innerHTML.includes("60% weight"));
  assert.ok(elements["site-nav"].innerHTML.includes("evaluation.html"));
  assert.ok(elements["site-nav"].innerHTML.includes("nav-pill active"));
}




function checkDatasetStory() {
  const elements = {
    "dataset-current-note": { innerHTML: "" },
    "dataset-decision-trail": { innerHTML: "" },
    "dataset-response-list": { innerHTML: "" },
    "site-nav": { innerHTML: "" },
    "promo-strategy": { textContent: "" },
    "promo-model-count": { textContent: "" },
    "footer-method-text": { textContent: "" },
  };
  const overviewLink = { href: "" };
  const context = {
    window: { STATIC_DASHBOARD: true },
    console,
    document: {
      getElementById: (id) => elements[id] || null,
      querySelectorAll: () => [],
      querySelector: (selector) => selector === "[data-overview-link]" ? overviewLink : null,
    },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
  const datasetSource = fs.readFileSync(path.join(staticDir, "dataset.js"), "utf8");
  vm.runInContext(datasetSource.slice(0, datasetSource.lastIndexOf("main();")), context);

  const data = {
    config: {
      horizon: 7,
      num_products: 30,
      training_window_days: null,
      recency_half_life_days: null,
      baseline_variant: "weighted_4321",
      enable_trend_features: false,
      c2_feature_groups: ["price", "campaign", "lifecycle", "market", "event"],
      nn_loss: "mse",
      nn_target_mode: "residual",
      xgboost_target_mode: "residual",
      lightgbm_target_mode: "log1p",
    },
    models: [
      { key: "NeuralNet", slug: "neuralnet", label: "NeuralNet", color: "#111", strategies: ["direct", "recursive"] },
    ],
    forecasts_by_strategy: { direct: {} },
    selection: { canonical_model: "NeuralNet", canonical_strategy: "direct" },
    ensemble: {
      strategies: {
        direct: { weights: { NeuralNet: 0.36, XGBoost: 0.25, LightGBM: 0.39 } },
      },
    },
    final_audit_test_aligned_scores: [
      { model: "NeuralNet", test_aligned_score: 0.278328 },
      { model: "Ensemble", test_aligned_score: 0.279737 },
    ],
  };

  context.renderDatasetCurrentDecision(data);
  context.renderDatasetDecisionTrail(data);
  context.renderDatasetResponses(data);
  context.renderNav(data, "dataset");
  context.wireDatasetOverviewLink();

  assert.ok(elements["dataset-current-note"].innerHTML.includes("recency decay were screened rather than assumed"));
  assert.ok(elements["dataset-decision-trail"].innerHTML.includes("C1 · Nonstationarity"));
  assert.ok(elements["dataset-decision-trail"].innerHTML.includes("LightGBM 39%"));
  assert.ok(elements["dataset-response-list"].innerHTML.includes("Staggered launches and isolated gaps"));
  assert.ok(elements["dataset-response-list"].innerHTML.includes("channel-aware variants worsened total-demand accuracy"));
  assert.ok(elements["site-nav"].innerHTML.includes("dataset.html"));
  assert.ok(elements["site-nav"].innerHTML.includes("nav-pill active"));
  assert.strictEqual(overviewLink.href, "index.html");
}

function checkSubmissionGridMarkup() {
  const tableWrap = { innerHTML: "" };
  const context = {
    window: {},
    console,
    document: { getElementById: (id) => (id === "submission-table-wrap" ? tableWrap : {}) },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
  const appSource = fs.readFileSync(path.join(staticDir, "app.js"), "utf8");
  vm.runInContext(appSource.slice(0, appSource.lastIndexOf("main();")), context);

  context.renderSubmissionTable({
    submission: [
      { ProductId: 1, DateKey: "2026-01-12", Quantity: 10 },
      { ProductId: 1, DateKey: "2026-01-13", Quantity: 11 },
      { ProductId: 2, DateKey: "2026-01-12", Quantity: 20 },
      { ProductId: 2, DateKey: "2026-01-13", Quantity: 21 },
    ],
  });
  assert.ok(tableWrap.innerHTML.includes('class="data-table submission-table"'));
  assert.strictEqual((tableWrap.innerHTML.match(/submission-date-column/g) || []).length, 2);
  assert.ok(tableWrap.innerHTML.includes("<th>2026-01-12</th>"));
  assert.ok(tableWrap.innerHTML.includes("<td>10</td>"));
}

checkSyntax();
checkStrategyHelpers();
checkProductExplorerControls();
checkSingleModelProductExplorer();
checkModelMethodDescriptions();
checkEvaluationMethodology();
checkDatasetStory();
checkSubmissionGridMarkup();
console.log("8 JavaScript smoke checks passed");
