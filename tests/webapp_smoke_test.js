"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const staticDir = path.join(root, "webapp", "static");

function checkSyntax() {
  for (const name of ["common.js", "app.js", "model.js"]) {
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
    isDatasetVisible(index) { return this.data.datasets[index].hidden !== true; }
    setDatasetVisibility(index, visible) { this.data.datasets[index].hidden = !visible; }
    update() {}
  }

  const elements = {
    "chart-product": {},
    "product-history-toggle": { checked: true },
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
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances.length, 1);
  assert.strictEqual(instances[0].data.labels.join(","), "2026-01-01,2026-01-02,2026-01-03,2026-01-04");

  instances[0].options.plugins.legend.onClick({}, { datasetIndex: 0 }, { chart: instances[0] });
  assert.strictEqual(instances.length, 2);
  assert.strictEqual(instances[1].data.labels.join(","), "2026-01-03,2026-01-04");
  assert.strictEqual(instances[1].data.datasets[0].data.join(","), "3,4");
  assert.strictEqual(elements["product-history-toggle"].checked, false);

  context.setAllProductModels(data, false);
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances[2].data.datasets[0].hidden, true);
  context.setAllProductModels(data, true);
  context.renderProductChart(data, "1", "direct");
  assert.strictEqual(instances[3].data.datasets[0].hidden, false);
}

checkSyntax();
checkStrategyHelpers();
checkProductExplorerControls();
console.log("3 JavaScript smoke checks passed");
