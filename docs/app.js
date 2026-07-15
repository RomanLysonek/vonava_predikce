const MODEL_ORDER = [
  "NeuralNet",
  "Ensemble",
  "XGBoost",
  "LightGBM",
  "DynamicRidge",
  "SeasonalNaive",
  "MovingAvg28",
];

function modelRank(model) {
  const idx = MODEL_ORDER.indexOf(model);
  return idx === -1 ? MODEL_ORDER.length : idx;
}

function summaryMap(data, strategy, regime) {
  const map = {};
  summaryRows(data, { strategy, regime }).forEach((row) => {
    map[row.model] = row;
  });
  return map;
}

function skillAgainstNaive(summary, modelName) {
  const modelMae = summary[modelName]?.MAE;
  const naiveMae = summary.SeasonalNaive?.MAE;
  if (!Number.isFinite(Number(modelMae)) || !Number.isFinite(Number(naiveMae)) || Number(naiveMae) === 0) {
    return null;
  }
  return 1 - Number(modelMae) / Number(naiveMae);
}

function renderKpis(data, strategy, regime) {
  const rows = summaryRows(data, { strategy, regime });
  if (!rows.length) return;

  const summary = Object.fromEntries(rows.map((row) => [row.model, row]));
  const bestMae = rows.reduce((a, b) => (Number(a.MAE) <= Number(b.MAE) ? a : b));
  const bestWape = rows.reduce((a, b) => (Number(a.WAPE) <= Number(b.WAPE) ? a : b));
  const canonical = canonicalModel(data);
  const canonicalSkill = skillAgainstNaive(summary, canonical);

  const cards = [
    {
      label: "Best MAE",
      value: fmt(bestMae.MAE),
      sub: `${bestMae.model} · ${strategyLabel(strategy)}`,
      color: modelByKey(data, bestMae.model)?.color,
    },
    {
      label: "Best WAPE",
      value: ratePct(bestWape.WAPE),
      sub: `${bestWape.model} · conditional/common`,
      color: modelByKey(data, bestWape.model)?.color,
    },
    {
      label: `${canonical} skill vs. Seasonal-Naive`,
      value: canonicalSkill === null ? "—" : pct(canonicalSkill),
      sub: "MAE improvement on selected view",
      color: modelByKey(data, canonical)?.color,
    },
    {
      label: "Models compared",
      value: String(rows.length),
      sub: `${data.config?.n_cv_folds || "—"} benchmark folds × ${data.config?.horizon || 7} days`,
    },
  ];

  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = cards.map((card) => `
    <div class="kpi-card">
      <p class="kpi-label">${card.label}</p>
      <p class="kpi-value" style="${card.color ? `color:${card.color}` : ""}">${card.value}</p>
      <p class="kpi-sub">${card.sub}</p>
    </div>
  `).join("");
}

function renderColumns(data, strategy, regime) {
  const grid = document.getElementById("columns-grid");
  const summary = summaryMap(data, strategy, regime);
  const kindLabel = { primary: "Submission", ensemble: "OOF Ensemble", baseline: "Baseline", naive: "Naive" };

  grid.innerHTML = (data.models || []).map((model) => {
    const stats = summary[model.key] || {};
    const skill = skillAgainstNaive(summary, model.key);
    return `
      <a class="model-column" style="--mc:${model.color}" href="${modelHref(model.slug)}">
        <div class="model-column-header">
          <span class="model-badge">${kindLabel[model.kind] || model.kind}</span>
          <h3>${model.label}</h3>
          <span class="source">${model.short}</span>
        </div>
        <div class="model-stats">
          <div class="model-stat-row"><span>MAE</span><span>${fmt(stats.MAE)}</span></div>
          <div class="model-stat-row"><span>WAPE</span><span>${ratePct(stats.WAPE)}</span></div>
          <div class="model-stat-row"><span>Bias</span><span style="color:${Number(stats.Bias) >= 0 ? "var(--bad)" : "var(--good)"}">${fmt(stats.Bias)}</span></div>
          <div class="model-stat-row"><span>vs. Naive</span><span>${skill === null ? "—" : pct(skill)}</span></div>
        </div>
        <span class="model-column-cta">View details →</span>
      </a>
    `;
  }).join("");
}

let comparisonChart = null;
function renderComparisonChart(data, strategy, regime) {
  const rows = summaryRows(data, { strategy, regime }).sort(
    (a, b) => modelRank(a.model) - modelRank(b.model),
  );
  const models = rows.map((row) => row.model);
  const colors = models.map((model) => modelByKey(data, model)?.color || "#0a0a0a");

  if (comparisonChart) comparisonChart.destroy();
  comparisonChart = new Chart(document.getElementById("chart-comparison"), {
    type: "bar",
    data: {
      labels: models,
      datasets: [
        { label: "MAE", data: rows.map((row) => row.MAE), backgroundColor: colors, borderRadius: 0 },
        {
          label: "RMSE",
          data: rows.map((row) => row.RMSE),
          backgroundColor: colors.map((color) => `${color}66`),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: CHART_GRID }, beginAtZero: true },
      },
    },
  });
}

function renderCvTable(data, strategy, regime) {
  const tbody = document.querySelector("#cv-table tbody");
  const rows = cvRows(data, { strategy, regime })
    .slice()
    .sort((a, b) => a.fold - b.fold || modelRank(a.model) - modelRank(b.model));

  tbody.innerHTML = rows.map((row) => {
    const color = modelByKey(data, row.model)?.color || "#0a0a0a";
    return `
      <tr>
        <td>${row.fold}</td>
        <td class="model-cell" style="color:${color}">${row.model}</td>
        <td>${fmt(row.MAE)}</td>
        <td>${fmt(row.RMSE)}</td>
        <td style="color:${Number(row.Bias) >= 0 ? "var(--bad)" : "var(--good)"}">${fmt(row.Bias)}</td>
        <td>${pct(row.BiasRatio)}</td>
      </tr>
    `;
  }).join("");
}

let productChart = null;
function renderProductChart(data, productId, strategy) {
  const hist = data.history[productId];
  const strategyForecasts = forecastsFor(data, strategy);
  const labels = [...hist.dates];
  const bridge = hist.quantity[hist.quantity.length - 1];
  let forecastDates = null;

  const datasets = [{
    label: "History",
    data: [...hist.quantity],
    borderColor: "#0a0a0a",
    backgroundColor: "transparent",
    tension: 0.25,
    pointRadius: 0,
    borderWidth: 2,
  }];

  (data.models || []).forEach((model) => {
    const forecast = strategyForecasts[model.key]?.[productId];
    if (!forecast) return;
    if (!forecastDates) forecastDates = forecast.dates;
    datasets.push({
      label: model.label,
      data: [...hist.dates.slice(0, -1).map(() => null), bridge, ...forecast.quantity],
      borderColor: model.color,
      backgroundColor: "transparent",
      borderDash: [6, 4],
      tension: 0.25,
      pointRadius: 3,
      borderWidth: 2,
    });
  });

  const allLabels = forecastDates ? [...labels, ...forecastDates] : labels;
  datasets[0].data = [...datasets[0].data, ...(forecastDates || []).map(() => null)];

  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product"), {
    type: "line",
    data: { labels: allLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
        y: { grid: { color: CHART_GRID }, beginAtZero: true },
      },
    },
  });
}

function populateProductSelector(data) {
  const select = document.getElementById("product-select");
  const ids = Object.keys(data.history || {}).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = ids.map((id) => `<option value="${id}">Product ${id}</option>`).join("");
  return ids[0];
}

function renderSubmissionTable(data) {
  const dates = [...new Set((data.submission || []).map((row) => row.DateKey))].sort();
  const productIds = [...new Set((data.submission || []).map((row) => row.ProductId))].sort((a, b) => a - b);
  const lookup = new Map((data.submission || []).map((row) => [`${row.ProductId}_${row.DateKey}`, row.Quantity]));

  let html = '<table class="data-table"><thead><tr><th>Product</th>';
  dates.forEach((date) => { html += `<th>${date}</th>`; });
  html += "</tr></thead><tbody>";
  productIds.forEach((productId) => {
    html += `<tr><td>#${productId}</td>`;
    dates.forEach((date) => { html += `<td>${lookup.get(`${productId}_${date}`) ?? "—"}</td>`; });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("submission-table-wrap").innerHTML = html;
}

function renderStrategyComparison(data, metric) {
  const panel = document.getElementById("strategy-comparison-panel");
  const tbody = document.querySelector("#strategy-comparison-table tbody");
  const rows = (data.strategy_comparison || [])
    .filter((row) => row.metric === metric)
    .sort((a, b) => modelRank(a.model) - modelRank(b.model));

  panel.hidden = rows.length === 0;
  const metricValue = (value, isDelta = false) => {
    if (metric === "WAPE") return isDelta ? pct(value, 2) : ratePct(value, 2);
    if (metric === "BiasRatio") return isDelta ? pct(value, 2) : pct(value, 2);
    return fmt(value, 2);
  };
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${metricValue(row.direct_value)}</td>
      <td>${metricValue(row.recursive_value)}</td>
      <td>${metricValue(row.absolute_delta, true)}</td>
      <td><span class="winner-badge">${row.winner}</span></td>
      <td>${row.paired_n}</td>
    </tr>
  `).join("");
}

let horizonChart = null;
function renderHorizonChart(data, model, metric, regime) {
  const rows = (data.strategy_by_horizon || []).filter((row) =>
    row.model === model
    && row.evaluation_regime === regime
    && row.comparison_population === "common"
    && row.aggregation === "global",
  );
  const strategies = [...new Set(rows.map((row) => row.strategy))];
  const horizons = [...new Set(rows.map((row) => Number(row.horizon)))].sort((a, b) => a - b);
  const modelColor = modelByKey(data, model)?.color || "#0a0a0a";

  const datasets = strategies.map((strategy) => {
    const byHorizon = new Map(
      rows.filter((row) => row.strategy === strategy).map((row) => [Number(row.horizon), row[metric]]),
    );
    return {
      label: strategyLabel(strategy),
      data: horizons.map((horizon) => byHorizon.get(horizon) ?? null),
      borderColor: modelColor,
      backgroundColor: "transparent",
      borderDash: strategy === "recursive" ? [7, 4] : [],
      pointRadius: 4,
      tension: 0.2,
    };
  });

  if (horizonChart) horizonChart.destroy();
  horizonChart = new Chart(document.getElementById("chart-horizon"), {
    type: "line",
    data: { labels: horizons.map((horizon) => `H${horizon}`), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: CHART_GRID }, beginAtZero: true },
      },
    },
  });
}


function emptyTable(tbody, message, colspan) {
  tbody.innerHTML = `<tr><td colspan="${colspan}" class="empty-state">${message}</td></tr>`;
}

function renderEnsemble(data, strategy) {
  const panel = document.getElementById("ensemble-panel");
  const status = document.getElementById("ensemble-status");
  const weightsBody = document.querySelector("#ensemble-weights-table tbody");
  const comparisonBody = document.querySelector("#ensemble-comparison-table tbody");
  const details = data.ensemble?.strategies?.[strategy];
  if (!details) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  status.textContent = details.accepted ? "Accepted" : "Diagnostic";
  status.classList.toggle("accepted", Boolean(details.accepted));
  status.classList.toggle("rejected", !details.accepted);
  const weights = Object.entries(details.weights || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!weights.length) {
    emptyTable(weightsBody, "No fitted weights.", 2);
  } else {
    weightsBody.innerHTML = weights.map(([model, weight]) => `
      <tr>
        <td class="model-cell" style="color:${modelByKey(data, model)?.color || "#0a0a0a"}">${model}</td>
        <td>${ratePct(weight, 1)}</td>
      </tr>`).join("");
  }
  const benchmark = details.benchmark || {};
  const rows = [
    ["Development aligned WAPE", details.ensemble_test_aligned_wape, details.best_single_test_aligned_wape,
      -Number(details.relative_improvement)],
    ["Development broad WAPE", details.broad_wape, details.best_single_broad_wape,
      details.best_single_broad_wape ? (details.broad_wape - details.best_single_broad_wape) / details.best_single_broad_wape : null],
    ["Recent aligned WAPE", benchmark.ensemble_test_aligned_wape, benchmark.best_single_test_aligned_wape,
      benchmark.relative_test_aligned_change],
    ["Recent broad WAPE", benchmark.ensemble_broad_wape, benchmark.best_single_broad_wape,
      benchmark.relative_broad_change],
  ];
  comparisonBody.innerHTML = rows.map(([label, ensemble, single, change]) => `
    <tr>
      <td>${label}</td>
      <td>${ratePct(ensemble, 2)}</td>
      <td>${ratePct(single, 2)}</td>
      <td class="${Number(change) <= 0 ? "good-text" : "bad-text"}">${pct(change, 2)}</td>
    </tr>`).join("");
}

function populateDiagnosticModelSelector(data, select) {
  const models = (data.models || []).filter((model) => (
    (data.per_product_summary || []).some((row) => row.model === model.key)
  ));
  select.innerHTML = models.map((model) => `<option value="${model.key}">${model.label}</option>`).join("");
  const preferred = models.some((model) => model.key === canonicalModel(data))
    ? canonicalModel(data)
    : (models[0]?.key || "");
  select.value = preferred;
  return preferred;
}

function renderPerProductDiagnostics(data, strategy, model, split) {
  const tbody = document.querySelector("#per-product-table tbody");
  const rows = (data.per_product_summary || [])
    .filter((row) => row.strategy === strategy && row.model === model && row.origin_type === split)
    .sort((a, b) => Number(b.WAPE) - Number(a.WAPE));
  if (!rows.length) {
    emptyTable(tbody, "No per-product diagnostics for this selection.", 5);
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>#${row.ProductId}</td>
      <td>${ratePct(row.WAPE, 1)}</td>
      <td>${fmt(row.MAE, 1)}</td>
      <td class="${Math.abs(Number(row.BiasRatio)) <= 0.05 ? "good-text" : "bad-text"}">${pct(row.BiasRatio, 1)}</td>
      <td>${fmt(row.actual_total, 0)}</td>
    </tr>`).join("");
}

function renderRegimeDiagnostics(data, strategy, regime) {
  const tbody = document.querySelector("#regime-table tbody");
  const rows = (data.validation_strata_summary || [])
    .filter((row) => (
      row.strategy === strategy
      && row.origin_type === "development"
      && row.evaluation_regime === regime
      && row.comparison_population === "common"
      && row.aggregation === "global"
    ))
    .sort((a, b) => String(a.validation_stratum).localeCompare(String(b.validation_stratum)) || modelRank(a.model) - modelRank(b.model));
  if (!rows.length) {
    emptyTable(tbody, "No validation-stratum diagnostics.", 5);
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${String(row.validation_stratum).replaceAll("_", " ")}</td>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${ratePct(row.WAPE, 1)}</td>
      <td>${pct(row.BiasRatio, 1)}</td>
      <td>${row.n_scored ?? row.n ?? "—"}</td>
    </tr>`).join("");
}

function renderTopDecile(data, strategy) {
  const tbody = document.querySelector("#top-decile-table tbody");
  const rows = (data.top_decile_summary || [])
    .filter((row) => row.strategy === strategy && row.origin_type === "recent_benchmark")
    .sort((a, b) => Number(a.WAPE) - Number(b.WAPE));
  if (!rows.length) {
    emptyTable(tbody, "No top-decile diagnostics.", 5);
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${ratePct(row.WAPE, 1)}</td>
      <td>${fmt(row.MAE, 1)}</td>
      <td>${pct(row.BiasRatio, 1)}</td>
      <td>${fmt(row.actual_threshold, 1)}</td>
    </tr>`).join("");
}

function renderTopErrors(data, strategy) {
  const tbody = document.querySelector("#top-error-table tbody");
  const rows = (data.top_error_rows || [])
    .filter((row) => row.strategy === strategy && row.origin_type === "recent_benchmark")
    .sort((a, b) => Number(b.absolute_error) - Number(a.absolute_error))
    .slice(0, 30);
  if (!rows.length) {
    emptyTable(tbody, "No recent row-level errors.", 6);
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${String(row.DateKey || "—").slice(0, 10)}</td>
      <td>#${row.ProductId}</td>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${fmt(row.actual, 1)}</td>
      <td>${fmt(row.prediction, 1)}</td>
      <td>${fmt(row.absolute_error, 1)}</td>
    </tr>`).join("");
}

function renderChannelShare(data, strategy) {
  const target = document.getElementById("channel-share-content");
  const rows = (data.channel_share_summary || []).filter((row) => (
    row.strategy === strategy && Number(row.n_scored ?? row.app_share_n ?? 0) > 0
  ));
  if (!rows.length) {
    target.className = "empty-state";
    target.textContent = "The screened channel-aware head was not selected; total demand remains canonical.";
    return;
  }
  target.className = "table-wrap";
  target.innerHTML = `<table class="data-table"><thead><tr><th>Split</th><th>Share MAE</th><th>Weighted MAE</th><th>n</th></tr></thead><tbody>${rows.map((row) => `
    <tr><td>${row.origin_type || "development"}</td><td>${fmt(row.app_share_MAE, 3)}</td><td>${fmt(row.app_share_weighted_MAE, 3)}</td><td>${row.n_scored ?? row.app_share_n ?? row.n ?? "—"}</td></tr>`).join("")}</tbody></table>`;
}


function renderFinalAudit(data, strategy, regime) {
  const panel = document.getElementById("final-audit-panel");
  const tbody = document.querySelector("#final-audit-table tbody");
  const rows = (data.final_audit_summary || [])
    .filter((row) => (
      row.strategy === strategy
      && row.evaluation_regime === regime
      && row.comparison_population === "common"
      && row.aggregation === "global"
    ))
    .sort((a, b) => Number(a.WAPE) - Number(b.WAPE));
  panel.hidden = rows.length === 0;
  if (!rows.length) return;
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${ratePct(row.WAPE, 2)}</td>
      <td>${fmt(row.MAE, 2)}</td>
      <td>${pct(row.BiasRatio, 2)}</td>
      <td>${ratePct(row.coverage, 1)}</td>
    </tr>`).join("");
}

function renderAblations(data) {
  const tbody = document.querySelector("#ablation-table tbody");
  const all = data.ablation_showcase || [];
  const selected = all.filter((row) => row.selected);
  const remaining = all.filter((row) => !row.selected)
    .sort((a, b) => Number(a.test_aligned_WAPE ?? Infinity) - Number(b.test_aligned_WAPE ?? Infinity));
  const rows = [...selected, ...remaining].slice(0, 36);
  if (!rows.length) {
    emptyTable(tbody, "No persisted ablation artifacts.", 5);
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr class="${row.selected ? "selected-row" : ""}" title="${row.description || ""}">
      <td>${row.tier}</td>
      <td>${row.stage}</td>
      <td>${row.selected ? "★ " : ""}${row.candidate}</td>
      <td>${row.model}</td>
      <td>${ratePct(row.test_aligned_WAPE ?? row.WAPE, 2)}</td>
    </tr>`).join("");
}

async function main() {
  try {
    const data = await loadResults();
    const strategySelect = document.getElementById("strategy-select");
    const regimeSelect = document.getElementById("regime-select");
    const productSelect = document.getElementById("product-select");
    const pairMetricSelect = document.getElementById("pair-metric-select");
    const horizonModelSelect = document.getElementById("horizon-model-select");
    const horizonMetricSelect = document.getElementById("horizon-metric-select");
    const productErrorModelSelect = document.getElementById("product-error-model-select");
    const productErrorSplitSelect = document.getElementById("product-error-split-select");

    pairMetricSelect.value = data.config?.selection_metric || "WAPE";
    horizonMetricSelect.value = data.config?.selection_metric || "WAPE";
    horizonModelSelect.innerHTML = (data.models || [])
      .map((model) => `<option value="${model.key}">${model.label}</option>`)
      .join("");
    horizonModelSelect.value = canonicalModel(data);

    const firstProduct = populateProductSelector(data);
    populateDiagnosticModelSelector(data, productErrorModelSelect);
    configureStrategySelect(data, strategySelect, refresh);
    regimeSelect.value = data.config?.primary_evaluation_regime || "conditional";

    function refresh() {
      const strategy = strategySelect.value || canonicalStrategy(data);
      const regime = regimeSelect.value || "conditional";
      updateStrategyCopy(data, strategy);
      renderKpis(data, strategy, regime);
      renderColumns(data, strategy, regime);
      renderComparisonChart(data, strategy, regime);
      renderCvTable(data, strategy, regime);
      renderProductChart(data, productSelect.value || firstProduct, strategy);
      renderStrategyComparison(data, pairMetricSelect.value);
      renderHorizonChart(data, horizonModelSelect.value, horizonMetricSelect.value, regime);
      renderEnsemble(data, strategy);
      renderPerProductDiagnostics(data, strategy, productErrorModelSelect.value, productErrorSplitSelect.value);
      renderRegimeDiagnostics(data, strategy, regime);
      renderTopDecile(data, strategy);
      renderTopErrors(data, strategy);
      renderChannelShare(data, strategy);
      renderFinalAudit(data, strategy, regime);
      renderAblations(data);
      document.getElementById("product-strategy-note").textContent = strategyLabel(strategy);
    }

    [regimeSelect, productSelect, pairMetricSelect, horizonModelSelect, horizonMetricSelect,
      productErrorModelSelect, productErrorSplitSelect]
      .forEach((select) => select.addEventListener("change", refresh));

    renderNav(data, "");
    renderSubmissionTable(data);
    refresh();
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load results</h2></div>
        <p style="color:var(--bad)">${err.message}</p>
      </div>`;
  }
}

main();
