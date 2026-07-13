const MODEL_ORDER = [
  "NeuralNet",
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
  const kindLabel = { primary: "Submission", baseline: "Baseline", naive: "Naive" };

  grid.innerHTML = (data.models || []).map((model) => {
    const stats = summary[model.key] || {};
    const skill = skillAgainstNaive(summary, model.key);
    return `
      <a class="model-column" style="--mc:${model.color}" href="/model/${model.slug}">
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

async function main() {
  try {
    const data = await loadResults();
    const strategySelect = document.getElementById("strategy-select");
    const regimeSelect = document.getElementById("regime-select");
    const productSelect = document.getElementById("product-select");
    const pairMetricSelect = document.getElementById("pair-metric-select");
    const horizonModelSelect = document.getElementById("horizon-model-select");
    const horizonMetricSelect = document.getElementById("horizon-metric-select");

    pairMetricSelect.value = data.config?.selection_metric || "WAPE";
    horizonMetricSelect.value = data.config?.selection_metric || "WAPE";
    horizonModelSelect.innerHTML = (data.models || [])
      .map((model) => `<option value="${model.key}">${model.label}</option>`)
      .join("");
    horizonModelSelect.value = canonicalModel(data);

    const firstProduct = populateProductSelector(data);
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
      document.getElementById("product-strategy-note").textContent = strategyLabel(strategy);
    }

    [regimeSelect, productSelect, pairMetricSelect, horizonModelSelect, horizonMetricSelect]
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
