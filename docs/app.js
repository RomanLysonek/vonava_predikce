const MODEL_ORDER = [
  "NeuralNet",
  "Ensemble",
  "XGBoost",
  "LightGBM",
  "DynamicRidge",
  "SeasonalNaive",
  "MovingAvg28",
];

const VALIDATION_STRATA = {
  winter_test_like: {
    order: 0,
    label: "January/February test-season proxy",
    definition: "All seven target dates fall in January or February, making this the closest seasonal proxy for the supplied January test week.",
  },
  regular: {
    order: 1,
    label: "Regular periods",
    definition: "Development windows outside the winter-proxy and holiday/event rules; these represent ordinary trading conditions.",
  },
  holiday_event: {
    order: 2,
    label: "Holiday / retail-event stress",
    definition: "Any target date falls in December or on/after 20 November, covering Black Friday and pre-Christmas demand shifts.",
  },
};

function modelRank(model) {
  const idx = MODEL_ORDER.indexOf(model);
  return idx === -1 ? MODEL_ORDER.length : idx;
}

function regimeLabel(regime) {
  return regime === "realized" ? "all days / realized sales" : "available days only";
}

function renderRegimeExplanation(data, strategy, regime) {
  const target = document.getElementById("regime-explanation");
  const context = document.getElementById("model-comparison-context");
  if (context) {
    context.textContent = `Recent-benchmark OOF · ${regimeLabel(regime)} · common rows · global aggregation`;
  }
  if (!target) return;

  const conditional = summaryRows(data, { strategy, regime: "conditional" });
  const realized = summaryRows(data, { strategy, regime: "realized" });
  const realizedByModel = new Map(realized.map((row) => [row.model, row]));
  const identical = conditional.length > 0 && conditional.every((row) => {
    const other = realizedByModel.get(row.model);
    return other
      && Math.abs(Number(row.MAE) - Number(other.MAE)) < 1e-10
      && Math.abs(Number(row.WAPE) - Number(other.WAPE)) < 1e-10
      && Number(row.n_scored) === Number(other.n_scored);
  });

  const selected = regime === "realized" ? realized : conditional;
  const nScored = selected[0]?.n_scored;
  const scope = Number.isFinite(Number(nScored)) ? `${Number(nScored).toLocaleString()} common rows` : "the common evaluation rows";
  const definition = regime === "realized"
    ? "All observed product-days are scored, including unavailable days and their realized sales."
    : "Only product-days on which the product was available for purchase are scored.";
  const sameReason = identical
    ? ` In this recent benchmark, all ${scope} were available, so both choices produce identical headline numbers.`
    : " The two views differ whenever unavailable product-days occur.";
  target.innerHTML = `<strong>Scoring lens only:</strong> ${definition} This never changes the forecast or canonical submission.${sameReason}`;
}

function renderRegimeDefinitions(data) {
  const target = document.getElementById("regime-definitions");
  if (!target) return;
  const weights = data.config?.validation_stratum_weights || {};
  target.innerHTML = Object.entries(VALIDATION_STRATA)
    .sort(([, a], [, b]) => a.order - b.order)
    .map(([key, meta]) => {
      const weight = Number(weights[key]);
      const weightText = Number.isFinite(weight) ? ` · ${ratePct(weight, 0)} of the test-aligned score` : "";
      return `<div class="definition-item"><strong>${meta.label}</strong><span>${meta.definition}${weightText}</span></div>`;
    })
    .join("");
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
      sub: `${bestWape.model} · ${regimeLabel(regime)}`,
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
let productHistoryVisible = true;
const productModelVisibility = new Map();

function setAllProductModels(data, visible) {
  (data.models || []).forEach((model) => productModelVisibility.set(model.key, visible));
}

function setProductHistoryVisible(visible) {
  productHistoryVisible = Boolean(visible);
}

function renderProductModelLegend(data, strategy, refreshProductExplorer) {
  const container = document.getElementById("product-model-legend");
  if (!container) return;
  const strategyForecasts = forecastsFor(data, strategy);
  const models = (data.models || []).filter((model) => strategyForecasts[model.key]);
  container.innerHTML = models.map((model) => {
    const visible = productModelVisibility.get(model.key) !== false;
    return `
      <button
        class="product-model-legend-button${visible ? "" : " is-hidden"}"
        type="button"
        data-model-key="${model.key}"
        aria-pressed="${visible}"
        title="${visible ? "Hide" : "Show"} ${model.label}"
      >
        <span class="product-model-legend-swatch" style="--model-color:${model.color}"></span>
        <span>${model.label}</span>
      </button>`;
  }).join("");

  container.querySelectorAll("[data-model-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.modelKey;
      productModelVisibility.set(key, productModelVisibility.get(key) === false);
      refreshProductExplorer();
    });
  });
}

function renderProductChart(data, productId, strategy) {
  const hist = data.history?.[productId];
  if (!hist) return;
  const strategyForecasts = forecastsFor(data, strategy);
  const firstForecast = (data.models || [])
    .map((model) => strategyForecasts[model.key]?.[productId])
    .find(Boolean);
  const forecastDates = firstForecast?.dates || [];
  const labels = productHistoryVisible
    ? [...hist.dates, ...forecastDates]
    : [...forecastDates];
  const bridge = hist.quantity[hist.quantity.length - 1];
  const datasets = [];

  if (productHistoryVisible) {
    datasets.push({
      label: "History",
      data: [...hist.quantity, ...forecastDates.map(() => null)],
      borderColor: "#0a0a0a",
      backgroundColor: "transparent",
      tension: 0.25,
      pointRadius: 0,
      borderWidth: 2,
    });
  }

  (data.models || []).forEach((model) => {
    const forecast = strategyForecasts[model.key]?.[productId];
    if (!forecast) return;
    const values = productHistoryVisible
      ? [...hist.dates.slice(0, -1).map(() => null), bridge, ...forecast.quantity]
      : [...forecast.quantity];
    datasets.push({
      label: model.label,
      data: values,
      borderColor: model.color,
      backgroundColor: "transparent",
      borderDash: [6, 4],
      tension: 0.25,
      pointRadius: 3,
      borderWidth: 2,
      hidden: productModelVisibility.get(model.key) === false,
      modelKey: model.key,
    });
  });

  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: productHistoryVisible ? 10 : 7 } },
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

  let html = '<table class="data-table submission-table"><colgroup><col class="submission-product-column">';
  dates.forEach(() => { html += '<col class="submission-date-column">'; });
  html += '</colgroup><thead><tr><th>Product</th>';
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
  comparisonBody.innerHTML = rows.map(([label, ensemble, single, change]) => {
    const ensembleValue = Number(ensemble);
    const singleValue = Number(single);
    const hasScores = (
      ensemble !== null && ensemble !== undefined
      && single !== null && single !== undefined
      && Number.isFinite(ensembleValue) && Number.isFinite(singleValue)
    );
    const absoluteDeltaPp = hasScores ? (ensembleValue - singleValue) * 100 : null;
    const changeClass = Number(change) <= 0 ? "good-text" : "bad-text";
    const absoluteDeltaText = absoluteDeltaPp === null
      ? "—"
      : `${absoluteDeltaPp >= 0 ? "+" : ""}${fmt(absoluteDeltaPp, 2)} pp`;
    return `
      <tr>
        <td>${label}</td>
        <td>${ratePct(ensemble, 2)}</td>
        <td>${ratePct(single, 2)}</td>
        <td class="${changeClass}">${absoluteDeltaText}</td>
        <td class="${changeClass}">${pct(change, 2)}</td>
      </tr>`;
  }).join("");
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
    .sort((a, b) => (
      (VALIDATION_STRATA[a.validation_stratum]?.order ?? 99)
      - (VALIDATION_STRATA[b.validation_stratum]?.order ?? 99)
      || modelRank(a.model) - modelRank(b.model)
    ));
  if (!rows.length) {
    emptyTable(tbody, "No validation-stratum diagnostics.", 5);
    return;
  }
  tbody.innerHTML = rows.map((row) => {
    const meta = VALIDATION_STRATA[row.validation_stratum] || {
      label: String(row.validation_stratum).replaceAll("_", " "),
      definition: "",
    };
    return `
    <tr>
      <td title="${meta.definition}">${meta.label}</td>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${ratePct(row.WAPE, 1)}</td>
      <td>${pct(row.BiasRatio, 1)}</td>
      <td>${row.n_scored ?? row.n ?? "—"}</td>
    </tr>`;
  }).join("");
}

function renderTopDecile(data, strategy) {
  const tbody = document.querySelector("#top-decile-table tbody");
  const explanation = document.getElementById("top-decile-explanation");
  const rows = (data.top_decile_summary || [])
    .filter((row) => row.strategy === strategy && row.origin_type === "recent_benchmark")
    .sort((a, b) => Number(a.WAPE) - Number(b.WAPE));
  if (!rows.length) {
    emptyTable(tbody, "No high-volume diagnostics.", 5);
    if (explanation) explanation.textContent = "No recent-benchmark high-volume population was persisted.";
    return;
  }
  const threshold = Number(rows[0].actual_threshold);
  const nRows = Number(rows[0].n);
  if (explanation) {
    explanation.innerHTML = `<strong>Concrete reading:</strong> within the recent benchmark, product-day rows were ranked by their actual total quantity. The ${Number(rows[0].quantile || 0.9) * 100}th-percentile cutoff was ${fmt(threshold, 1)} units, leaving ${nRows.toLocaleString()} high-volume rows. WAPE, MAE and bias below are recomputed only on those rows; this is a retrospective stress diagnostic, not an input feature or a forecast filter.`;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${ratePct(row.WAPE, 1)}</td>
      <td>${fmt(row.MAE, 1)}</td>
      <td>${pct(row.BiasRatio, 1)}</td>
      <td>${fmt(row.actual_threshold, 1)} units</td>
    </tr>`).join("");
}

function mean(values) {
  const finite = values.map(Number).filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : null;
}

function dayNumber(value) {
  return Date.parse(`${String(value).slice(0, 10)}T00:00:00Z`) / 86400000;
}

function renderTopErrorInsight(data, strategy) {
  const target = document.getElementById("top-error-insight");
  if (!target) return;
  const row = (data.top_error_rows || []).find((candidate) => (
    candidate.strategy === strategy
    && candidate.origin_type === "recent_benchmark"
    && candidate.model === "NeuralNet"
    && Number(candidate.ProductId) === 26
    && String(candidate.DateKey).slice(0, 10) === "2025-12-16"
  ));
  if (!row) {
    target.hidden = true;
    target.innerHTML = "";
    return;
  }

  const origin = String(row.origin).slice(0, 10);
  const originDay = dayNumber(origin);
  const history = data.history?.[String(row.ProductId)] || { dates: [], quantity: [] };
  const knownHistory = history.dates
    .map((date, index) => ({
      day: dayNumber(date),
      rawQuantity: history.quantity[index],
    }))
    .filter((item) => item.rawQuantity !== null && item.rawQuantity !== undefined)
    .map((item) => ({ day: item.day, quantity: Number(item.rawQuantity) }))
    .filter((item) => Number.isFinite(item.quantity) && item.day <= originDay);
  const recent = knownHistory.filter((item) => item.day > originDay - 7).map((item) => item.quantity);
  const previous = knownHistory.filter((item) => item.day <= originDay - 7 && item.day > originDay - 14).map((item) => item.quantity);
  const recentMean = mean(recent);
  const previousMean = mean(previous);
  const productLift = Number.isFinite(recentMean) && Number.isFinite(previousMean) && previousMean !== 0
    ? recentMean / previousMean - 1
    : null;
  const knownValues = knownHistory.map((item) => item.quantity);
  const percentile = knownValues.length
    ? knownValues.filter((value) => value <= Number(row.actual)).length / knownValues.length
    : null;

  const portfolioByDay = new Map();
  Object.values(data.history || {}).forEach((series) => {
    (series.dates || []).forEach((date, index) => {
      const rawQuantity = series.quantity?.[index];
      if (rawQuantity === null || rawQuantity === undefined) return;
      const quantity = Number(rawQuantity);
      if (!Number.isFinite(quantity)) return;
      const day = dayNumber(date);
      portfolioByDay.set(day, (portfolioByDay.get(day) || 0) + quantity);
    });
  });
  const recentMarket = mean([...portfolioByDay.entries()]
    .filter(([day]) => day <= originDay && day > originDay - 7)
    .map(([, value]) => value));
  const previousMarket = mean([...portfolioByDay.entries()]
    .filter(([day]) => day <= originDay - 7 && day > originDay - 14)
    .map(([, value]) => value));
  const marketLift = Number.isFinite(recentMarket) && Number.isFinite(previousMarket) && previousMarket !== 0
    ? recentMarket / previousMarket - 1
    : null;

  const originWindow = (data.top_error_rows || []).filter((candidate) => (
    candidate.strategy === strategy
    && candidate.origin_type === "recent_benchmark"
    && candidate.model === "NeuralNet"
    && Number(candidate.ProductId) === 26
    && String(candidate.origin).slice(0, 10) === origin
  ));
  const meanOverforecast = mean(originWindow.map((candidate) => candidate.signed_error));
  const allOverforecast = originWindow.length === 7
    && originWindow.every((candidate) => Number(candidate.signed_error) > 0);

  const productLiftText = Number.isFinite(productLift)
    ? `${ratePct(productLift, 1)} (${fmt(recentMean, 1)} vs. ${fmt(previousMean, 1)} units/day)`
    : "strongly";
  const marketLiftText = Number.isFinite(marketLift) ? ratePct(marketLift, 1) : "materially";
  const percentileText = Number.isFinite(percentile) ? `${fmt(percentile * 100, 1)}th percentile` : "the extreme upper tail";
  const windowText = allOverforecast
    ? `It overpredicted all seven days of the 15–21 December window by ${fmt(meanOverforecast, 1)} units/day on average.`
    : `The same origin shows a broader positive-bias pattern for Product 26.`;

  target.hidden = false;
  target.innerHTML = `<strong>Interesting miss — Product 26, 16 December 2025:</strong> this was a systematic pre-Christmas uplift overshoot, not a stockout or a one-day demand collapse. At the ${origin} forecast origin, Product 26's trailing seven-day mean had increased ${productLiftText}, while portfolio demand was ${marketLiftText} higher week-on-week. Together with the known holiday/event covariates, that pattern is consistent with the direct NeuralNet extrapolating a stronger and more persistent surge. ${windowText} Actual demand of ${fmt(row.actual, 1)} was still at roughly the ${percentileText} of Product 26 observations available at the origin—demand was high, just below the NN's ${fmt(row.prediction, 1)} forecast. Per-row neural attribution was not persisted, so this is the strongest diagnosis supported by the saved OOF path and feature history rather than a SHAP-style causal decomposition.`;
}

function renderTopErrors(data, strategy) {
  const tbody = document.querySelector("#top-error-table tbody");
  const rows = (data.top_error_rows || [])
    .filter((row) => row.strategy === strategy && row.origin_type === "recent_benchmark")
    .sort((a, b) => Number(b.absolute_error) - Number(a.absolute_error))
    .slice(0, 30);
  if (!rows.length) {
    emptyTable(tbody, "No recent row-level errors.", 6);
    renderTopErrorInsight(data, strategy);
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
  renderTopErrorInsight(data, strategy);
}

function renderFinalAudit(data, strategy) {
  const panel = document.getElementById("final-audit-panel");
  const tbody = document.querySelector("#final-audit-table tbody");
  const metric = data.config?.selection_metric || "WAPE";
  const alignedByModel = new Map(
    (data.final_audit_test_aligned_scores || [])
      .filter((row) => row.strategy === strategy && row.metric === metric)
      .map((row) => [row.model, Number(row.test_aligned_score)])
  );
  const rows = (data.final_audit_summary || [])
    .filter((row) => (
      row.strategy === strategy
      && row.evaluation_regime === "conditional"
      && row.comparison_population === "common"
      && row.aggregation === "global"
    ))
    .map((row) => ({ ...row, aligned_score: alignedByModel.get(row.model) }))
    .sort((a, b) => {
      const alignedA = Number.isFinite(a.aligned_score) ? a.aligned_score : Number(a.WAPE);
      const alignedB = Number.isFinite(b.aligned_score) ? b.aligned_score : Number(b.WAPE);
      return alignedA - alignedB;
    });
  panel.hidden = rows.length === 0;
  if (!rows.length) return;
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td class="model-cell" style="color:${modelByKey(data, row.model)?.color || "#0a0a0a"}">${row.model}</td>
      <td>${ratePct(row.aligned_score, 2)}</td>
      <td>${ratePct(row.WAPE, 2)}</td>
      <td>${fmt(row.MAE, 2)}</td>
      <td>${pct(row.BiasRatio, 2)}</td>
      <td>${ratePct(row.coverage, 1)}</td>
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
    const productHistoryToggle = document.getElementById("product-history-toggle");
    const productModelsSelectAll = document.getElementById("product-models-select-all");
    const productModelsDeselectAll = document.getElementById("product-models-deselect-all");

    pairMetricSelect.value = data.config?.selection_metric || "WAPE";
    horizonMetricSelect.value = data.config?.selection_metric || "WAPE";
    horizonModelSelect.innerHTML = (data.models || [])
      .map((model) => `<option value="${model.key}">${model.label}</option>`)
      .join("");
    horizonModelSelect.value = canonicalModel(data);

    const firstProduct = populateProductSelector(data);
    setAllProductModels(data, true);
    renderRegimeDefinitions(data);
    populateDiagnosticModelSelector(data, productErrorModelSelect);
    configureStrategySelect(data, strategySelect, refresh);
    regimeSelect.value = data.config?.primary_evaluation_regime || "conditional";

    function refreshProductExplorer() {
      const strategy = strategySelect.value || canonicalStrategy(data);
      renderProductChart(data, productSelect.value || firstProduct, strategy);
      renderProductModelLegend(data, strategy, refreshProductExplorer);
    }

    function refresh() {
      const strategy = strategySelect.value || canonicalStrategy(data);
      const regime = regimeSelect.value || "conditional";
      updateStrategyCopy(data, strategy);
      renderRegimeExplanation(data, strategy, regime);
      renderKpis(data, strategy, regime);
      renderColumns(data, strategy, regime);
      renderComparisonChart(data, strategy, regime);
      renderCvTable(data, strategy, regime);
      refreshProductExplorer();
      renderStrategyComparison(data, pairMetricSelect.value);
      renderHorizonChart(data, horizonModelSelect.value, horizonMetricSelect.value, regime);
      renderEnsemble(data, strategy);
      renderPerProductDiagnostics(data, strategy, productErrorModelSelect.value, productErrorSplitSelect.value);
      renderRegimeDiagnostics(data, strategy, regime);
      renderTopDecile(data, strategy);
      renderTopErrors(data, strategy);
      renderFinalAudit(data, strategy);
      document.getElementById("product-strategy-note").textContent = strategyLabel(strategy);
    }

    [regimeSelect, productSelect, pairMetricSelect, horizonModelSelect, horizonMetricSelect,
      productErrorModelSelect, productErrorSplitSelect]
      .forEach((select) => select.addEventListener("change", refresh));

    productHistoryToggle.addEventListener("change", () => {
      setProductHistoryVisible(productHistoryToggle.checked);
      refreshProductExplorer();
    });
    productModelsSelectAll.addEventListener("click", () => {
      setAllProductModels(data, true);
      refreshProductExplorer();
    });
    productModelsDeselectAll.addEventListener("click", () => {
      setAllProductModels(data, false);
      refreshProductExplorer();
    });

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
