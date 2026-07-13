const MODEL_ORDER = ["NeuralNet", "XGBoost", "LightGBM", "DynamicRidge", "SeasonalNaive", "MovingAvg28"];

function modelRank(model) {
  const idx = MODEL_ORDER.indexOf(model);
  return idx === -1 ? MODEL_ORDER.length : idx;
}

function renderKpis(data) {
  const summary = {};
  data.cv_summary.filter(r => r.aggregation === "global").forEach((row) => (summary[row.model] = row));
  const bestMae = data.cv_summary.filter(r => r.aggregation === "global").reduce((a, b) => (a.MAE <= b.MAE ? a : b));
  const bestRmse = data.cv_summary.filter(r => r.aggregation === "global").reduce((a, b) => (a.RMSE <= b.RMSE ? a : b));
  const nn = summary["NeuralNet"] || {};
  const nnMeta = modelByKey(data, "NeuralNet") || {};
  const skillPct = nnMeta.skill_vs_seasonal_naive;

  const cards = [
    {
      label: "Best MAE",
      value: fmt(bestMae.MAE),
      sub: bestMae.model,
      color: (modelByKey(data, bestMae.model) || {}).color,
    },
    {
      label: "Best RMSE",
      value: fmt(bestRmse.RMSE),
      sub: bestRmse.model,
      color: (modelByKey(data, bestRmse.model) || {}).color,
    },
    {
      label: "NeuralNet skill vs. Seasonal-Naive",
      value: skillPct !== null && skillPct !== undefined ? pct(skillPct) : "—",
      sub: "MAE improvement, avg over folds",
      color: nnMeta.color,
    },
    {
      label: "Models compared",
      value: String(data.cv_summary.length),
      sub: `${data.config.n_cv_folds} folds × ${data.config.horizon}d each`,
    },
  ];

  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = "";
  cards.forEach((c) => {
    const div = document.createElement("div");
    div.className = "kpi-card";
    div.innerHTML = `
      <p class="kpi-label">${c.label}</p>
      <p class="kpi-value" style="${c.color ? `color:${c.color}` : ""}">${c.value}</p>
      <p class="kpi-sub">${c.sub}</p>
    `;
    grid.appendChild(div);
  });
}

function renderColumns(data) {
  const grid = document.getElementById("columns-grid");
  const summary = {};
  data.cv_summary.filter(r => r.aggregation === "global").forEach((row) => (summary[row.model] = row));
  const kindLabel = { primary: "Submission", baseline: "Baseline", naive: "Naive" };

  grid.innerHTML = data.models
    .map((m) => {
      const s = summary[m.key] || {};
      const skill = m.skill_vs_seasonal_naive;
      return `
        <a class="model-column" style="--mc:${m.color}" href="/model/${m.slug}">
          <div class="model-column-header">
            <span class="model-badge">${kindLabel[m.kind] || m.kind}</span>
            <h3>${m.label}</h3>
            <span class="source">${m.short}</span>
          </div>
          <div class="model-stats">
            <div class="model-stat-row"><span>MAE</span><span>${fmt(s.MAE)}</span></div>
            <div class="model-stat-row"><span>RMSE</span><span>${fmt(s.RMSE)}</span></div>
            <div class="model-stat-row"><span>MAPE</span><span>${fmt(s.MAPE)}%</span></div>
            <div class="model-stat-row"><span>vs. Seasonal-Naive</span><span>${skill !== null && skill !== undefined ? pct(skill) : "—"}</span></div>
          </div>
          <span class="model-column-cta">View details →</span>
        </a>
      `;
    })
    .join("");
}

function renderComparisonChart(data) {
  const models = data.cv_summary.filter(r => r.aggregation === "global").map((r) => r.model);
  const colors = models.map((m) => (modelByKey(data, m) || {}).color || "#0a0a0a");
  const mae = data.cv_summary.filter(r => r.aggregation === "global").map((r) => r.MAE);
  const rmse = data.cv_summary.filter(r => r.aggregation === "global").map((r) => r.RMSE);

  new Chart(document.getElementById("chart-comparison"), {
    type: "bar",
    data: {
      labels: models,
      datasets: [
        { label: "MAE", data: mae, backgroundColor: colors, borderRadius: 0 },
        { label: "RMSE", data: rmse, backgroundColor: colors.map((c) => `${c}66`), borderColor: colors, borderWidth: 1, borderRadius: 0 },
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

function renderCvTable(data) {
  const tbody = document.querySelector("#cv-table tbody");
  tbody.innerHTML = "";
  data.cv_results
    .filter((r) => r.regime === "realized")
    .slice()
    .sort((a, b) => a.fold - b.fold || modelRank(a.model) - modelRank(b.model))
    .forEach((row) => {
      const color = (modelByKey(data, row.model) || {}).color || "#0a0a0a";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.fold}</td>
        <td class="model-cell" style="color:${color}">${row.model}</td>
        <td>${fmt(row.MAE)}</td>
        <td>${fmt(row.RMSE)}</td>
        <td>${fmt(row.MAPE)}%</td>
      `;
      tbody.appendChild(tr);
    });
}

let productChart = null;

function renderProductChart(data, productId) {
  const hist = data.history[productId];
  const labels = [...hist.dates];
  const bridge = hist.quantity[hist.quantity.length - 1];
  let firstForecastDates = null;

  const datasets = [
    {
      label: "History",
      data: [...hist.quantity],
      borderColor: "#0a0a0a",
      backgroundColor: "transparent",
      tension: 0.25,
      pointRadius: 0,
      borderWidth: 2,
    },
  ];

  data.models.forEach((m) => {
    const fc = (data.forecasts[m.key] || {})[productId];
    if (!fc) return;
    if (!firstForecastDates) firstForecastDates = fc.dates;
    datasets.push({
      label: m.label,
      data: [...hist.dates.slice(0, -1).map(() => null), bridge, ...fc.quantity],
      borderColor: m.color,
      backgroundColor: "transparent",
      borderDash: [6, 4],
      tension: 0.25,
      pointRadius: 3,
      borderWidth: 2,
    });
  });

  const allLabels = firstForecastDates ? [...labels, ...firstForecastDates] : labels;
  datasets[0].data = [...datasets[0].data, ...(firstForecastDates || []).map(() => null)];

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

function renderProductSelector(data) {
  const select = document.getElementById("product-select");
  const ids = Object.keys(data.history).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = ids.map((id) => `<option value="${id}">Product ${id}</option>`).join("");
  select.addEventListener("change", () => renderProductChart(data, select.value));
  renderProductChart(data, ids[0]);
}

function renderSubmissionTable(data) {
  const dates = [...new Set(data.submission.map((r) => r.DateKey))].sort();
  const productIds = [...new Set(data.submission.map((r) => r.ProductId))].sort((a, b) => a - b);
  const lookup = new Map(data.submission.map((r) => [`${r.ProductId}_${r.DateKey}`, r.Quantity]));

  let html = '<table class="data-table"><thead><tr><th>Product</th>';
  dates.forEach((d) => (html += `<th>${d}</th>`));
  html += "</tr></thead><tbody>";
  productIds.forEach((pid) => {
    html += `<tr><td>#${pid}</td>`;
    dates.forEach((d) => {
      html += `<td>${lookup.get(`${pid}_${d}`) ?? "—"}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("submission-table-wrap").innerHTML = html;
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "");
    renderKpis(data);
    renderColumns(data);
    renderComparisonChart(data);
    renderCvTable(data);
    renderProductSelector(data);
    renderSubmissionTable(data);
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load results</h2></div>
        <p style="color:#f87171">${err.message}</p>
      </div>`;
  }
}

main();
