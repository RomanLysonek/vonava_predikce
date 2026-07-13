function currentSlug() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return parts[parts.length - 1] || "";
}

const KIND_LABEL = { primary: "Submission", baseline: "Baseline", naive: "Naive" };

function renderHero(model) {
  document.title = `${model.label} — Notino Quantity Forecast`;
  const hero = document.getElementById("model-hero");
  hero.style.setProperty("--mc", model.color);
  document.getElementById("hero-badge").textContent = KIND_LABEL[model.kind] || model.kind;
  document.getElementById("hero-title").textContent = model.label;
  document.getElementById("hero-blurb").textContent = model.blurb;
  const link = document.getElementById("hero-source");
  if (model.source_url) {
    link.href = model.source_url;
    link.style.display = "inline-flex";
    link.textContent = `View ${model.short} ↗`;
  }
}

function renderKpis(data, model, regime = "realized") {
  const suffix = regime === "conditional" ? "_conditional" : "";
  const agg = `global${suffix}`;
  const summary = data.cv_summary.find((r) => r.model === model.key && r.aggregation === agg) || {};
  const skill = model.skill_vs_seasonal_naive;
  const cards = [
    { label: "MAE", value: fmt(summary.MAE), sub: "avg over folds" },
    { label: "RMSE", value: fmt(summary.RMSE), sub: "avg over folds" },
    { label: "Bias", value: fmt(summary.Bias), sub: "total over/under forecast" },
    {
      label: "Skill vs. Naive",
      value: (regime === 'realized' && skill !== null && skill !== undefined) ? pct(skill) : "—",
      sub: "MAE improvement over lag-7",
    },
  ];
  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = "";
  cards.forEach((c) => {
    const div = document.createElement("div");
    div.className = "kpi-card";
    div.innerHTML = `
      <p class="kpi-label">${c.label}</p>
      <p class="kpi-value model-accent">${c.value}</p>
      <p class="kpi-sub">${c.sub}</p>
    `;
    grid.appendChild(div);
  });
}

function renderFoldChart(data, model, regime = "realized") {
  const rows = data.cv_results.filter((r) => r.model === model.key && r.regime === regime).sort((a, b) => a.fold - b.fold);
  new Chart(document.getElementById("chart-folds"), {
    type: "bar",
    data: {
      labels: rows.map((r) => `Fold ${r.fold}`),
      datasets: [
        { label: "MAE", data: rows.map((r) => r.MAE), backgroundColor: model.color, borderRadius: 0 },
        { label: "RMSE", data: rows.map((r) => r.RMSE), backgroundColor: `${model.color}66`, borderColor: model.color, borderWidth: 1, borderRadius: 0 },
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

function renderFoldTable(data, model, regime = "realized") {
  const tbody = document.querySelector("#fold-table tbody");
  const rows = data.cv_results.filter((r) => r.model === model.key && r.regime === regime).sort((a, b) => a.fold - b.fold);
  tbody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${row.fold}</td>
        <td>${fmt(row.MAE)}</td>
        <td>${fmt(row.RMSE)}</td>
        <td style="color:${row.Bias >= 0 ? 'var(--bad)' : 'var(--good)'}">${fmt(row.Bias)}</td>
        <td>${fmt(row.BiasRatio * 100, 1)}%</td>
        <td>${row.n}</td>
      </tr>`
    )
    .join("");
}

let productChart = null;

function renderProductChart(data, model, productId) {
  const hist = data.history[productId];
  const fc = (data.forecasts[model.key] || {})[productId] || { dates: [], quantity: [] };
  const labels = [...hist.dates, ...fc.dates];
  const historyData = [...hist.quantity, ...fc.dates.map(() => null)];
  const bridge = hist.quantity[hist.quantity.length - 1];
  const forecastData = [...hist.dates.slice(0, -1).map(() => null), bridge, ...fc.quantity];

  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "History",
          data: historyData,
          borderColor: "#0a0a0a",
          backgroundColor: "transparent",
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: `${model.label} · 7-day forecast`,
          data: forecastData,
          borderColor: model.color,
          backgroundColor: "transparent",
          borderDash: [6, 4],
          tension: 0.25,
          pointRadius: 3,
          borderWidth: 2,
        },
      ],
    },
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

function renderProductSelector(data, model) {
  const select = document.getElementById("product-select");
  const ids = Object.keys(data.history).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = ids.map((id) => `<option value="${id}">Product ${id}</option>`).join("");
  select.addEventListener("change", () => renderProductChart(data, model, select.value));
  renderProductChart(data, model, ids[0]);
}

function renderNotFound(data, slug) {
  document.getElementById("app").innerHTML = `
    <div class="panel">
      <div class="panel-header"><h2>Unknown model "${slug}"</h2></div>
      <p style="color:var(--text-dim)">Try one of: ${data.models.map((m) => `<a href="/model/${m.slug}" style="color:${m.color}">${m.label}</a>`).join(", ")}</p>
    </div>`;
  document.getElementById("model-hero").style.display = "none";
}

async function main() {
  try {
    const data = await loadResults();
    const model = modelByKey(data, currentSlug());
    if (!model) {
      renderNotFound(data, currentSlug());
      return;
    }

    const regimeSelect = document.getElementById("regime-select");
    const refresh = () => {
      const r = regimeSelect.value;
      renderKpis(data, model, r);
      renderFoldChart(data, model, r);
      renderFoldTable(data, model, r);
    };
    regimeSelect.addEventListener("change", refresh);

    renderNav(data, currentSlug());
    renderHero(model);
    refresh();
    renderProductSelector(data, model);
    document.getElementById("footer-note").innerHTML =
      `Comparing against the other 4 models? See the <a href="/" style="color:${model.color}">Overview page</a>.`;
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load results</h2></div>
        <p style="color:#f87171">${err.message}</p>
      </div>`;
  }
}

main();
