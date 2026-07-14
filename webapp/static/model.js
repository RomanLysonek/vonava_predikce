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

function modelSkill(summary, modelName) {
  const modelMae = summary[modelName]?.MAE;
  const naiveMae = summary.SeasonalNaive?.MAE;
  if (!Number.isFinite(Number(modelMae)) || !Number.isFinite(Number(naiveMae)) || Number(naiveMae) === 0) {
    return null;
  }
  return 1 - Number(modelMae) / Number(naiveMae);
}

function renderKpis(data, model, strategy, regime) {
  const rows = summaryRows(data, { strategy, regime });
  const byModel = Object.fromEntries(rows.map((row) => [row.model, row]));
  const summary = byModel[model.key] || {};
  const skill = modelSkill(byModel, model.key);
  const cards = [
    { label: "MAE", value: fmt(summary.MAE), sub: "global common population" },
    { label: "WAPE", value: ratePct(summary.WAPE), sub: `${regime} demand` },
    { label: "Bias", value: fmt(summary.Bias), sub: "positive = over-forecast" },
    {
      label: "Skill vs. Naive",
      value: skill === null ? "—" : pct(skill),
      sub: `${strategyLabel(strategy)} · MAE improvement`,
    },
  ];
  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = cards.map((card) => `
    <div class="kpi-card">
      <p class="kpi-label">${card.label}</p>
      <p class="kpi-value model-accent">${card.value}</p>
      <p class="kpi-sub">${card.sub}</p>
    </div>
  `).join("");
}

let foldChart = null;
function renderFoldChart(data, model, strategy, regime) {
  const rows = cvRows(data, { strategy, regime })
    .filter((row) => row.model === model.key)
    .sort((a, b) => a.fold - b.fold);

  if (foldChart) foldChart.destroy();
  foldChart = new Chart(document.getElementById("chart-folds"), {
    type: "bar",
    data: {
      labels: rows.map((row) => `Fold ${row.fold}`),
      datasets: [
        { label: "MAE", data: rows.map((row) => row.MAE), backgroundColor: model.color, borderRadius: 0 },
        {
          label: "RMSE",
          data: rows.map((row) => row.RMSE),
          backgroundColor: `${model.color}66`,
          borderColor: model.color,
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

function renderFoldTable(data, model, strategy, regime) {
  const rows = cvRows(data, { strategy, regime })
    .filter((row) => row.model === model.key)
    .sort((a, b) => a.fold - b.fold);
  document.querySelector("#fold-table tbody").innerHTML = rows.map((row) => `
    <tr>
      <td>${row.fold}</td>
      <td>${fmt(row.MAE)}</td>
      <td>${fmt(row.RMSE)}</td>
      <td style="color:${Number(row.Bias) >= 0 ? "var(--bad)" : "var(--good)"}">${fmt(row.Bias)}</td>
      <td>${pct(row.BiasRatio)}</td>
      <td>${row.n ?? row.n_scored ?? "—"}</td>
    </tr>
  `).join("");
}

let productChart = null;
function renderProductChart(data, model, productId, strategy) {
  const hist = data.history[productId];
  const forecast = forecastsFor(data, strategy)[model.key]?.[productId] || { dates: [], quantity: [] };
  const labels = [...hist.dates, ...forecast.dates];
  const historyData = [...hist.quantity, ...forecast.dates.map(() => null)];
  const bridge = hist.quantity[hist.quantity.length - 1];
  const forecastData = [...hist.dates.slice(0, -1).map(() => null), bridge, ...forecast.quantity];

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
          label: `${model.label} · ${strategyLabel(strategy)}`,
          data: forecastData,
          borderColor: model.color,
          backgroundColor: "transparent",
          borderDash: strategy === "recursive" ? [7, 4] : [6, 4],
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

function populateProductSelector(data) {
  const select = document.getElementById("product-select");
  const ids = Object.keys(data.history || {}).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = ids.map((id) => `<option value="${id}">Product ${id}</option>`).join("");
  return ids[0];
}

function renderNotFound(data, slug) {
  document.getElementById("app").innerHTML = `
    <div class="panel">
      <div class="panel-header"><h2>Unknown model "${slug}"</h2></div>
      <p style="color:var(--text-dim)">Try one of: ${(data.models || []).map((model) => `<a href="/model/${model.slug}" style="color:${model.color}">${model.label}</a>`).join(", ")}</p>
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

    const strategySelect = document.getElementById("strategy-select");
    const regimeSelect = document.getElementById("regime-select");
    const productSelect = document.getElementById("product-select");
    const firstProduct = populateProductSelector(data);

    configureStrategySelect(data, strategySelect, refresh, model.key);
    regimeSelect.value = data.config?.primary_evaluation_regime || "conditional";

    function refresh() {
      const strategy = strategySelect.value || canonicalStrategy(data);
      const regime = regimeSelect.value || "conditional";
      updateStrategyCopy(data, strategy);
      renderKpis(data, model, strategy, regime);
      renderFoldChart(data, model, strategy, regime);
      renderFoldTable(data, model, strategy, regime);
      renderProductChart(data, model, productSelect.value || firstProduct, strategy);
      document.getElementById("model-strategy-note").textContent = strategyLabel(strategy);
    }

    [regimeSelect, productSelect].forEach((select) => select.addEventListener("change", refresh));

    renderNav(data, currentSlug());
    renderHero(model);
    refresh();
    document.getElementById("footer-note").innerHTML =
      `Comparing against the other ${(data.models || []).length - 1} models? See the <a href="/" style="color:${model.color}">Overview page</a>.`;
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load results</h2></div>
        <p style="color:var(--bad)">${err.message}</p>
      </div>`;
  }
}

main();
