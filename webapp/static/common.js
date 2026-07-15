async function loadResults() {
  const candidates = window.STATIC_DASHBOARD
    ? ["./results.json"]
    : ["/api/results", "/static/results.json", "./results.json"];
  let lastError = null;
  for (const url of candidates) {
    try {
      const res = await fetch(url);
      if (res.ok) return res.json();
      const body = await res.json().catch(() => ({}));
      lastError = new Error(body.detail || `HTTP ${res.status} from ${url}`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("Could not load forecast results");
}

function overviewHref() {
  return window.STATIC_DASHBOARD ? "index.html" : "/";
}

function modelHref(slug) {
  return window.STATIC_DASHBOARD
    ? `model.html?model=${encodeURIComponent(slug)}`
    : `/model/${slug}`;
}

function datasetHref() {
  return window.STATIC_DASHBOARD ? "dataset.html" : "/dataset";
}

function evaluationHref() {
  return window.STATIC_DASHBOARD ? "evaluation.html" : "/evaluation";
}

function wireSharedLinks() {
  if (typeof document === "undefined" || !document.querySelectorAll) return;
  document.querySelectorAll("[data-dataset-link]").forEach((link) => {
    link.href = datasetHref();
  });
  document.querySelectorAll("[data-evaluation-link]").forEach((link) => {
    link.href = evaluationHref();
  });
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function pct(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  const v = Number(n) * 100;
  return `${v >= 0 ? "+" : ""}${fmt(v, digits)}%`;
}

function ratePct(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return `${fmt(Number(n) * 100, digits)}%`;
}

function modelByKey(data, key) {
  return (data.models || []).find((m) => m.key === key || m.slug === key);
}

function availableStrategies(data, modelKey = null) {
  let keys = Object.keys(data.forecasts_by_strategy || {});
  if (modelKey) {
    const model = modelByKey(data, modelKey);
    if (!model) return [];
    const supported = model.strategies || keys;
    keys = keys.filter((strategy) => (
      supported.includes(strategy)
      && Boolean(data.forecasts_by_strategy?.[strategy]?.[model.key])
    ));
  }
  if (keys.length) return keys;
  const canonical = data.selection?.canonical_strategy || data.config?.primary_strategy;
  return canonical ? [canonical] : ["direct"];
}

function canonicalStrategy(data) {
  const preferred = data.selection?.canonical_strategy || data.config?.primary_strategy;
  const available = availableStrategies(data);
  return available.includes(preferred) ? preferred : available[0];
}

function canonicalModel(data) {
  return data.selection?.canonical_model || data.config?.submission_model || "NeuralNet";
}

function strategyLabel(strategy) {
  const labels = {
    direct: "Direct multi-horizon",
    recursive: "Recursive one-step",
    both: "Direct + recursive",
  };
  return labels[strategy] || strategy || "Forecast";
}

function forecastsFor(data, strategy) {
  return data.forecasts_by_strategy?.[strategy] || data.forecasts || {};
}

function summaryRows(
  data,
  {
    strategy = canonicalStrategy(data),
    regime = "conditional",
    population = "common",
    aggregation = "global",
    source = "benchmark",
  } = {},
) {
  const firstNonEmpty = (...candidates) => (
    candidates.find((candidate) => Array.isArray(candidate) && candidate.length > 0) || []
  );
  let rows;
  if (source === "development") {
    rows = firstNonEmpty(data.dev_summary_all, data.dev_summary);
  } else {
    rows = firstNonEmpty(data.benchmark_summary_all, data.benchmark_summary, data.cv_summary);
  }

  return rows.filter((row) => {
    if (row.strategy && row.strategy !== strategy) return false;

    if (row.evaluation_regime) {
      if (row.evaluation_regime !== regime) return false;
      if (row.comparison_population && row.comparison_population !== population) return false;
      return row.aggregation === aggregation;
    }

    // Compatibility with pre-B4 payloads where regime was encoded in the
    // aggregation string.
    const legacyAggregation = regime === "conditional"
      ? `${aggregation}_conditional`
      : aggregation;
    return row.aggregation === legacyAggregation;
  });
}

function cvRows(
  data,
  {
    strategy = canonicalStrategy(data),
    regime = "conditional",
    population = "common",
  } = {},
) {
  const rows = (
    Array.isArray(data.cv_results_all) && data.cv_results_all.length > 0
      ? data.cv_results_all
      : (data.cv_results || [])
  );
  return rows.filter((row) => {
    if (row.strategy && row.strategy !== strategy) return false;
    if (row.regime && row.regime !== regime) return false;
    if (row.evaluation_regime && row.evaluation_regime !== regime) return false;
    if (row.comparison_population && row.comparison_population !== population) return false;
    return true;
  });
}

function configureStrategySelect(data, select, onChange, modelKey = null) {
  const strategies = availableStrategies(data, modelKey);
  const canonical = canonicalStrategy(data);
  const selected = strategies.includes(canonical) ? canonical : strategies[0];
  select.innerHTML = strategies
    .map((strategy) => `<option value="${strategy}">${strategyLabel(strategy)}</option>`)
    .join("");
  select.value = selected;
  select.disabled = strategies.length < 2;
  select.addEventListener("change", onChange);
  return selected;
}

function updateStrategyCopy(data, strategy) {
  const promo = document.getElementById("promo-strategy");
  if (promo) promo.textContent = `${data.config?.horizon || 7}-Day ${strategyLabel(strategy)} Forecast`;

  const canonical = `${canonicalModel(data)} / ${strategyLabel(canonicalStrategy(data))}`;
  const canonicalText = document.getElementById("canonical-selection-text");
  if (canonicalText) canonicalText.textContent = `Canonical submission: ${canonical}`;

  const footer = document.getElementById("footer-method-text");
  if (footer) {
    footer.textContent = `Canonical submission: ${canonical}. The dashboard can compare every available strategy without changing the submitted forecast.`;
  }
  const modelCount = document.getElementById("promo-model-count");
  if (modelCount) modelCount.textContent = `${(data.models || []).length} Models Compared`;
}

/** Renders the shared top nav: Overview + Dataset + Evaluation + one pill per model. */
function renderNav(data, activeSlug) {
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const items = [
    { slug: "", label: "Overview", color: "#ffffff", href: overviewHref() },
    { slug: "dataset", label: "Data story", color: "#a78bfa", href: datasetHref() },
    { slug: "evaluation", label: "Evaluation", color: "#9ca3af", href: evaluationHref() },
    ...(data.models || []).map((m) => ({
      slug: m.slug,
      label: m.label,
      color: m.color,
      href: modelHref(m.slug),
    })),
  ];
  nav.innerHTML = items
    .map((it) => {
      const active = it.slug === (activeSlug || "");
      return `<a class="nav-pill${active ? " active" : ""}" style="--pill-color:${it.color}" href="${it.href}">${it.label}</a>`;
    })
    .join("");
}

const CHART_GRID = "#e4e4e4";
const CHART_TEXT = "#6b6b6b";

if (window.Chart) {
  Chart.defaults.color = CHART_TEXT;
  Chart.defaults.font.family = "Roboto, -apple-system, sans-serif";
  Chart.defaults.borderColor = CHART_GRID;
}

wireSharedLinks();
