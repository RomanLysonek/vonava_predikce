async function loadResults() {
  const res = await fetch("/api/results");
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
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

function availableStrategies(data) {
  const keys = Object.keys(data.forecasts_by_strategy || {});
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

function configureStrategySelect(data, select, onChange) {
  const strategies = availableStrategies(data);
  const selected = canonicalStrategy(data);
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
}

/** Renders the shared top nav: Overview + one pill per model. */
function renderNav(data, activeSlug) {
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const items = [{ slug: "", label: "Overview", color: "#ffffff" }].concat(
    (data.models || []).map((m) => ({ slug: m.slug, label: m.label, color: m.color })),
  );
  nav.innerHTML = items
    .map((it) => {
      const href = it.slug ? `/model/${it.slug}` : "/";
      const active = it.slug === (activeSlug || "");
      return `<a class="nav-pill${active ? " active" : ""}" style="--pill-color:${it.color}" href="${href}">${it.label}</a>`;
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
