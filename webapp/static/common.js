async function loadResults() {
  const res = await fetch("/api/results");
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function pct(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const v = n * 100;
  return `${v >= 0 ? "+" : ""}${fmt(v, digits)}%`;
}

function modelByKey(data, key) {
  return data.models.find((m) => m.key === key || m.slug === key);
}

/** Renders the shared top nav: Overview + one pill per model, dot-colored
 * to match that model's brand color, active pill highlighted. */
function renderNav(data, activeSlug) {
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const items = [{ slug: "", label: "Overview", color: "#ffffff" }].concat(
    data.models.map((m) => ({ slug: m.slug, label: m.label, color: m.color }))
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
