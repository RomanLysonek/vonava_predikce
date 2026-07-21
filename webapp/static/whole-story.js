function wholeStoryEnsembleCopy(data) {
  const strategy = canonicalStrategy(data);
  const weights = data.ensemble?.strategies?.[strategy]?.weights || {};
  const entries = Object.entries(weights);
  if (!entries.length) return "No accepted secondary blend is present in this artifact";
  return entries
    .sort((left, right) => Number(right[1]) - Number(left[1]))
    .map(([model, weight]) => `${model} ${ratePct(weight, 0)}`)
    .join(" / ");
}

function renderWholeStoryCurrentDecision(data) {
  const note = document.getElementById("wholestory-current-note");
  if (!note) return;
  note.innerHTML = `<strong>What we finally used:</strong> ${canonicalModel(data)} / ${strategyLabel(canonicalStrategy(data))}, forecasting total App + Web sales ${data.config?.horizon || 7} days ahead. It adjusts a same-weekday baseline, and we also kept ${wholeStoryEnsembleCopy(data)} as a strong comparison. It forecasts the planned offer; it does not choose the price or promo.`;
}

function wireWholeStoryOverviewLink() {
  const link = document.querySelector?.("[data-overview-link]");
  if (link) link.href = overviewHref();
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "whole-story");
    updateStrategyCopy(data, canonicalStrategy(data));
    renderWholeStoryCurrentDecision(data);
    wireWholeStoryOverviewLink();
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load whole-story metadata</h2></div>
        <p style="color:var(--bad)">${err.message}</p>
      </div>`;
  }
}

main();
