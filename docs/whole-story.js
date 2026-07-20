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
  note.innerHTML = `<strong>Published assignment:</strong> ${canonicalModel(data)} / ${strategyLabel(canonicalStrategy(data))} predicts total observed App + Web sales for ${data.config?.horizon || 7} days. It learns a baseline-relative correction; the competitive secondary blend is ${wholeStoryEnsembleCopy(data)}. This is a demand-forecasting result, not a price or promotion optimizer.`;
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
