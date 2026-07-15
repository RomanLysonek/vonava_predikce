function currentSlug() {
  const query = new URLSearchParams(window.location.search).get("model");
  if (query) return query;
  const parts = window.location.pathname.split("/").filter(Boolean);
  return parts[parts.length - 1] || "";
}

const KIND_LABEL = { primary: "Submission", ensemble: "OOF Ensemble", baseline: "Baseline", naive: "Naive" };

function renderHero(model) {
  document.title = `${model.label} — Notino Quantity Forecast`;
  const hero = document.getElementById("model-hero");
  document.documentElement.style.setProperty("--mc", model.color);
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


function modelForecastMechanism(strategy) {
  if (strategy === "recursive") {
    return "The recursive view predicts one day, appends that prediction as synthetic history, rebuilds the lag features, and then predicts the next day. This exposes the model to compounding error, so it is diagnostic rather than the selected submission route.";
  }
  return "The selected direct multi-horizon view creates a separate row for horizons 1–7 at one fixed origin. Every day is predicted from genuinely observed pre-origin history plus covariates known for the target date; earlier forecast values are never fed into later horizons.";
}

function friendlyFeatureGroups(groups) {
  const labels = {
    price: "price and effective-price context",
    campaign: "campaign semantics",
    lifecycle: "availability and lifecycle state",
    market: "portfolio-level demand context",
    event: "retail-event proximity",
  };
  return (groups || []).map((group) => labels[group] || group).join(", ");
}

function ensembleWeightsText(data, strategy) {
  const strategyFit = data.ensemble?.strategies?.[strategy]
    || data.ensemble?.strategies?.direct
    || {};
  const weights = strategyFit.weights || {};
  const entries = Object.entries(weights);
  if (!entries.length) return "the configured member predictions";
  return entries.map(([name, value]) => `${ratePct(value, 0)} ${name}`).join(" + ");
}

function finalAuditAligned(data, modelKey) {
  const row = (data.final_audit_test_aligned_scores || []).find(
    (candidate) => candidate.model === modelKey,
  );
  return Number.isFinite(Number(row?.test_aligned_score))
    ? ratePct(row.test_aligned_score)
    : null;
}

function modelMethodology(data, model, strategy) {
  const cfg = data.config || {};
  const directMechanism = modelForecastMechanism(strategy);
  const featureGroups = friendlyFeatureGroups(cfg.c2_feature_groups);
  const historyScope = cfg.training_window_days == null
    ? "all available training history"
    : `the latest ${cfg.training_window_days} training days`;
  const recency = cfg.recency_half_life_days == null
    ? "without recency weighting"
    : `with a ${cfg.recency_half_life_days}-day recency half-life`;
  const auditNn = finalAuditAligned(data, "NeuralNet");
  const auditEnsemble = finalAuditAligned(data, "Ensemble");

  const commonPanel = `The structured panel combines future-known calendar, price, discount, campaign and event covariates with availability-aware 7/14/28-day rolling history, recent point lags, same-weekday seasonal lags, annual references, lifecycle state and market context. The currently retained semantic groups are ${featureGroups || "the configured feature set"}.`;

  const methods = {
    NeuralNet: {
      intro: "The submitted forecast is the end result of a controlled selection path: a direct, baseline-relative PyTorch model whose complexity was retained only where rolling-origin validation supported it.",
      items: [
        ["Final architecture", `Three feed-forward networks, seeded ${Array.isArray(cfg.seeds) ? cfg.seeds.join(", ") : "independently"}, are trained independently and averaged to reduce seed-level variance. Product, web-campaign, app-campaign and forecast-horizon identifiers use learned embeddings; numeric inputs are median-imputed with explicit missingness indicators and standardized.`],
        ["What the network predicts", `Instead of learning raw quantity from scratch, it predicts log1p(actual quantity) minus log1p of the 4:3:2:1 weighted 7/14/21/28-day same-weekday baseline. The retained ${String(cfg.nn_loss || "selected").toUpperCase()} objective learns the correction around that strong seasonal anchor; the correction is then added back in log space and transformed to nonnegative natural-scale demand.`],
        ["Information available to it", `${commonPanel} The final fit uses ${historyScope} ${recency}, ${cfg.final_epochs || "the configured number of"} epochs and batch size ${cfg.nn_batch_size || "as configured"}. Future inputs are limited to covariates known for the target date; availability is used to construct safer history and scoring populations, not as an unavailable test-time predictor.`],
        ["Seven-day forecast", `${directMechanism} The same global network handles the stacked product × horizon panel, with horizon supplied explicitly, so Monday through Sunday can have different responses without feeding predicted demand back into later days.`],
        ["What screening removed", "Extra trend features, exponential recency decay, channel-history features and an auxiliary app-share output were all plausible from the data profile, but none earned a place in the final configuration on the frozen test-aligned objective. The app/web experiment is summarized in the decision panel below."],
        ["How we use it", `NeuralNet / Direct multi-horizon generates the canonical 210-row submission grid. It remained the submission after the untouched final audit${auditNn ? ` (${auditNn} aligned WAPE)` : ""}; the convex ensemble is retained as a strong secondary forecast rather than replacing the predeclared canonical output after audit inspection.`],
      ],
    },
    Ensemble: {
      intro: "A frozen row-by-row convex blend used as a high-performing secondary forecast, not as the submitted model.",
      items: [
        ["Mechanism", `For each product-date row, the ensemble calculates ${ensembleWeightsText(data, strategy)}. Weights are nonnegative and sum to one, so the result remains inside the range implied by the member forecasts rather than introducing a separate extrapolating model.`],
        ["How the weights were fitted", `An exhaustive 0.01 simplex grid was evaluated only on development out-of-fold predictions shared by NeuralNet, XGBoost and LightGBM. The objective was the frozen test-aligned WAPE: 60% January/February proxy, 25% regular periods and 15% holiday/event stress. No test-week actuals or final-audit rows entered the fit.`],
        ["Seven-day forecast", `${directMechanism} The ensemble has no independent features, target or training pass: it combines the already-produced member quantities at each horizon. Its weights were frozen before recent-benchmark confirmation, the final audit and final forecasting.`],
        ["How we use it", `It is retained for comparison and as an alternative forecast because it improved development and recent-benchmark performance. The canonical submission remains NeuralNet because the untouched final audit slightly favored NeuralNet on the predeclared aligned objective${auditNn && auditEnsemble ? ` (${auditNn} vs. ${auditEnsemble})` : ""}, even though the ensemble was marginally better on broad global WAPE.`],
      ],
    },
    XGBoost: {
      intro: "A gradient-boosted decision-tree benchmark and one of the three frozen ensemble members.",
      items: [
        ["Mechanism", "The fitted XGBRegressor uses 400 histogram-based boosting rounds, depth-5 trees, learning rate 0.05 and row/feature subsampling. Product and campaign identifiers are native categorical inputs; successive trees correct the residual errors of the preceding tree ensemble."],
        ["Training target and inputs", `The selected XGBoost variant predicts the same baseline-relative log residual used by the neural network, then reconstructs natural-scale quantity from the 4:3:2:1 same-weekday baseline. It is trained on ${historyScope} ${recency}. ${commonPanel}`],
        ["Seven-day forecast", `${directMechanism} One model handles the stacked horizons, with horizon supplied as an ordered feature. Reconstructed predictions are clipped at zero.`],
        ["How we use it", `XGBoost is an honest standard-approach comparator and contributes ${ratePct(data.ensemble?.strategies?.direct?.weights?.XGBoost ?? 0, 0)} to the accepted direct ensemble. It is evaluated on exactly the same common rows as the submission model but is not itself used for the canonical submission.`],
      ],
    },
    LightGBM: {
      intro: "A leaf-wise gradient-boosted tree benchmark and the largest-weight member of the frozen ensemble.",
      items: [
        ["Mechanism", "The fitted LGBMRegressor uses 400 boosting rounds, 31 leaves, learning rate 0.05 and 0.8 feature subsampling. Its leaf-wise growth can capture nonlinear interactions efficiently, while product and campaign identifiers remain categorical features."],
        ["Training target and inputs", `The selected LightGBM variant directly predicts log1p(total quantity), which is transformed back with expm1 and clipped at zero. It is trained on ${historyScope} ${recency}. ${commonPanel}`],
        ["Seven-day forecast", `${directMechanism} A single stacked-panel model learns horizon-dependent splits from the horizon feature rather than fitting seven unrelated models.`],
        ["How we use it", `LightGBM is a standard tree baseline, was the strongest single development model used during ensemble fitting, and contributes ${ratePct(data.ensemble?.strategies?.direct?.weights?.LightGBM ?? 0, 0)} to the frozen direct ensemble. It is not the canonical submitted model.`],
      ],
    },
    DynamicRidge: {
      intro: "A deliberately simple structured linear benchmark showing how far nonlinear models improve over an L2-regularized panel model.",
      items: [
        ["Mechanism", "Numeric inputs are median-imputed and standardized; product and campaign categories are one-hot encoded. Ridge regression with L2 penalty alpha 10 fits one global linear coefficient set, shrinking correlated feature effects rather than creating tree splits or neural interactions."],
        ["Training target and inputs", `It predicts the baseline-relative log residual and reconstructs quantity from the same 4:3:2:1 same-weekday baseline. It receives the same direct structured feature panel as the nonlinear models, including the retained ${featureGroups || "semantic"} groups.`],
        ["Seven-day forecast", "Dynamic Ridge is direct-only. Recursive Ridge was removed after it remained statistically unstable even with numerical guards. Unsafe direct reconstructions fall back to the seasonal baseline, and all outputs are constrained to nonnegative quantities."],
        ["How we use it", "It is a structured statistical baseline: useful for demonstrating the incremental value of nonlinear interactions and embeddings. It is included in validation and Product Explorer comparisons, but it is neither an ensemble member nor a submission candidate."],
      ],
    },
    SeasonalNaive: {
      intro: "A no-training weekly persistence baseline: each target day repeats the same product’s observed demand from exactly seven days earlier.",
      items: [
        ["Mechanism", "For every product and target date, the forecast looks up Quantity on target date minus seven calendar days. Because each horizon references a different prior-week day, the seven-day line can rise and fall rather than remaining flat."],
        ["Availability handling", "Only observed-and-available demand is accepted as the lag-7 value. If that exact day was unavailable, a calendar gap, or otherwise missing, the row falls back to the availability-aware 4:3:2:1 weighted same-weekday baseline rather than treating censored demand as zero."],
        ["Seven-day forecast", "There are no fitted parameters and no feedback loop. The direct and recursive labels therefore produce the same weekly-persistence values; the baseline simply copies the corresponding shape of the preceding week."],
        ["How we use it", "This is the principal sanity-check floor and the denominator for the displayed ‘skill vs. naive’ metric. A trained model should beat it consistently; otherwise the extra complexity has not demonstrated useful forecasting skill."],
      ],
    },
    MovingAvg28: {
      intro: "A no-training local-level baseline that deliberately ignores horizon shape and repeats one recent average across the whole forecast week.",
      items: [
        ["Mechanism", "At the forecast origin, we take each product’s latest 28 calendar rows, exclude stockout or unknown-gap rows, and calculate the arithmetic mean of the remaining observed quantities. That single product-level number is used as the forecast."],
        ["Seven-day forecast", "The same mean is repeated for horizons 1 through 7, so the plotted forecast jumps from the last observation to one value and stays perfectly flat. Direct and recursive views are identical because no predicted value is fed back and no target-date feature is used."],
        ["When it works or fails", "It can resemble the ensemble when the product’s recent demand level is stable. It lags abrupt regime changes: for Products 1 and 26, high pre-Christmas observations remain inside the 28-day window even though January demand has already fallen, pulling the flat mean far above the current level."],
        ["How we use it", "It is an intentionally weak floor answering whether the trained models add value beyond a recent average. It is never considered for the canonical submission or the fitted ensemble; its poor test-aligned WAPE is useful evidence that seasonality, covariates and horizon-specific structure matter."],
      ],
    },
  };

  const method = methods[model.key] || {
    intro: model.blurb || "Model implementation details are unavailable.",
    items: [["How we use it", model.blurb || "No additional description is available."]],
  };
  let evaluationAction;
  if (model.key === "Ensemble") {
    evaluationAction = "Its member forecasts come from origin-safe model fits; the convex weights are fitted once from development OOF and then frozen before recent-benchmark confirmation and the final audit";
  } else if (["SeasonalNaive", "MovingAvg28"].includes(model.key)) {
    evaluationAction = "It is recomputed at every origin using only history available by that date";
  } else {
    evaluationAction = "It is refitted at every origin using only training information available by that date";
  }
  return {
    ...method,
    items: [
      ...method.items,
      [
        "Evaluation contract",
        `${evaluationAction}, then scored on the following seven days under the ${strategyLabel(strategy)} contract. Model comparisons use the same conditional-demand common population; walk-forward validation is the outer evaluation loop and does not mean recursive inference.`,
      ],
    ],
  };
}

function renderModelMethod(data, model, strategy) {
  const method = modelMethodology(data, model, strategy);
  const intro = document.getElementById("model-method-intro");
  const list = document.getElementById("model-method-list");
  if (!intro || !list) return;
  intro.textContent = method.intro;
  list.innerHTML = method.items.map(([title, description]) => `
    <div class="definition-item model-method-item">
      <strong>${title}</strong>
      <span>${description}</span>
    </div>
  `).join("") + `
    <div class="model-method-links">
      <a class="model-evaluation-link" href="${datasetHref()}">See the dataset story & decision rationale →</a>
      <a class="model-evaluation-link" href="${evaluationHref()}">See the complete evaluation process →</a>
    </div>
  `;
}

function ablationRows(data, stage) {
  return (data.ablation_showcase || []).filter((row) => row.stage === stage);
}

function alignedScoreText(value) {
  return Number.isFinite(Number(value)) ? ratePct(value, 2) : "not persisted";
}

function renderNeuralNetSelection(data, model, strategy) {
  const panel = document.getElementById("nn-selection-panel");
  const grid = document.getElementById("nn-selection-grid");
  if (!panel || !grid) return;
  if (model.key !== "NeuralNet") {
    panel.hidden = true;
    grid.innerHTML = "";
    return;
  }

  const cfg = data.config || {};
  const channelRows = ablationRows(data, "channel_aux");
  const channelControl = channelRows.find((row) => row.candidate === "channel_control");
  const bestChannel = channelRows
    .filter((row) => row.candidate !== "channel_control" && Number.isFinite(Number(row.test_aligned_WAPE)))
    .sort((a, b) => Number(a.test_aligned_WAPE) - Number(b.test_aligned_WAPE))[0];
  const channelDeterioration = channelControl && bestChannel && Number(channelControl.test_aligned_WAPE) !== 0
    ? Number(bestChannel.test_aligned_WAPE) / Number(channelControl.test_aligned_WAPE) - 1
    : null;
  const auditNn = (data.final_audit_test_aligned_scores || []).find((row) => row.model === "NeuralNet");
  const auditEnsemble = (data.final_audit_test_aligned_scores || []).find((row) => row.model === "Ensemble");
  const groups = friendlyFeatureGroups(cfg.c2_feature_groups) || "the retained structured feature groups";
  const seeds = Array.isArray(cfg.seeds) ? cfg.seeds.join(", ") : "three fixed seeds";
  const channelEvidence = channelControl && bestChannel
    ? `The control scored ${alignedScoreText(channelControl.test_aligned_WAPE)} aligned WAPE. The best channel-aware candidate scored ${alignedScoreText(bestChannel.test_aligned_WAPE)}${Number.isFinite(channelDeterioration) ? `, ${ratePct(channelDeterioration, 2)} worse` : ""}.`
    : "The persisted screening did not show an improvement from the channel-aware candidates.";
  const auditEvidence = auditNn && auditEnsemble
    ? `On the untouched aligned audit, NeuralNet scored ${alignedScoreText(auditNn.test_aligned_score)} versus ${alignedScoreText(auditEnsemble.test_aligned_score)} for the ensemble.`
    : "The canonical model decision was frozen before final forecasting.";

  const cards = [
    {
      status: "Retained",
      title: "Direct seven-day formulation",
      body: `All seven horizons are created at one origin from observed history and target-date covariates. Recursive feedback remains a diagnostic alternative, not the submitted route. Current contract: ${strategyLabel(strategy)}.`,
    },
    {
      status: "Retained",
      title: "Seasonal anchor + learned correction",
      body: `The 4:3:2:1 same-weekday baseline carries stable weekly structure; the network learns only the log-scale departure from it using ${String(cfg.nn_loss || "the selected").toUpperCase()} loss. This proved more robust than asking the network to rediscover the full level unaided.`,
    },
    {
      status: "Retained",
      title: "Structured global context",
      body: `One model pools all 30 products while preserving product and campaign identity through embeddings. The retained inputs cover ${groups}; extra trend features were left out after screening.`,
    },
    {
      status: "Retained",
      title: "Stable final fit",
      body: `The final model uses all available history without exponential recency weighting, batch size ${cfg.nn_batch_size || "the selected batch size"}, ${cfg.final_epochs || "the selected number of"} epochs and seeds ${seeds}. Averaging the three fits reduces dependence on one random initialization.`,
    },
    {
      status: "Rejected",
      title: "App-vs-web split experiment",
      body: `Because app share changed dramatically over time, we tested channel-history inputs and a second output predicting app share alongside total demand. ${channelEvidence} The idea was therefore rejected for the submitted model: total quantity remains the only forecast target, not because channel migration is unimportant, but because the auxiliary task hurt the metric we must optimize.`,
    },
    {
      status: "Canonical",
      title: "Why this model is submitted",
      body: `NeuralNet / Direct produces the fixed 30 × 7 submission. The later convex ensemble improved development and recent confirmation, but ${auditEvidence} We therefore preserved the predeclared NeuralNet submission and present the ensemble as a competitive secondary forecast rather than using the audit as another tuning set.`,
    },
  ];

  grid.innerHTML = cards.map((card) => `
    <article class="nn-decision-card">
      <span class="nn-decision-status ${card.status.toLowerCase()}">${card.status}</span>
      <h3>${card.title}</h3>
      <p>${card.body}</p>
    </article>
  `).join("");
  panel.hidden = false;
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
let modelProductHistoryVisible = true;

function setModelProductHistoryVisible(visible) {
  modelProductHistoryVisible = Boolean(visible);
}

function renderProductChart(data, model, productId, strategy) {
  const hist = data.history?.[productId];
  if (!hist) return;
  const forecast = forecastsFor(data, strategy)[model.key]?.[productId] || { dates: [], quantity: [] };
  const labels = modelProductHistoryVisible
    ? [...hist.dates, ...forecast.dates]
    : [...forecast.dates];
  const datasets = [];

  if (modelProductHistoryVisible) {
    datasets.push({
      label: "History",
      data: [...hist.quantity, ...forecast.dates.map(() => null)],
      borderColor: "#0a0a0a",
      backgroundColor: "transparent",
      tension: 0.25,
      pointRadius: 0,
      borderWidth: 2,
    });
  }

  const bridge = hist.quantity[hist.quantity.length - 1];
  datasets.push({
    label: `${model.label} · ${strategyLabel(strategy)}`,
    data: modelProductHistoryVisible
      ? [...hist.dates.slice(0, -1).map(() => null), bridge, ...forecast.quantity]
      : [...forecast.quantity],
    borderColor: model.color,
    backgroundColor: "transparent",
    borderDash: strategy === "recursive" ? [7, 4] : [6, 4],
    tension: 0.25,
    pointRadius: 3,
    borderWidth: 2,
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
        x: {
          grid: { display: false },
          ticks: { maxTicksLimit: modelProductHistoryVisible ? 10 : 7 },
        },
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
      <p style="color:var(--text-dim)">Try one of: ${(data.models || []).map((model) => `<a href="${modelHref(model.slug)}" style="color:${model.color}">${model.label}</a>`).join(", ")}</p>
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

    const productSelect = document.getElementById("product-select");
    const historyToggle = document.getElementById("model-product-history-toggle");
    const firstProduct = populateProductSelector(data);
    const supportedStrategies = availableStrategies(data, model.key);
    const canonical = canonicalStrategy(data);
    const strategy = supportedStrategies.includes(canonical) ? canonical : supportedStrategies[0];
    const regime = data.config?.primary_evaluation_regime || "conditional";

    function refresh() {
      updateStrategyCopy(data, strategy);
      renderModelMethod(data, model, strategy);
      renderNeuralNetSelection(data, model, strategy);
      renderKpis(data, model, strategy, regime);
      renderFoldChart(data, model, strategy, regime);
      renderFoldTable(data, model, strategy, regime);
      renderProductChart(data, model, productSelect.value || firstProduct, strategy);
      document.getElementById("model-strategy-note").textContent = strategyLabel(strategy);
    }

    productSelect.addEventListener("change", refresh);
    historyToggle.addEventListener("change", () => {
      setModelProductHistoryVisible(historyToggle.checked);
      renderProductChart(data, model, productSelect.value || firstProduct, strategy);
    });

    renderNav(data, currentSlug());
    renderHero(model);
    refresh();
    document.getElementById("footer-note").innerHTML =
      `Comparing against the other ${(data.models || []).length - 1} models? See the <a href="${overviewHref()}" style="color:${model.color}">Overview page</a>.`;
  } catch (err) {
    document.getElementById("app").innerHTML = `
      <div class="panel">
        <div class="panel-header"><h2>Could not load results</h2></div>
        <p style="color:var(--bad)">${err.message}</p>
      </div>`;
  }
}

main();
