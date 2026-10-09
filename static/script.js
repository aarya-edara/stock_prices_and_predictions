

const searchInput = document.getElementById("stock-search");
const suggestionsBox = document.getElementById("suggestions");
const quickPicksBox = document.getElementById("quick-picks");
const statusEl = document.getElementById("status");
const dashboardEl = document.getElementById("dashboard");

const QUICK_PICKS = [
  { symbol: "KO", name: "Coca-Cola" },
  { symbol: "AAPL", name: "Apple" },
  { symbol: "MSFT", name: "Microsoft" },
  { symbol: "NVDA", name: "NVIDIA" },
];

let searchDebounce = null;
let activeRequestId = 0;

document.addEventListener("DOMContentLoaded", () => {
  renderQuickPicks();
});


function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("is-error", isError);
}

function money(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return `$${num.toFixed(2)}`;
}

function volume(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return num.toLocaleString("en-US");
}

function themeColors() {
  const styles = getComputedStyle(document.documentElement);
  return {
    ink: styles.getPropertyValue("--ink").trim() || "#ffffff",
    gain: styles.getPropertyValue("--gain").trim() || "#00ffe4",
    loss: styles.getPropertyValue("--loss").trim() || "#8C2F26",
    accent: styles.getPropertyValue("--accent").trim() || "#ffc1f0",
  };
}

function chartOptions(xTitle = "Date", yTitle = "Price (USD)") {
  const { ink } = themeColors();
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: ink } } },
    scales: {
      x: {
        title: { display: true, text: xTitle, color: ink, font: { family: "IBM Plex Mono", size: 12 } },
        ticks: { color: ink, maxTicksLimit: 10 },
        grid: { display: false },
        border: { color: ink },
      },
      y: {
        title: { display: true, text: yTitle, color: ink, font: { family: "IBM Plex Mono", size: 12 } },
        ticks: { color: ink, callback: (value) => `$${value}` },
        grid: { color: "rgba(150,150,150,0.15)" },
        border: { color: ink },
      },
    },
  };
}


function renderQuickPicks() {
  quickPicksBox.innerHTML = "";
  QUICK_PICKS.forEach((pick) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = pick.name;
    chip.addEventListener("click", () => {
      searchInput.value = pick.name;
      hideSuggestions();
      loadStock(pick.symbol);
    });
    quickPicksBox.appendChild(chip);
  });
}

function markActiveChip(symbol) {
  quickPicksBox.querySelectorAll(".chip").forEach((chip, i) => {
    chip.classList.toggle("is-active", QUICK_PICKS[i].symbol === symbol);
  });
}



searchInput.addEventListener("input", () => {
  const query = searchInput.value.trim();
  clearTimeout(searchDebounce);

  if (!query) {
    hideSuggestions();
    return;
  }

  searchDebounce = setTimeout(() => runSearch(query), 250);
});

searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    clearTimeout(searchDebounce);
    const query = searchInput.value.trim();
    if (query) loadStock(query);
  }
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".search-container")) {
    hideSuggestions();
  }
});

async function runSearch(query) {
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const results = await response.json();
    renderSuggestions(results);
  } catch (err) {
    hideSuggestions();
  }
}

function renderSuggestions(results) {
  if (!results || results.length === 0) {
    hideSuggestions();
    return;
  }

  suggestionsBox.innerHTML = "";
  results.forEach((company) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion-item";
    button.innerHTML = `
      <span class="suggestion-name">${company.name}</span>
      <span class="suggestion-symbol">${company.symbol}</span>
    `;
    button.addEventListener("click", () => {
      searchInput.value = company.name;
      hideSuggestions();
      loadStock(company.symbol);
    });
    item.appendChild(button);
    suggestionsBox.appendChild(item);
  });

  suggestionsBox.hidden = false;
}

function hideSuggestions() {
  suggestionsBox.hidden = true;
  suggestionsBox.innerHTML = "";
}



async function loadStock(symbolOrQuery) {
  const requestId = ++activeRequestId;
  setStatus(`Loading ${symbolOrQuery}…`);
  dashboardEl.hidden = true;

  try {
    const response = await fetch(`/api/stock/${encodeURIComponent(symbolOrQuery)}`);
    const data = await response.json();


    if (requestId !== activeRequestId) return;

    if (!response.ok) {
      setStatus(data.error || `Could not load ${symbolOrQuery}.`, true);
      return;
    }

    setStatus(null);
    renderCompany(data);
  } catch (err) {
    if (requestId !== activeRequestId) return;
    setStatus(`Could not reach the server: ${err.message}`, true);
  }
}



let charts = {};

function destroyCharts() {
  Object.values(charts).forEach((chart) => chart && chart.destroy());
  charts = {};
}

function renderCompany(company) {
  dashboardEl.hidden = false;
  markActiveChip(company.symbol);

  document.title = `${company.companyName} (${company.symbol}) | Stock Prices, Predictions and Changes`;


  document.getElementById("company-name").textContent = company.companyName;
  document.getElementById("company-symbol").textContent = company.symbol;
  document.getElementById("price").textContent = money(company.today.close);
  document.getElementById("stat-date").textContent = company.asOf;

  const change = company.today.change;
  const changePercent = company.today.changePercent;
  const changeEl = document.getElementById("change");
  changeEl.textContent = `${change >= 0 ? "▲" : "▼"} ${money(Math.abs(change))} (${Math.abs(changePercent).toFixed(2)}%)`;
  changeEl.classList.toggle("positive", change >= 0);
  changeEl.classList.toggle("negative", change < 0);

  // statistics for today
  document.getElementById("stat-open").textContent = money(company.today.open);
  document.getElementById("stat-prev-close").textContent = money(company.today.previousClose);
  document.getElementById("stat-range").textContent = `${money(company.today.low)} – ${money(company.today.high)}`;
  document.getElementById("stat-volume").textContent = volume(company.today.volume);

  // next-day prediction
  document.getElementById("outlook-next-close").textContent = money(company.nextDayPrediction.predictedClose);
  const predictedChange = company.nextDayPrediction.predictedChangePercent;
  const nextChangeEl = document.getElementById("outlook-next-change");
  nextChangeEl.textContent = `${predictedChange >= 0 ? "▲" : "▼"} ${Math.abs(predictedChange).toFixed(3)}%`;
  nextChangeEl.classList.toggle("positive", predictedChange >= 0);
  nextChangeEl.classList.toggle("negative", predictedChange < 0);

  // horizons
  renderHorizon("horizon-5day", "horizon-5day-range", company.horizonSummary && company.horizonSummary.fiveDay);
  renderHorizon("horizon-1month", "horizon-1month-range", company.horizonSummary && company.horizonSummary.oneMonth);
  renderHorizon("horizon-3month", "horizon-3month-range", company.horizonSummary && company.horizonSummary.threeMonth);

  // forecast information
  document.getElementById("simulation-count").textContent = company.forecastInfo ? company.forecastInfo.simulations : "—";
  document.getElementById("forecast-days").textContent = company.forecastInfo ? company.forecastInfo.forecastDays : "—";
  document.getElementById("forecast-generated-date").textContent = company.forecastInfo ? company.forecastInfo.generatedFromDate : "—";

  // performance of the model
  const perf = company.modelPerformance;
  document.getElementById("perf-mae").textContent = perf.mae.toFixed(5);
  document.getElementById("perf-baseline-mae").textContent = perf.baselineMae.toFixed(5);
  document.getElementById("perf-r2").textContent = perf.r2.toFixed(4);
  document.getElementById("perf-beats-baseline").textContent = perf.beatsBaselineOnMae ? "Yes (MAE)" : "No";

  destroyCharts();
  renderHistoryChart(company);
  renderAvpCharts(company);
  renderForecastChart(company);
  renderForecastTable(company);
  renderImportanceTable(company);

  dashboardEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHorizon(priceId, rangeId, horizon) {
  const priceEl = document.getElementById(priceId);
  const rangeEl = document.getElementById(rangeId);
  if (!horizon) {
    priceEl.textContent = "—";
    rangeEl.textContent = "—";
    return;
  }

  const arrow = horizon.medianChangePercent >= 0 ? "▲" : "▼";
  priceEl.textContent = `${money(horizon.medianPrice)} ${arrow} ${Math.abs(horizon.medianChangePercent).toFixed(2)}%`;
  rangeEl.textContent = `90% range: ${money(horizon.lowerPrice)} – ${money(horizon.upperPrice)}`;
  priceEl.classList.toggle("positive", horizon.medianChangePercent >= 0);
  priceEl.classList.toggle("negative", horizon.medianChangePercent < 0);
}

function renderHistoryChart(company) {
  const { ink, accent } = themeColors();
  charts.history = new Chart(document.getElementById("history-chart"), {
    type: "line",
    data: {
      labels: company.closePriceHistory.map((row) => row.date),
      datasets: [
        {
          label: `${company.companyName} Close`,
          data: company.closePriceHistory.map((row) => row.close),
          borderColor: accent,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: chartOptions(),
  });
}

function renderAvpCharts(company) {
  const { ink, gain } = themeColors();
  const rows = company.actualVsPredicted;
  const recent = rows.slice(-100);

  const makeDatasets = (data) => [
    { label: "Actual", data: data.map((r) => r.actual), borderColor: ink, borderWidth: 1.5, pointRadius: 0 },
    { label: "Predicted", data: data.map((r) => r.predicted), borderColor: gain, borderWidth: 1.5, pointRadius: 0 },
  ];

  charts.avp = new Chart(document.getElementById("avp-chart"), {
    type: "line",
    data: { labels: rows.map((r) => r.date), datasets: makeDatasets(rows) },
    options: chartOptions(),
  });

  charts.avpRecent = new Chart(document.getElementById("avp-recent-chart"), {
    type: "line",
    data: { labels: recent.map((r) => r.date), datasets: makeDatasets(recent) },
    options: chartOptions(),
  });
}

function renderForecastChart(company) {
  const canvas = document.getElementById("forecast-chart");
  const forecast = company.forecast3Month;
  if (!forecast || forecast.length === 0) {
    canvas.parentElement.hidden = true;
    return;
  }
  canvas.parentElement.hidden = false;

  const { ink, gain, accent } = themeColors();
  const recentHistory = company.closePriceHistory.slice(-40);

  const labels = [...recentHistory.map((r) => r.date), ...forecast.map((r) => r.date)];
  const actualSeries = [...recentHistory.map((r) => r.close), ...forecast.map(() => null)];

  const blankHistory = recentHistory.slice(0, -1).map(() => null);
  const lastClose = recentHistory[recentHistory.length - 1].close;

  const medianSeries = [...blankHistory, lastClose, ...forecast.map((r) => r.median)];
  const lowSeries = [...blankHistory, lastClose, ...forecast.map((r) => r.lower90)];
  const highSeries = [...blankHistory, lastClose, ...forecast.map((r) => r.upper90)];

  charts.forecast = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Actual Close", data: actualSeries, borderColor: ink, borderWidth: 2, pointRadius: 0 },
        { label: "Median Forecast", data: medianSeries, borderColor: gain, borderWidth: 2, pointRadius: 0 },
        { label: "Lower 90%", data: lowSeries, borderColor: accent, borderWidth: 1, borderDash: [5, 5], pointRadius: 0 },
        { label: "Upper 90%", data: highSeries, borderColor: accent, borderWidth: 1, borderDash: [5, 5], pointRadius: 0 },
      ],
    },
    options: chartOptions(),
  });
}

function renderForecastTable(company) {
  const body = document.getElementById("forecast-table-body");
  body.innerHTML = "";
  (company.forecast3Month || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.date}</td>
      <td>${money(row.lower90)}</td>
      <td>${money(row.median)}</td>
      <td>${money(row.upper90)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderImportanceTable(company) {
  const body = document.getElementById("importance-table-body");
  body.innerHTML = "";
  company.featureImportance.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.feature}</td><td>${Number(row.importance).toFixed(5)}</td>`;
    body.appendChild(tr);
  });
}
