const CHART_PALETTE = {
  sky: "#3aa6e0",
  skyLight: "#7cd3f2",
  navy: "#0c3b5e",
  leaf: "#6fd6a8",
  leafDeep: "#38b08a",
  amber: "#f1c574",
  coral: "#ef7a6d",
  grid: "rgba(12, 59, 94, 0.08)",
  ink: "#4c6b80",
};

const BAR_COLORS = [CHART_PALETTE.sky, CHART_PALETTE.leaf, CHART_PALETTE.amber, CHART_PALETTE.coral, CHART_PALETTE.skyLight, CHART_PALETTE.navy];

const chartInstances = {};

function destroyChart(key) {
  if (chartInstances[key]) {
    chartInstances[key].destroy();
    delete chartInstances[key];
  }
}

function baseOptions(extra = {}) {
  return Object.assign({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: CHART_PALETTE.ink, font: { family: "Inter" } } },
      y: { grid: { color: CHART_PALETTE.grid }, ticks: { color: CHART_PALETTE.ink, font: { family: "Inter" } } },
    },
  }, extra);
}

function renderDoughnut(canvasId, chart) {
  const ctx = document.getElementById(canvasId);
  destroyChart(canvasId);
  chartInstances[canvasId] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: chart.labels,
      datasets: [{
        data: chart.data,
        backgroundColor: [CHART_PALETTE.coral, CHART_PALETTE.leaf],
        borderColor: "#ffffff",
        borderWidth: 3,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: { position: "bottom", labels: { color: CHART_PALETTE.ink, font: { family: "Inter" } } },
      },
    },
  });
}

function renderBar(canvasId, chart) {
  const ctx = document.getElementById(canvasId);
  destroyChart(canvasId);
  chartInstances[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chart.labels,
      datasets: [{
        label: "Attrition rate (%)",
        data: chart.data,
        backgroundColor: chart.labels.map((_, i) => BAR_COLORS[i % BAR_COLORS.length]),
        borderRadius: 8,
        maxBarThickness: 48,
      }],
    },
    options: baseOptions({
      scales: {
        x: { grid: { display: false }, ticks: { color: CHART_PALETTE.ink, font: { family: "Inter" } } },
        y: {
          beginAtZero: true,
          grid: { color: CHART_PALETTE.grid },
          ticks: {
            color: CHART_PALETTE.ink,
            font: { family: "Inter" },
            callback: (v) => v + "%",
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed.y}% attrition` } },
      },
    }),
  });
}

function renderLine(canvasId, chart) {
  const ctx = document.getElementById(canvasId);
  destroyChart(canvasId);
  chartInstances[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels: chart.labels,
      datasets: [{
        label: "Attrition rate (%)",
        data: chart.data,
        borderColor: CHART_PALETTE.sky,
        backgroundColor: "rgba(58, 166, 224, 0.18)",
        fill: true,
        tension: 0.35,
        pointBackgroundColor: "#ffffff",
        pointBorderColor: CHART_PALETTE.sky,
        pointRadius: 5,
        pointBorderWidth: 2,
      }],
    },
    options: baseOptions({
      scales: {
        x: { grid: { display: false }, ticks: { color: CHART_PALETTE.ink, font: { family: "Inter" } } },
        y: {
          beginAtZero: true,
          grid: { color: CHART_PALETTE.grid },
          ticks: {
            color: CHART_PALETTE.ink,
            font: { family: "Inter" },
            callback: (v) => v + "%",
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed.y}% attrition` } },
      },
    }),
  });
}

const CHART_RENDERERS = {
  doughnut: renderDoughnut,
  bar: renderBar,
  line: renderLine,
};

const CARD_KEYS = ["overall", "department", "overtime", "age", "job_satisfaction", "tenure"];

function renderCharts(payload) {
  document.getElementById("stat-total").textContent = payload.summary.total_employees.toLocaleString();
  document.getElementById("stat-left").textContent = payload.summary.total_attrition.toLocaleString();
  document.getElementById("stat-rate").textContent = `${payload.summary.attrition_rate}%`;

  CARD_KEYS.forEach((key) => {
    const card = document.getElementById(`card-${key}`);
    const chart = payload[key];
    if (!card) return;

    if (!chart) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    const renderer = CHART_RENDERERS[chart.type];
    if (renderer) renderer(`chart-${key}`, chart);
  });

  const warningsEl = document.getElementById("charts-warnings");
  warningsEl.textContent = (payload.warnings || []).join("\n");

  document.getElementById("charts-empty").hidden = true;
  document.getElementById("charts-grid").hidden = false;
}

function loadCharts() {
  const emptyText = document.getElementById("charts-empty-text");

  fetch("/api/charts")
    .then(async (response) => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load chart data.");
      return data;
    })
    .then((data) => {
      renderCharts(data);
    })
    .catch((err) => {
      if (emptyText) emptyText.textContent = err.message;
    });
}

document.addEventListener("DOMContentLoaded", () => {
  if (!document.getElementById("charts-grid")) return; // not on the home page
  loadCharts();
});
