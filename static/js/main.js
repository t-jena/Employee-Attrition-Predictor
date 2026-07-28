// Retain — assessment page interactivity

const RING_CIRCUMFERENCE = 2 * Math.PI * 100; // r=100 in the SVG

function setGauge(gaugeEl, percentage, tier) {
  const progress = gaugeEl.querySelector(".gauge-orb__progress");
  const offset = RING_CIRCUMFERENCE * (1 - percentage / 100);
  progress.style.strokeDasharray = RING_CIRCUMFERENCE;
  progress.style.strokeDashoffset = offset;
  gaugeEl.dataset.tier = tier;
}

function tierMessage(tier) {
  switch (tier) {
    case "low":
      return "Low risk. No immediate action flagged.";
    case "moderate":
      return "Moderate risk. Worth a check-in on engagement and workload.";
    case "high":
      return "High risk. Consider a retention conversation soon.";
    default:
      return "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("assessment-form");
  if (!form) return; // not on the assessment page

  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("form-status");
  const resultEmpty = document.getElementById("result-empty");
  const resultFilled = document.getElementById("result-filled");
  const resultGauge = document.getElementById("result-gauge");
  const resultPct = document.getElementById("result-pct");
  const resultTier = document.getElementById("result-tier");
  const resultNote = document.getElementById("result-note");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    statusEl.textContent = "";
    statusEl.className = "form-status";

    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
      payload[key] = value;
    }
    // checkbox isn't included by FormData when unchecked
    payload.OverTime = form.elements.OverTime.checked ? "Yes" : "No";

    submitBtn.disabled = true;
    submitBtn.textContent = "Assessing...";
    statusEl.textContent = "Scoring profile against the model...";
    statusEl.classList.add("is-loading");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Something went wrong while scoring this profile.");
      }

      statusEl.textContent = "";
      statusEl.classList.remove("is-loading");

      resultEmpty.hidden = true;
      resultFilled.hidden = false;

      setGauge(resultGauge, data.percentage, data.risk_level);
      resultPct.textContent = `${data.percentage}%`;
      resultTier.textContent = `${data.risk_level} risk`;
      resultNote.textContent = tierMessage(data.risk_level);
    } catch (err) {
      resultEmpty.hidden = false;
      resultFilled.hidden = true;
      statusEl.textContent = err.message;
      statusEl.classList.remove("is-loading");
      statusEl.classList.add("is-error");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Assess risk";
    }
  });

  form.addEventListener("reset", () => {
    statusEl.textContent = "";
    statusEl.className = "form-status";
    resultEmpty.hidden = false;
    resultFilled.hidden = true;
  });
});
