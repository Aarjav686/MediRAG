/* ============================================================
   MediRAG – main.js
   Premium light theme frontend logic
   ============================================================ */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────
const el = (id) => document.getElementById(id);

const heroSection       = el("hero-section");
const loaderSection     = el("loader-section");
const resultsSection    = el("results-section");

const analyzeBtn        = el("analyze-btn");
const symptomsInput     = el("symptoms-input");
const newAnalysisBtn    = el("new-analysis-btn");

const sidebarEl         = el("sidebar");
const sidebarOverlay    = el("sidebar-overlay");
const toggleSidebarBtn  = el("toggle-sidebar-btn");
const sidebarCloseBtn   = el("sidebar-close-btn");
const llmToggle         = el("llm-toggle");

const statusPill        = el("status-pill");
const statusText        = el("status-text");

const sysMode           = el("sys-mode");
const sysDocs           = el("sys-docs");
const symptomTagsCont   = el("symptom-tags-container");

const riskBanner        = el("risk-banner");
const riskIcon          = el("risk-icon");
const riskValue         = el("risk-value");
const riskDesc          = el("risk-desc");

const summaryText       = el("summary-text");
const predTabs          = el("pred-tabs");
const predPanels        = el("pred-panels");
const comparisonTbody   = el("comparison-tbody");

const quickChips        = document.querySelectorAll(".chip");

// ── Risk config ───────────────────────────────────────────────
const RISK_CONFIG = {
  Low:      { icon: "✅", desc: "Symptoms indicate a lower risk. Monitor them and consult a doctor if they persist." },
  Moderate: { icon: "🔔", desc: "Consider scheduling a medical check-up. Monitor your symptoms closely." },
  High:     { icon: "⚠️", desc: "Please consult a doctor soon. Do not delay seeking medical attention." },
  Critical: { icon: "🚨", desc: "Immediate medical attention is strongly recommended. Go to an emergency room." },
  Unknown:  { icon: "❓", desc: "" },
};

// ── Sidebar ───────────────────────────────────────────────────
function openSidebar() {
  sidebarEl.classList.add("open");
  sidebarEl.setAttribute("aria-hidden", "false");
  sidebarOverlay.classList.add("visible");
  document.body.style.overflow = "hidden";
}
function closeSidebar() {
  sidebarEl.classList.remove("open");
  sidebarEl.setAttribute("aria-hidden", "true");
  sidebarOverlay.classList.remove("visible");
  document.body.style.overflow = "";
}
toggleSidebarBtn.addEventListener("click", openSidebar);
sidebarCloseBtn.addEventListener("click", closeSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);

// ── Quick chips ───────────────────────────────────────────────
quickChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const val = chip.dataset.symptom;
    const cur = symptomsInput.value.trim();
    const terms = cur ? cur.split(",").map((s) => s.trim()).filter(Boolean) : [];
    if (!terms.includes(val)) {
      terms.push(val);
      symptomsInput.value = terms.join(", ");
    }
    symptomsInput.dispatchEvent(new Event("input"));
    symptomsInput.focus();
  });
});

// Symptom tags in sidebar (clickable)
function addSidebarTag(symptom) {
  const tag = document.createElement("button");
  tag.className = "tag";
  tag.textContent = symptom;
  tag.setAttribute("aria-label", `Add ${symptom}`);
  tag.addEventListener("click", () => {
    const cur = symptomsInput.value.trim();
    const terms = cur ? cur.split(",").map((s) => s.trim()).filter(Boolean) : [];
    if (!terms.includes(symptom)) {
      terms.push(symptom);
      symptomsInput.value = terms.join(", ");
    }
    symptomsInput.dispatchEvent(new Event("input"));
    closeSidebar();
    symptomsInput.focus();
    heroSection.scrollIntoView({ behavior: "smooth" });
  });
  return tag;
}

// ── Analyze button enable/disable ─────────────────────────────
symptomsInput.addEventListener("input", () => {
  analyzeBtn.disabled = symptomsInput.value.trim().length === 0;
});

// ── Section visibility utils ──────────────────────────────────
function showSection(section) {
  heroSection.classList.add("hidden");
  loaderSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
  section.classList.remove("hidden");
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Fetch system status ───────────────────────────────────────
async function loadSystemStatus() {
  try {
    const resp = await fetch("/api/status");
    if (!resp.ok) throw new Error("status error");
    const data = await resp.json();

    const mode = data.llm?.mode ?? "—";
    const num = data.retriever?.num_documents ?? "—";

    sysMode.textContent = mode;
    sysDocs.textContent = num;

    statusPill.classList.add("online");
    statusText.textContent = `${num} diseases indexed`;
  } catch (e) {
    statusText.textContent = "Offline";
  }
}

// ── Fetch symptoms ────────────────────────────────────────────
async function loadSymptoms() {
  try {
    const resp = await fetch("/api/symptoms");
    if (!resp.ok) throw new Error("symptoms error");
    const data = await resp.json();
    const symptoms = data.symptoms || [];

    // Clear skeleton
    symptomTagsCont.innerHTML = "";
    symptoms.slice(0, 80).forEach((s) => {
      symptomTagsCont.appendChild(addSidebarTag(s));
    });
  } catch (e) {
    symptomTagsCont.innerHTML = "<span style='font-size:.78rem;color:var(--slate-400)'>Could not load symptoms.</span>";
  }
}

// ── Render results ────────────────────────────────────────────
function renderRiskBanner(riskLevel) {
  const cfg = RISK_CONFIG[riskLevel] || RISK_CONFIG.Unknown;
  riskBanner.className = `risk-banner ${riskLevel}`;
  riskIcon.textContent = cfg.icon;
  riskValue.textContent = riskLevel;
  riskDesc.textContent = cfg.desc;
}

function renderSummary(explanation) {
  summaryText.textContent = explanation;
}

function renderPredictions(predictions) {
  predTabs.innerHTML = "";
  predPanels.innerHTML = "";

  predictions.forEach((pred, i) => {
    // Tab
    const tab = document.createElement("button");
    tab.className = "pred-tab";
    tab.role = "tab";
    tab.setAttribute("aria-selected", i === 0 ? "true" : "false");
    tab.setAttribute("aria-controls", `panel-${i}`);
    tab.id = `tab-${i}`;
    tab.textContent = `#${i + 1} ${pred.disease}`;
    tab.addEventListener("click", () => activateTab(i, predictions.length));
    predTabs.appendChild(tab);

    // Panel
    const panel = document.createElement("div");
    panel.className = `pred-panel ${i === 0 ? "active" : ""}`;
    panel.id = `panel-${i}`;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${i}`);

    const conf = pred.confidence || 0;
    const confPct = Math.round(conf * 100);
    const riskLevel = pred.risk_level || "Unknown";

    const symptoms = (pred.symptoms || []).slice(0, 12).map((s) => `
      <span class="pred-symptom-tag">${s.replace(/_/g, " ")}</span>
    `).join("");

    const precautions = (pred.precautions || []).map((p) => `<li>${cap(p)}</li>`).join("") || "<li>None listed.</li>";

    panel.innerHTML = `
      <div class="pred-main">
        <div class="pred-disease-name">${pred.disease}</div>
        <p class="pred-description">${pred.description || "No description available."}</p>
        <div class="pred-section-label">Matched Symptoms</div>
        <div class="pred-symptoms-wrap">${symptoms || "<span style='color:var(--slate-400);font-size:.82rem'>None matched.</span>"}</div>
        <div class="pred-section-label">Recommended Precautions</div>
        <ul class="pred-precautions">${precautions}</ul>
      </div>
      <div class="pred-sidebar">
        <div class="confidence-label">Confidence</div>
        <div class="confidence-value">${confPct}%</div>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar" style="width:0%" data-target="${confPct}%"></div>
        </div>
        <span class="pred-risk-badge ${riskLevel}">${riskLevel} Risk</span>
      </div>
    `;

    predPanels.appendChild(panel);
  });

  // Animate confidence bars with a slight delay
  requestAnimationFrame(() => {
    document.querySelectorAll(".confidence-bar").forEach((bar) => {
      setTimeout(() => {
        bar.style.width = bar.dataset.target;
      }, 80);
    });
  });
}

function activateTab(index, total) {
  for (let i = 0; i < total; i++) {
    const tab = el(`tab-${i}`);
    const panel = el(`panel-${i}`);
    if (tab) tab.setAttribute("aria-selected", i === index ? "true" : "false");
    if (panel) panel.className = `pred-panel ${i === index ? "active" : ""}`;
  }
}

function renderComparisonTable(predictions) {
  comparisonTbody.innerHTML = predictions.map((p) => {
    const conf = Math.round((p.confidence || 0) * 100);
    const risk = p.risk_level || "Unknown";
    const sev = (p.severity_score || 0).toFixed(1);
    const syms = (p.symptoms || []).slice(0, 5).map((s) => s.replace(/_/g, " ")).join(", ");
    const extra = (p.symptoms || []).length > 5 ? "…" : "";
    return `
      <tr>
        <td class="disease-name">${p.disease}</td>
        <td class="conf-cell">${conf}%</td>
        <td><span class="risk-badge-inline ${risk}">${risk}</span></td>
        <td>${sev}</td>
        <td>${syms}${extra}</td>
      </tr>
    `;
  }).join("");
}

// ── Analyze ───────────────────────────────────────────────────
async function analyzeSymptoms() {
  const input = symptomsInput.value.trim();
  if (!input) return;

  showSection(loaderSection);

  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symptoms: input,
        use_llm: llmToggle.checked,
      }),
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      alert(data.error || "An error occurred. Please try again.");
      showSection(heroSection);
      return;
    }

    // Render
    renderRiskBanner(data.risk_level || "Unknown");
    renderSummary(data.explanation || "No explanation available.");
    renderPredictions(data.predictions || []);
    renderComparisonTable(data.predictions || []);

    showSection(resultsSection);
  } catch (err) {
    console.error(err);
    alert("Failed to connect to the server. Please ensure the Flask server is running.");
    showSection(heroSection);
  }
}

// ── Events ────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", analyzeSymptoms);
symptomsInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    analyzeSymptoms();
  }
});

newAnalysisBtn.addEventListener("click", () => {
  symptomsInput.value = "";
  analyzeBtn.disabled = true;
  heroSection.classList.remove("hidden");
  loaderSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
  heroSection.scrollIntoView({ behavior: "smooth" });
  symptomsInput.focus();
});

// ── Helpers ───────────────────────────────────────────────────
function cap(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ── PDF Upload ────────────────────────────────────────────────
const pdfDropZone    = el("pdf-drop-zone");
const pdfFileInput   = el("pdf-file-input");
const pdfBrowseBtn   = el("pdf-browse-btn");
const pdfDropIdle    = el("pdf-drop-idle");
const pdfDropLoading = el("pdf-drop-loading");
const pdfDropSuccess = el("pdf-drop-success");
const pdfSuccessName = el("pdf-success-name");
const pdfSuccessMeta = el("pdf-success-meta");
const pdfClearBtn    = el("pdf-clear-btn");

function setPdfState(state, loadingMsg) {
  pdfDropIdle.classList.add("hidden");
  pdfDropLoading.classList.add("hidden");
  pdfDropSuccess.classList.add("hidden");
  if (state === "idle")    pdfDropIdle.classList.remove("hidden");
  if (state === "loading") {
    pdfDropLoading.classList.remove("hidden");
    const msgEl = pdfDropLoading.querySelector(".pdf-loading-msg");
    if (msgEl && loadingMsg) msgEl.textContent = loadingMsg;
  }
  if (state === "success") pdfDropSuccess.classList.remove("hidden");
}

async function uploadPdf(file) {
  if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
    alert("Please select a valid PDF file.");
    return;
  }
  setPdfState("loading", "Extracting text from PDF…");

  // After 3s with no response, hint that OCR may be running
  const ocrHintTimer = setTimeout(() => {
    setPdfState("loading", "Running OCR on scanned pages… this may take up to 60s");
  }, 3000);

  const formData = new FormData();
  formData.append("pdf", file);

  try {
    const resp = await fetch("/api/upload-pdf", { method: "POST", body: formData });
    clearTimeout(ocrHintTimer);
    const data = await resp.json();

    if (!resp.ok || data.error) {
      alert(data.error || "Failed to read PDF.");
      setPdfState("idle");
      return;
    }

    // Populate the textarea with extracted text
    symptomsInput.value = data.text;
    symptomsInput.dispatchEvent(new Event("input"));

    // Show success state with method badge
    const isOCR = data.method === "ocr";
    pdfSuccessName.textContent = data.filename;
    pdfSuccessMeta.textContent =
      `${data.pages} page${data.pages !== 1 ? "s" : ""} · ` +
      (isOCR ? "🔍 OCR extracted" : "⚡ text extracted");
    setPdfState("success");
  } catch (err) {
    clearTimeout(ocrHintTimer);
    console.error(err);
    alert("Could not connect to server for PDF parsing.");
    setPdfState("idle");
  }
}

// Browse click
pdfBrowseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  pdfFileInput.click();
});

// Zone click (but not the browse btn)
pdfDropZone.addEventListener("click", (e) => {
  if (e.target !== pdfBrowseBtn) pdfFileInput.click();
});

// Keyboard accessibility
pdfDropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pdfFileInput.click(); }
});

// File input change
pdfFileInput.addEventListener("change", () => {
  if (pdfFileInput.files.length) uploadPdf(pdfFileInput.files[0]);
  pdfFileInput.value = ""; // reset so same file can be re-uploaded
});

// Drag & drop
pdfDropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  pdfDropZone.classList.add("drag-over");
});
pdfDropZone.addEventListener("dragleave", () => pdfDropZone.classList.remove("drag-over"));
pdfDropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  pdfDropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadPdf(file);
});

// Clear
pdfClearBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  symptomsInput.value = "";
  symptomsInput.dispatchEvent(new Event("input"));
  setPdfState("idle");
});

// ── Init ──────────────────────────────────────────────────────
(async function init() {
  await Promise.all([loadSystemStatus(), loadSymptoms()]);
})();
