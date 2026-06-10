"use strict";

const STORAGE_KEY = "resource_triage_state_v2";
const STORAGE_KEY_LEGACY = "resource_triage_state_v1";

const ANNOTATION_HEADERS = [
  "paper",
  "dataset_url",
  "no_dataset_url",
  "code_url",
  "resource_license",
  "resource_format",
  "resource_access",
  "resource_notes",
];

const state = {
  papers: [],
  annotations: {},
  currentIndex: 0,
  queueSearch: "",
  filterPendingOnly: true,
  filterNoCandOnly: false,
  savedCurrentPaperId: null,
  noDatasetUrl: false,
  selectedDatasetUrl: "",
  draft: {
    code_url: "",
    resource_license: "",
    resource_format: "",
    resource_access: "",
    resource_notes: "",
  },
};

const els = {
  loadQueueBtn: document.getElementById("loadQueueBtn"),
  importAnnotationsBtn: document.getElementById("importAnnotationsBtn"),
  exportAnnotationsBtn: document.getElementById("exportAnnotationsBtn"),
  clearStateBtn: document.getElementById("clearStateBtn"),
  queueSearch: document.getElementById("queueSearch"),
  filterPendingOnly: document.getElementById("filterPendingOnly"),
  filterNoCandOnly: document.getElementById("filterNoCandOnly"),
  queueMeta: document.getElementById("queueMeta"),
  queueList: document.getElementById("queueList"),
  paperEmpty: document.getElementById("paperEmpty"),
  paperView: document.getElementById("paperView"),
  paperTitle: document.getElementById("paperTitle"),
  paperMeta: document.getElementById("paperMeta"),
  linkAnthology: document.getElementById("linkAnthology"),
  linkPdf: document.getElementById("linkPdf"),
  openDatasetBtn: document.getElementById("openDatasetBtn"),
  paperAbstract: document.getElementById("paperAbstract"),
  noDatasetUrlRadio: document.getElementById("noDatasetUrlRadio"),
  candidatesList: document.getElementById("candidatesList"),
  manualDatasetUrl: document.getElementById("manualDatasetUrl"),
  codeUrl: document.getElementById("codeUrl"),
  resourceLicense: document.getElementById("resourceLicense"),
  resourceFormat: document.getElementById("resourceFormat"),
  resourceAccess: document.getElementById("resourceAccess"),
  resourceNotes: document.getElementById("resourceNotes"),
  saveBtn: document.getElementById("saveBtn"),
  saveNextBtn: document.getElementById("saveNextBtn"),
  noResourceBtn: document.getElementById("noResourceBtn"),
  progressStats: document.getElementById("progressStats"),
  lastSavedPreview: document.getElementById("lastSavedPreview"),
  queueFileInput: document.getElementById("queueFileInput"),
  annotationsFileInput: document.getElementById("annotationsFileInput"),
};

function escapeCsv(value) {
  const s = String(value ?? "");
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function toCsv(rows, headers) {
  const headerRow = headers.join(",");
  const dataRows = rows.map((row) =>
    headers.map((h) => escapeCsv(row[h])).join(","),
  );
  return [headerRow, ...dataRows].join("\n");
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (!lines.length) {
    return [];
  }
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    for (let i = 0; i < headers.length; i += 1) {
      row[headers[i]] = values[i] || "";
    }
    return row;
  });
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function persistPayload() {
  const paper = currentPaper();
  return {
    annotations: state.annotations,
    savedCurrentPaperId: paper ? paper.anthology_id : state.savedCurrentPaperId,
    filterPendingOnly: state.filterPendingOnly,
    filterNoCandOnly: state.filterNoCandOnly,
  };
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persistPayload()));
    localStorage.removeItem(STORAGE_KEY_LEGACY);
  } catch (err) {
    if (err.name === "QuotaExceededError") {
      console.warn("localStorage plein", err);
      alert(
        "Sauvegarde locale impossible (quota dépassé). Exportez resource_annotations.csv régulièrement.",
      );
      return;
    }
    throw err;
  }
}

function applyPersistedSettings(parsed) {
  if (parsed.annotations && typeof parsed.annotations === "object") {
    state.annotations = parsed.annotations;
  }
  if (parsed.savedCurrentPaperId) {
    state.savedCurrentPaperId = parsed.savedCurrentPaperId;
  }
  state.filterPendingOnly = parsed.filterPendingOnly !== false;
  state.filterNoCandOnly = Boolean(parsed.filterNoCandOnly);
  els.filterPendingOnly.checked = state.filterPendingOnly;
  els.filterNoCandOnly.checked = state.filterNoCandOnly;
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      applyPersistedSettings(JSON.parse(raw));
      return;
    }
    const legacy = localStorage.getItem(STORAGE_KEY_LEGACY);
    if (!legacy) {
      return;
    }
    const parsed = JSON.parse(legacy);
    if (parsed.annotations && typeof parsed.annotations === "object") {
      state.annotations = parsed.annotations;
    }
    const legacyPapers = parsed.papers;
    if (Array.isArray(legacyPapers) && legacyPapers.length && parsed.currentIndex != null) {
      const idx = Math.min(parsed.currentIndex, legacyPapers.length - 1);
      state.savedCurrentPaperId = legacyPapers[idx]?.anthology_id || null;
    }
    state.filterPendingOnly = parsed.filterPendingOnly !== false;
    state.filterNoCandOnly = Boolean(parsed.filterNoCandOnly);
    els.filterPendingOnly.checked = state.filterPendingOnly;
    els.filterNoCandOnly.checked = state.filterNoCandOnly;
    localStorage.removeItem(STORAGE_KEY_LEGACY);
    saveState();
  } catch (err) {
    console.warn("loadState failed", err);
  }
}

function restoreCurrentIndexFromSavedId() {
  if (!state.savedCurrentPaperId || !state.papers.length) {
    return;
  }
  const visible = visibleIndices();
  for (let visIdx = 0; visIdx < visible.length; visIdx += 1) {
    const paper = state.papers[visible[visIdx]];
    if (paper.anthology_id === state.savedCurrentPaperId) {
      state.currentIndex = visIdx;
      return;
    }
  }
}

function isReviewed(paperId) {
  return Object.prototype.hasOwnProperty.call(state.annotations, paperId);
}

function currentPaper() {
  const visible = visibleIndices();
  if (!visible.length) {
    return null;
  }
  const idx = Math.min(state.currentIndex, visible.length - 1);
  state.currentIndex = idx;
  return state.papers[visible[idx]];
}

function visibleIndices() {
  const q = state.queueSearch.trim().toLowerCase();
  return state.papers
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => {
      if (state.filterPendingOnly && isReviewed(p.anthology_id)) {
        return false;
      }
      if (state.filterNoCandOnly && p.n_cand > 0) {
        return false;
      }
      if (!q) {
        return true;
      }
      return (
        p.anthology_id.toLowerCase().includes(q) ||
        (p.title || "").toLowerCase().includes(q)
      );
    })
    .map(({ i }) => i);
}

function getEffectiveDatasetUrl() {
  if (state.noDatasetUrl) {
    return "";
  }
  const manual = els.manualDatasetUrl.value.trim();
  if (manual) {
    return manual;
  }
  return state.selectedDatasetUrl.trim();
}

function setNoDatasetUrl(enabled) {
  state.noDatasetUrl = enabled;
  if (els.noDatasetUrlRadio) {
    els.noDatasetUrlRadio.checked = enabled;
  }
  if (enabled) {
    state.selectedDatasetUrl = "";
    els.manualDatasetUrl.value = "";
    document
      .querySelectorAll('input[name="datasetPick"]:not(#noDatasetUrlRadio)')
      .forEach((r) => {
        r.checked = false;
      });
    els.manualDatasetUrl.disabled = true;
  } else {
    els.manualDatasetUrl.disabled = false;
  }
  updateOpenDatasetBtn();
  updateDatasetSectionUi();
}

function updateDatasetSectionUi() {
  els.candidatesList.classList.toggle("candidates-muted", state.noDatasetUrl);
}

function loadDraftFromAnnotation(paperId) {
  const ann = state.annotations[paperId];
  state.selectedDatasetUrl = "";
  state.noDatasetUrl = false;
  els.manualDatasetUrl.value = "";
  els.manualDatasetUrl.disabled = false;
  state.draft = {
    code_url: "",
    resource_license: "",
    resource_format: "",
    resource_access: "",
    resource_notes: "",
  };
  if (!ann) {
    return;
  }
  const ds = (ann.dataset_url || "").trim();
  const flaggedNone =
    !ds &&
    (ann.no_dataset_url === true ||
      ann.no_dataset_url === "true" ||
      String(ann.resource_notes || "").includes("NO_RESOURCE"));
  if (flaggedNone) {
    setNoDatasetUrl(true);
  }
  if (ds && !state.noDatasetUrl) {
    const paper = state.papers.find((p) => p.anthology_id === paperId);
    const inCandidates =
      paper &&
      paper.candidates.some((c) => c.url === ds);
    if (inCandidates) {
      state.selectedDatasetUrl = ds;
    } else {
      els.manualDatasetUrl.value = ds;
    }
  }
  state.draft.code_url = ann.code_url || "";
  state.draft.resource_license = ann.resource_license || "";
  state.draft.resource_format = ann.resource_format || "";
  state.draft.resource_access = ann.resource_access || "";
  state.draft.resource_notes = ann.resource_notes || "";
}

function syncFormFromDraft() {
  els.codeUrl.value = state.draft.code_url;
  els.resourceLicense.value = state.draft.resource_license;
  els.resourceFormat.value = state.draft.resource_format;
  els.resourceAccess.value = state.draft.resource_access;
  els.resourceNotes.value = state.draft.resource_notes;
  updateOpenDatasetBtn();
}

function readDraftFromForm() {
  state.draft.code_url = els.codeUrl.value.trim();
  state.draft.resource_license = els.resourceLicense.value.trim();
  state.draft.resource_format = els.resourceFormat.value.trim();
  state.draft.resource_access = els.resourceAccess.value.trim();
  state.draft.resource_notes = els.resourceNotes.value.trim();
}

function updateOpenDatasetBtn() {
  const url = getEffectiveDatasetUrl();
  els.openDatasetBtn.disabled = !url;
}

function buildAnnotation(paperId, datasetUrl, noDataset) {
  readDraftFromForm();
  return {
    paper: paperId,
    dataset_url: noDataset ? "" : datasetUrl || "",
    no_dataset_url: noDataset ? "true" : "false",
    code_url: state.draft.code_url,
    resource_license: state.draft.resource_license,
    resource_format: state.draft.resource_format,
    resource_access: state.draft.resource_access,
    resource_notes: state.draft.resource_notes,
    updated_at: new Date().toISOString(),
  };
}

function saveCurrent(markNoResource = false) {
  const paper = currentPaper();
  if (!paper) {
    return false;
  }
  const noDataset = markNoResource || state.noDatasetUrl;
  const datasetUrl = noDataset ? "" : getEffectiveDatasetUrl();
  if (!noDataset && !datasetUrl) {
    alert(
      "Choisissez une URL, saisissez une URL manuelle, ou cochez « Aucune URL de ressource ».",
    );
    return false;
  }
  const ann = buildAnnotation(paper.anthology_id, datasetUrl, noDataset);
  if (noDataset && !ann.resource_notes.includes("NO_RESOURCE")) {
    ann.resource_notes = [ann.resource_notes, "NO_RESOURCE"]
      .filter(Boolean)
      .join(" | ");
  }
  state.annotations[paper.anthology_id] = ann;
  els.lastSavedPreview.textContent = JSON.stringify(ann, null, 2);
  saveState();
  renderAll();
  return true;
}

function goNext() {
  const visible = visibleIndices();
  if (!visible.length) {
    return;
  }
  if (state.currentIndex < visible.length - 1) {
    state.currentIndex += 1;
  }
  saveState();
  renderPaper();
  renderQueue();
}

function goPrev() {
  if (state.currentIndex > 0) {
    state.currentIndex -= 1;
  }
  saveState();
  renderPaper();
  renderQueue();
}

function openDatasetUrl() {
  const url = getEffectiveDatasetUrl();
  if (url) {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}

function renderQueue() {
  const visible = visibleIndices();
  const reviewed = state.papers.filter((p) => isReviewed(p.anthology_id)).length;
  els.queueMeta.textContent = `${visible.length} affichés · ${reviewed}/${state.papers.length} revus`;

  els.queueList.innerHTML = "";
  visible.forEach((paperIdx, visIdx) => {
    const p = state.papers[paperIdx];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "queue-item";
    if (visIdx === state.currentIndex) {
      btn.classList.add("active");
    }
    if (isReviewed(p.anthology_id)) {
      btn.classList.add("done");
    }
    btn.innerHTML = `<div class="qid">${p.anthology_id} · ${p.n_cand} cand · ${p.max_score}</div><div class="qtitle">${escapeHtml(p.title || "")}</div>`;
    btn.addEventListener("click", () => {
      state.currentIndex = visIdx;
      saveState();
      renderPaper();
      renderQueue();
    });
    els.queueList.appendChild(btn);
  });
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function stopLinkActivatingLabel(ev) {
  ev.stopPropagation();
}

function appendCandidateRow(container, candidate, checked) {
  const label = document.createElement("label");
  label.className = "candidate";

  const radio = document.createElement("input");
  radio.type = "radio";
  radio.name = "datasetPick";
  radio.checked = checked;

  const body = document.createElement("div");
  body.className = "candidate-body";

  const link = document.createElement("a");
  link.className = "candidate-url";
  link.href = candidate.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = candidate.url;
  link.title = candidate.url;
  link.addEventListener("click", stopLinkActivatingLabel);
  link.addEventListener("mousedown", stopLinkActivatingLabel);

  const meta = document.createElement("div");
  meta.className = "candidate-meta";
  meta.textContent = `score ${candidate.score} · ${candidate.host_type} · ${candidate.zone}`;

  const context = document.createElement("div");
  context.className = "candidate-context";
  context.textContent = candidate.context;

  body.append(link, meta, context);
  label.append(radio, body);

  radio.addEventListener("change", () => {
    setNoDatasetUrl(false);
    state.selectedDatasetUrl = candidate.url;
    els.manualDatasetUrl.value = "";
    radio.checked = true;
    updateOpenDatasetBtn();
  });

  container.appendChild(label);
}

function renderPaper() {
  const paper = currentPaper();
  if (!paper) {
    els.paperEmpty.classList.remove("hidden");
    els.paperView.classList.add("hidden");
    return;
  }
  els.paperEmpty.classList.add("hidden");
  els.paperView.classList.remove("hidden");

  loadDraftFromAnnotation(paper.anthology_id);

  els.paperTitle.textContent = paper.title || paper.anthology_id;
  els.paperMeta.textContent = [
    paper.anthology_id,
    paper.year ? `(${paper.year})` : "",
    paper.mother_task || paper.task || "",
    `${paper.n_cand} candidat(s) · score max ${paper.max_score}`,
  ]
    .filter(Boolean)
    .join(" · ");

  els.linkAnthology.href = paper.anthology_url;
  els.linkPdf.href = paper.pdf_url || "#";
  els.linkPdf.style.display = paper.pdf_url ? "" : "none";
  els.paperAbstract.textContent =
    paper.abstract && paper.abstract.trim()
      ? paper.abstract
      : "(pas d'abstract)";

  if (els.noDatasetUrlRadio) {
    els.noDatasetUrlRadio.checked = state.noDatasetUrl;
  }
  els.manualDatasetUrl.disabled = state.noDatasetUrl;

  els.candidatesList.innerHTML = "";
  if (!paper.candidates.length) {
    const p = document.createElement("p");
    p.className = "meta-line";
    p.textContent =
      "Aucun candidat extrait — choisissez « Aucune URL » ou saisissez une URL manuelle.";
    els.candidatesList.appendChild(p);
  } else {
    paper.candidates.forEach((c) => {
      appendCandidateRow(
        els.candidatesList,
        c,
        !state.noDatasetUrl && state.selectedDatasetUrl === c.url,
      );
    });
  }
  updateDatasetSectionUi();

  syncFormFromDraft();
  renderProgress();
}

function renderProgress() {
  const total = state.papers.length;
  const reviewed = Object.keys(state.annotations).length;
  const withDataset = Object.values(state.annotations).filter(
    (a) => (a.dataset_url || "").trim(),
  ).length;
  const noUrl = Object.values(state.annotations).filter(
    (a) =>
      a.no_dataset_url === "true" ||
      a.no_dataset_url === true ||
      (!(a.dataset_url || "").trim() &&
        String(a.resource_notes || "").includes("NO_RESOURCE")),
  ).length;
  els.progressStats.innerHTML = `
    <div><strong>${reviewed}</strong> / ${total} revus</div>
    <div>${withDataset} avec dataset_url</div>
    <div>${noUrl} sans URL (no_dataset_url)</div>
  `;
}

function renderAll() {
  renderQueue();
  renderPaper();
}

function exportAnnotations() {
  const rows = Object.values(state.annotations).map((a) => ({
    paper: a.paper,
    dataset_url: a.dataset_url || "",
    no_dataset_url: a.no_dataset_url || "false",
    code_url: a.code_url || "",
    resource_license: a.resource_license || "",
    resource_format: a.resource_format || "",
    resource_access: a.resource_access || "",
    resource_notes: a.resource_notes || "",
  }));
  rows.sort((a, b) => a.paper.localeCompare(b.paper));
  downloadFile(
    "resource_annotations.csv",
    toCsv(rows, ANNOTATION_HEADERS),
    "text/csv;charset=utf-8",
  );
}

async function loadQueueFromText(text) {
  const papers = JSON.parse(text);
  if (!Array.isArray(papers)) {
    throw new Error("papers_queue.json doit être un tableau JSON.");
  }
  state.papers = papers;
  state.currentIndex = 0;
  restoreCurrentIndexFromSavedId();
  saveState();
  renderAll();
}

async function importAnnotationsFromText(text) {
  const rows = parseCsv(text);
  for (const row of rows) {
    const paper = row.paper || row.anthology_id;
    if (!paper) {
      continue;
    }
    state.annotations[paper] = {
      paper,
      dataset_url: row.dataset_url || "",
      no_dataset_url: row.no_dataset_url || "false",
      code_url: row.code_url || "",
      resource_license: row.resource_license || "",
      resource_format: row.resource_format || "",
      resource_access: row.resource_access || "",
      resource_notes: row.resource_notes || "",
      updated_at: row.updated_at || "",
    };
  }
  saveState();
  renderAll();
}

function initEvents() {
  els.loadQueueBtn.addEventListener("click", () => els.queueFileInput.click());
  els.queueFileInput.addEventListener("change", async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) {
      return;
    }
    await loadQueueFromText(await file.text());
    ev.target.value = "";
  });

  els.importAnnotationsBtn.addEventListener("click", () =>
    els.annotationsFileInput.click(),
  );
  els.annotationsFileInput.addEventListener("change", async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) {
      return;
    }
    await importAnnotationsFromText(await file.text());
    ev.target.value = "";
  });

  els.exportAnnotationsBtn.addEventListener("click", exportAnnotations);
  els.clearStateBtn.addEventListener("click", () => {
    if (!confirm("Effacer la session locale (localStorage) ?")) {
      return;
    }
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY_LEGACY);
    state.papers = [];
    state.annotations = {};
    state.currentIndex = 0;
    state.savedCurrentPaperId = null;
    renderAll();
  });

  els.queueSearch.addEventListener("input", (ev) => {
    state.queueSearch = ev.target.value;
    state.currentIndex = 0;
    renderQueue();
    renderPaper();
  });

  els.filterPendingOnly.addEventListener("change", (ev) => {
    state.filterPendingOnly = ev.target.checked;
    state.currentIndex = 0;
    saveState();
    renderAll();
  });

  els.filterNoCandOnly.addEventListener("change", (ev) => {
    state.filterNoCandOnly = ev.target.checked;
    state.currentIndex = 0;
    saveState();
    renderAll();
  });

  els.noDatasetUrlRadio.addEventListener("change", () => {
    if (els.noDatasetUrlRadio.checked) {
      setNoDatasetUrl(true);
    }
  });

  els.manualDatasetUrl.addEventListener("input", () => {
    if (els.manualDatasetUrl.value.trim()) {
      setNoDatasetUrl(false);
      state.selectedDatasetUrl = "";
      document.querySelectorAll('input[name="datasetPick"]').forEach((r) => {
        r.checked = false;
      });
    }
    updateOpenDatasetBtn();
  });

  ["codeUrl", "resourceLicense", "resourceFormat", "resourceAccess", "resourceNotes"].forEach(
    (id) => {
      document.getElementById(id).addEventListener("input", readDraftFromForm);
      document.getElementById(id).addEventListener("change", readDraftFromForm);
    },
  );

  els.openDatasetBtn.addEventListener("click", openDatasetUrl);
  els.saveBtn.addEventListener("click", () => saveCurrent(false));
  els.saveNextBtn.addEventListener("click", () => {
    if (saveCurrent(false)) {
      goNext();
    }
  });
  els.noResourceBtn.addEventListener("click", () => {
    setNoDatasetUrl(true);
    if (saveCurrent(true)) {
      goNext();
    }
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, textarea, select") && ev.key !== "s") {
      return;
    }
    if (ev.key === "j" || ev.key === "J") {
      ev.preventDefault();
      goNext();
    } else if (ev.key === "k" || ev.key === "K") {
      ev.preventDefault();
      goPrev();
    } else if (ev.key === "o" || ev.key === "O") {
      ev.preventDefault();
      openDatasetUrl();
    } else if (ev.key === "s" || ev.key === "S") {
      ev.preventDefault();
      if (saveCurrent(false)) {
        goNext();
      }
    }
  });

  document.body.addEventListener("dragover", (ev) => {
    ev.preventDefault();
  });
  document.body.addEventListener("drop", async (ev) => {
    ev.preventDefault();
    const file = ev.dataTransfer?.files?.[0];
    if (!file) {
      return;
    }
    if (file.name.endsWith(".json")) {
      await loadQueueFromText(await file.text());
    } else if (file.name.endsWith(".csv")) {
      await importAnnotationsFromText(await file.text());
    }
  });
}

loadState();
initEvents();
renderProgress();
if (Object.keys(state.annotations).length > 0) {
  els.queueMeta.textContent = `${Object.keys(state.annotations).length} annotations en local — chargez papers_queue.json`;
}
