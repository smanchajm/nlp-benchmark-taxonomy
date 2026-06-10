"use strict";

const STORAGE_KEY = "taxonomy_builder_state_v1";
const SNAPSHOTS_KEY = "taxonomy_builder_snapshots_v1";

function buildDefaultTree() {
  return {
    id: "root",
    label: "taxonomy_root",
    description: "Racine de la taxonomie.",
    children: [],
  };
}

const state = {
  leaves: [],
  tree: buildDefaultTree(),
  assignments: [],
  decisions: [],
  selectedNodeId: "root",
  leavesSearch: "",
  nodesSearch: "",
  showUnassignedOnly: true,
  showPendingOnly: false,
  hidePendingLeaves: false,
  showLeavesInTree: true,
  centerFocusMode: false,
  sunburstFocusNodeId: null,
  leavesPanelCollapsed: false,
  pendingLeafIds: [],
  boundaryDecision: false,
  extensibleOk: false,
  collapsed: {},
  treeView: "list",
  sessionId: `s_${Date.now()}`,
  phase: "E->C",
  roundManager: {
    filterCurrentRound: false,
    currentRoundId: 1,
    rounds: [
      { id: 1, name: "Round 1", target: 30, leaf_ids: [], structural_changes: 0, status: "pending", started_at: null, completed_at: null, assignments: 0 },
      { id: 2, name: "Round 2", target: 60, leaf_ids: [], structural_changes: 0, status: "pending", started_at: null, completed_at: null, assignments: 0 },
      { id: 3, name: "Round 3", target: 0, leaf_ids: [], structural_changes: 0, status: "pending", started_at: null, completed_at: null, assignments: 0 },
    ],
  },
  conceptMode: {
    query: "",
    candidates: [],
  },
};

const els = {
  loadLeavesBtn: document.getElementById("loadLeavesBtn"),
  exportBtn: document.getElementById("exportBtn"),
  importSessionBtn: document.getElementById("importSessionBtn"),
  snapshotBtn: document.getElementById("snapshotBtn"),
  compareBtn: document.getElementById("compareBtn"),
  resetTreeBtn: document.getElementById("resetTreeBtn"),
  exportSessionBtn: document.getElementById("exportSessionBtn"),
  exportTreeWithLeavesBtn: document.getElementById("exportTreeWithLeavesBtn"),
  exportReviewHtmlBtn: document.getElementById("exportReviewHtmlBtn"),
  exportSupervisorBtn: document.getElementById("exportSupervisorBtn"),
  treeViewToggleBtn: document.getElementById("treeViewToggleBtn"),
  boundaryDecisionToggle: document.getElementById("boundaryDecisionToggle"),
  leafSearchInput: document.getElementById("leafSearchInput"),
  nodeSearchInput: document.getElementById("nodeSearchInput"),
  toggleCenterFocusBtn: document.getElementById("toggleCenterFocusBtn"),
  showUnassignedOnly: document.getElementById("showUnassignedOnly"),
  showPendingOnly: document.getElementById("showPendingOnly"),
  hidePendingLeaves: document.getElementById("hidePendingLeaves"),
  showLeavesInTree: document.getElementById("showLeavesInTree"),
  toggleLeavesPanelBtn: document.getElementById("toggleLeavesPanelBtn"),
  leavesPanelBody: document.getElementById("leavesPanelBody"),
  leavesMeta: document.getElementById("leavesMeta"),
  leavesList: document.getElementById("leavesList"),
  treeMeta: document.getElementById("treeMeta"),
  treeContainer: document.getElementById("treeContainer"),
  layoutRoot: document.getElementById("layoutRoot"),
  panelLeft: document.getElementById("panelLeft"),
  panelCenter: document.getElementById("panelCenter"),
  panelRight: document.getElementById("panelRight"),
  inspector: document.getElementById("inspector"),
  decisionLog: document.getElementById("decisionLog"),
  nickersonPanel: document.getElementById("nickersonPanel"),
  extensibleCheckbox: document.getElementById("extensibleCheckbox"),
  comparePanel: document.getElementById("comparePanel"),
  roundManagerCard: document.getElementById("roundManagerCard"),
  conceptSearchCard: document.getElementById("conceptSearchCard"),
  iterationHistoryCard: document.getElementById("iterationHistoryCard"),
  memoInput: document.getElementById("memoInput"),
  saveMemoBtn: document.getElementById("saveMemoBtn"),
  contextMenu: document.getElementById("contextMenu"),
  decisionDialog: document.getElementById("decisionDialog"),
  decisionDialogTitle: document.getElementById("decisionDialogTitle"),
  decisionDialogTarget: document.getElementById("decisionDialogTarget"),
  decisionForm: document.getElementById("decisionForm"),
  decisionJustification: document.getElementById("decisionJustification"),
  leavesFileInput: document.getElementById("leavesFileInput"),
  sessionFilesInput: document.getElementById("sessionFilesInput"),
};

let pendingDecisionResolver = null;

function nowIso() {
  return new Date().toISOString();
}

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function wordCount(text) {
  const cleaned = (text || "").trim();
  return cleaned ? cleaned.split(/\s+/).length : 0;
}

function normalizeLeafRow(row) {
  return {
    id: String(row.id),
    label: String(row.label || row.id || "no_label"),
    description: String(row.description || ""),
    paraphrase_task: String(row.paraphrase_task || ""),
    freq: Number(row.freq || 0),
    contexts: Array.isArray(row.contexts) ? row.contexts.map(String) : [],
  };
}

function normalizeRoundManager(roundManager) {
  const fallback = clone(state.roundManager);
  if (!roundManager || !Array.isArray(roundManager.rounds) || !roundManager.rounds.length) {
    return fallback;
  }
  return {
    filterCurrentRound: Boolean(roundManager.filterCurrentRound),
    currentRoundId: Number(roundManager.currentRoundId || 1),
    rounds: roundManager.rounds.map((r, idx) => ({
      id: Number(r.id || idx + 1),
      name: String(r.name || `Round ${idx + 1}`),
      target: Number(r.target || 0),
      leaf_ids: Array.isArray(r.leaf_ids) ? r.leaf_ids.map(String) : [],
      structural_changes: Number(r.structural_changes || 0),
      status: String(r.status || "pending"),
      started_at: r.started_at || null,
      completed_at: r.completed_at || null,
      assignments: Number(r.assignments || 0),
    })),
  };
}

function normalizeNode(node) {
  return {
    id: String(node.id),
    label: String(node.label || node.id || "no_label"),
    description: String(node.description || ""),
    children: Array.isArray(node.children) ? node.children.map(normalizeNode) : [],
  };
}

function flattenTree(root) {
  const nodesById = new Map();
  const parentById = new Map();
  function walk(node, parentId = null) {
    nodesById.set(node.id, node);
    parentById.set(node.id, parentId);
    for (const child of node.children) {
      walk(child, node.id);
    }
  }
  walk(root);
  return { nodesById, parentById };
}

function generateNodeId(prefix = "node") {
  const { nodesById } = flattenTree(state.tree);
  let i = 1;
  while (nodesById.has(`${prefix}_${String(i).padStart(4, "0")}`)) {
    i += 1;
  }
  return `${prefix}_${String(i).padStart(4, "0")}`;
}

function nextDecisionId() {
  let maxId = 0;
  for (const row of state.decisions) {
    const m = String(row.id || "").match(/(\d+)/);
    if (m) {
      maxId = Math.max(maxId, Number(m[1]));
    }
  }
  return `d_${String(maxId + 1).padStart(4, "0")}`;
}

function getAssignedNodeByLeafId() {
  const map = new Map();
  for (const row of state.assignments) {
    map.set(String(row.leaf_id), String(row.node_id));
  }
  return map;
}

function collectDescendantIds(node) {
  const ids = [node.id];
  for (const child of node.children) {
    ids.push(...collectDescendantIds(child));
  }
  return ids;
}

function collectLeavesForNode(nodeId) {
  const { nodesById } = flattenTree(state.tree);
  const node = nodesById.get(nodeId);
  if (!node) {
    return [];
  }
  const descendantIds = new Set(collectDescendantIds(node));
  const leafSet = new Set();
  for (const row of state.assignments) {
    if (descendantIds.has(String(row.node_id))) {
      leafSet.add(String(row.leaf_id));
    }
  }
  return Array.from(leafSet);
}

function subtreeAssignedCount(nodeId) {
  return collectLeavesForNode(nodeId).length;
}

function directAssignedCount(nodeId) {
  return state.assignments.filter((r) => String(r.node_id) === String(nodeId)).length;
}

function getLeafById(leafId) {
  return state.leaves.find((l) => String(l.id) === String(leafId)) || null;
}

function leafLabel(leaf) {
  return String(leaf?.label || leaf?.id || "no_label");
}

function leafDescription(leaf) {
  return String(leaf?.description || "");
}

function collectDirectAssignedLeafIds(nodeId) {
  return state.assignments
    .filter((row) => String(row.node_id) === String(nodeId))
    .map((row) => String(row.leaf_id));
}

function isPendingLeaf(leafId) {
  return state.pendingLeafIds.includes(String(leafId));
}

function setLeafPending(leafId, pending) {
  const lid = String(leafId);
  if (pending) {
    if (!state.pendingLeafIds.includes(lid)) {
      state.pendingLeafIds.push(lid);
    }
  } else {
    state.pendingLeafIds = state.pendingLeafIds.filter((id) => id !== lid);
  }
}

function collectDirectAssignmentRows(nodeId) {
  return state.assignments.filter((row) => String(row.node_id) === String(nodeId));
}

function assignedLeafNodesForTree(nodeId) {
  return collectDirectAssignmentRows(nodeId)
    .map((row) => {
      const leaf = getLeafById(row.leaf_id);
      return {
        leaf_id: String(row.leaf_id),
        label: leafLabel(leaf),
        description: leafDescription(leaf),
        date: String(row.date || ""),
      };
    })
    .sort((a, b) => a.label.localeCompare(b.label));
}

function leafPaperCount(leafId) {
  const leaf = getLeafById(leafId);
  return Number(leaf?.freq || 0);
}

function directPaperCount(nodeId) {
  return collectDirectAssignedLeafIds(nodeId).reduce(
    (acc, leafId) => acc + leafPaperCount(leafId),
    0,
  );
}

function subtreePaperCount(nodeId) {
  return collectLeavesForNode(nodeId).reduce((acc, leafId) => acc + leafPaperCount(leafId), 0);
}

function buildTreeWithAssignedLeaves(node) {
  const leafChildren = assignedLeafNodesForTree(node.id).map((leaf) => ({
    node_type: "leaf_assignment",
    leaf_id: leaf.leaf_id,
    label: leaf.label,
    description: leaf.description,
    assigned_at: leaf.date,
    children: [],
  }));
  return {
    node_type: "taxonomy_node",
    id: node.id,
    label: node.label,
    description: node.description,
    children: [
      ...node.children.map((child) => buildTreeWithAssignedLeaves(child)),
      ...leafChildren,
    ],
  };
}

function renderLeafPreviewList(leafIds, maxRows = 10) {
  if (!leafIds.length) {
    return "<div class=\"small-muted\">Aucune feuille.</div>";
  }
  const rows = leafIds.slice(0, maxRows).map((leafId) => {
    const leaf = getLeafById(leafId);
    const lbl = leafLabel(leaf);
    const desc = leafDescription(leaf);
    const short = desc.length > 120 ? `${desc.slice(0, 119)}…` : desc;
    return `<div class="inspector-leaf-row"><b>${lbl}</b><span>${short || leafId}</span></div>`;
  });
  const extra = leafIds.length > maxRows ? `<div class="small-muted">+ ${leafIds.length - maxRows} autres...</div>` : "";
  return `${rows.join("")}${extra}`;
}

function saveState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      leaves: state.leaves,
      tree: state.tree,
      assignments: state.assignments,
      decisions: state.decisions,
      extensibleOk: state.extensibleOk,
      collapsed: state.collapsed,
      pendingLeafIds: state.pendingLeafIds,
      treeView: state.treeView,
      showPendingOnly: state.showPendingOnly,
      hidePendingLeaves: state.hidePendingLeaves,
      showLeavesInTree: state.showLeavesInTree,
      centerFocusMode: state.centerFocusMode,
      sunburstFocusNodeId: state.sunburstFocusNodeId,
      leavesPanelCollapsed: state.leavesPanelCollapsed,
      sessionId: state.sessionId,
      phase: state.phase,
      roundManager: state.roundManager,
      conceptMode: state.conceptMode,
    }),
  );
}

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return;
  }
  try {
    const parsed = JSON.parse(raw);
    state.leaves = Array.isArray(parsed.leaves)
      ? parsed.leaves.map((row) => normalizeLeafRow(row))
      : [];
    state.tree = parsed.tree ? normalizeNode(parsed.tree) : state.tree;
    state.assignments = Array.isArray(parsed.assignments) ? parsed.assignments : [];
    state.decisions = Array.isArray(parsed.decisions) ? parsed.decisions : [];
    state.extensibleOk = Boolean(parsed.extensibleOk);
    state.collapsed = parsed.collapsed || {};
    state.pendingLeafIds = Array.isArray(parsed.pendingLeafIds)
      ? parsed.pendingLeafIds.map(String)
      : [];
    const allowedTreeViews = new Set(["list", "d3", "sunburst", "treemap"]);
    state.treeView = allowedTreeViews.has(parsed.treeView) ? parsed.treeView : "list";
    state.showPendingOnly = Boolean(parsed.showPendingOnly);
    state.hidePendingLeaves = Boolean(parsed.hidePendingLeaves);
    state.showLeavesInTree = parsed.showLeavesInTree !== false;
    state.centerFocusMode = Boolean(parsed.centerFocusMode);
    state.sunburstFocusNodeId = parsed.sunburstFocusNodeId || null;
    state.leavesPanelCollapsed = Boolean(parsed.leavesPanelCollapsed);
    state.sessionId = parsed.sessionId || state.sessionId;
    state.phase = parsed.phase || "E->C";
    state.roundManager = normalizeRoundManager(parsed.roundManager);
    if (parsed.conceptMode) {
      state.conceptMode = {
        query: String(parsed.conceptMode.query || ""),
        candidates: Array.isArray(parsed.conceptMode.candidates)
          ? parsed.conceptMode.candidates
          : [],
      };
    }
  } catch (err) {
    console.warn("Cannot restore local state:", err);
  }
}

function escapeCsv(value) {
  const s = value == null ? "" : String(value);
  if (!/[,"\n]/.test(s)) {
    return s;
  }
  return `"${s.replace(/"/g, '""')}"`;
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

function getCurrentRound() {
  return state.roundManager.rounds.find((r) => r.id === state.roundManager.currentRoundId);
}

function hashString(text) {
  let h = 2166136261;
  const s = String(text || "");
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
  }
  return Math.abs(h >>> 0);
}

function deterministicShuffle(arr) {
  return [...arr].sort((a, b) => hashString(a.id) - hashString(b.id));
}

function stratifyLeaves(leaves) {
  const ordered = [...leaves].sort((a, b) => Number(b.freq || 0) - Number(a.freq || 0));
  const n = ordered.length;
  const cut1 = Math.floor(n / 3);
  const cut2 = Math.floor((2 * n) / 3);
  return [
    deterministicShuffle(ordered.slice(0, cut1)),
    deterministicShuffle(ordered.slice(cut1, cut2)),
    deterministicShuffle(ordered.slice(cut2)),
  ];
}

function consumeFromStrata(strata, target) {
  const picked = [];
  while (picked.length < target && strata.some((s) => s.length)) {
    for (const s of strata) {
      if (!s.length) {
        continue;
      }
      picked.push(s.shift());
      if (picked.length >= target) {
        break;
      }
    }
  }
  return picked;
}

function initializeRoundSampling() {
  if (!state.leaves.length) {
    alert("Charge d'abord les feuilles.");
    return;
  }
  const strata = stratifyLeaves(state.leaves);
  const round1 = consumeFromStrata(strata, 30);
  const round2 = consumeFromStrata(strata, 60);
  const round3 = [...strata[0], ...strata[1], ...strata[2]];
  const now = nowIso();
  state.roundManager.rounds = [
    { id: 1, name: "Round 1", target: 30, leaf_ids: round1.map((x) => x.id), structural_changes: 0, status: "active", started_at: now, completed_at: null, assignments: 0 },
    { id: 2, name: "Round 2", target: 60, leaf_ids: round2.map((x) => x.id), structural_changes: 0, status: "pending", started_at: null, completed_at: null, assignments: 0 },
    { id: 3, name: "Round 3", target: round3.length, leaf_ids: round3.map((x) => x.id), structural_changes: 0, status: "pending", started_at: null, completed_at: null, assignments: 0 },
  ];
  state.roundManager.currentRoundId = 1;
  state.roundManager.filterCurrentRound = true;
}

function switchRound(roundId) {
  const target = state.roundManager.rounds.find((r) => r.id === roundId);
  if (!target) {
    return;
  }
  state.roundManager.currentRoundId = roundId;
  if (target.status === "pending") {
    target.status = "active";
    target.started_at = target.started_at || nowIso();
  }
}

function completeCurrentRound() {
  const current = getCurrentRound();
  if (!current) {
    return;
  }
  current.status = "completed";
  current.completed_at = nowIso();
  const next = state.roundManager.rounds.find((r) => r.id > current.id && r.status !== "completed");
  if (next) {
    state.roundManager.currentRoundId = next.id;
    if (next.status === "pending") {
      next.status = "active";
      next.started_at = next.started_at || nowIso();
    }
  }
}

function computeStabilityTwoRounds() {
  const completed = state.roundManager.rounds.filter((r) => r.status === "completed");
  if (completed.length < 2) {
    return false;
  }
  const lastTwo = completed.slice(-2);
  return lastTwo.every((r) => Number(r.structural_changes || 0) === 0);
}

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9_ ]+/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

function jaccard(a, b) {
  if (!a.size || !b.size) {
    return 0;
  }
  let inter = 0;
  for (const x of a) {
    if (b.has(x)) {
      inter += 1;
    }
  }
  const union = a.size + b.size - inter;
  return union ? inter / union : 0;
}

function suggestConceptCandidates(query, limit = 20) {
  const qSet = new Set(tokenize(query));
  const assigned = getAssignedNodeByLeafId();
  const rows = state.leaves.map((leaf) => {
    const txt = `${leafLabel(leaf)} ${leafDescription(leaf)}`;
    const s = jaccard(qSet, new Set(tokenize(txt)));
    const bonus = assigned.has(String(leaf.id)) ? 0 : 0.05;
    return { leaf_id: leaf.id, score: s + bonus, assigned: assigned.get(String(leaf.id)) || null };
  });
  return rows
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function addMemoDecision(memoText) {
  const words = wordCount(memoText);
  if (!memoText.trim()) {
    alert("Memo vide.");
    return;
  }
  if (words > 300) {
    alert("Memo trop long (max 300 mots).");
    return;
  }
  addDecision({
    type: "memo",
    target: state.selectedNodeId || "global",
    justification: memoText.trim(),
    leavesConcernees: [],
    memo: memoText.trim(),
    structural: false,
  });
  els.memoInput.value = "";
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderTreeAsHtmlList(node, depth = 0) {
  const label = escapeHtml(node.label);
  const nodeId = escapeHtml(node.id);
  const desc = escapeHtml(node.description || "-");
  const children = node.children || [];
  const subtreePapers = subtreePaperCount(node.id);
  const directPapers = directPaperCount(node.id);
  const assignedLeaves = subtreeAssignedCount(node.id);
  const openByDefault = depth <= 1 ? "open" : "";
  const inner = children.map((child) => renderTreeAsHtmlList(child, depth + 1)).join("");
  return `<li class="tree-item">
    <details ${openByDefault}>
      <summary>
        <span class="node-title">${label}</span>
        <span class="node-id">(${nodeId})</span>
        <span class="chip">Papiers sous-arbre: ${subtreePapers}</span>
        <span class="chip">Papiers directs: ${directPapers}</span>
        <span class="chip">Feuilles assignees: ${assignedLeaves}</span>
      </summary>
      <div class="node-desc">${desc}</div>
      ${inner ? `<ul>${inner}</ul>` : ""}
    </details>
  </li>`;
}

function exportReviewHtmlNamed(filename) {
  const n = computeNickerson();
  const totalPapers = subtreePaperCount(state.tree.id);
  const totalNodes = flattenTree(state.tree).nodesById.size;
  const rounds = state.roundManager.rounds
    .map(
      (r) =>
        `<tr><td>${r.name}</td><td>${r.status}</td><td>${r.leaf_ids.length}</td><td>${r.assignments}</td><td>${r.structural_changes}</td></tr>`,
    )
    .join("");
  const decisions = state.decisions
    .slice(-200)
    .map(
      (d) =>
        `<tr><td>${d.id}</td><td>${d.date}</td><td>${d.type}</td><td>${d.target}</td><td>${d.justification || ""}</td></tr>`,
    )
    .join("");
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Taxonomy Review Export</title>
  <style>
    body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:16px;color:#111827;line-height:1.45}
    .header{border:1px solid #e5e7eb;background:#f8fafc;padding:12px;border-radius:10px}
    .metrics{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
    .chip{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#312e81;border-radius:999px;padding:1px 8px;font-size:11px;margin-left:6px}
    table{border-collapse:collapse;width:100%}
    td,th{border:1px solid #ddd;padding:6px;font-size:12px;text-align:left;vertical-align:top}
    ul{line-height:1.35;padding-left:18px}
    h2{margin-top:24px}
    .tree-item{margin:4px 0}
    details{border:1px solid #e5e7eb;border-radius:8px;background:#fff;padding:6px 8px}
    summary{cursor:pointer;list-style:none}
    summary::-webkit-details-marker{display:none}
    .node-title{font-weight:700}
    .node-id{color:#6b7280;font-size:12px;margin-left:4px}
    .node-desc{color:#374151;font-size:13px;margin:6px 0 4px 0;padding-left:2px}
  </style></head><body>
  <h1>Taxonomy Review Export</h1>
  <div class="header">
    <div>Date: ${new Date().toLocaleString()} | Session: ${state.sessionId}</div>
    <div class="metrics">
      <span class="chip">Noeuds: ${totalNodes}</span>
      <span class="chip">Papiers (total sous-arbre racine): ${totalPapers}</span>
      <span class="chip">Feuilles assignees: ${state.assignments.length}</span>
    </div>
  </div>
  <h2>Nickerson Snapshot</h2>
  <ul><li>Coverage: ${n.assignedUnique}/${n.totalLeaves} (${n.coverage}%)</li><li>Concise violations: ${
    n.conciseViolations.length
  }</li><li>Robust violations: ${n.robustViolations.length}</li><li>Explanatory violations: ${
    n.explanatoryViolations.length
  }</li><li>Two stable rounds: ${computeStabilityTwoRounds() ? "yes" : "no"}</li></ul>
  <h2>Rounds</h2><table><thead><tr><th>Round</th><th>Status</th><th>Leaves</th><th>Assignments</th><th>Structural changes</th></tr></thead><tbody>${rounds}</tbody></table>
  <h2>Taxonomy Tree</h2><ul>${renderTreeAsHtmlList(state.tree, 0)}</ul>
  <h2>Decisions (last 200)</h2><table><thead><tr><th>ID</th><th>Date</th><th>Type</th><th>Target</th><th>Justification/Memo</th></tr></thead><tbody>${decisions}</tbody></table>
  </body></html>`;
  downloadFile(filename, html, "text/html;charset=utf-8");
}

function exportReviewHtml() {
  exportReviewHtmlNamed("taxonomy_review.html");
}

function resetTreeToStart() {
  const confirmed = confirm(
    "Repartir du debut ?\n\nCela vide l'arbre, les assignations et les decisions, puis reinitialise les rounds.\nLes feuilles chargees sont conservees.",
  );
  if (!confirmed) {
    return;
  }
  state.tree = buildDefaultTree();
  state.assignments = [];
  state.pendingLeafIds = [];
  state.decisions = [];
  state.selectedNodeId = "root";
  state.collapsed = {};
  state.conceptMode = { query: "", candidates: [] };
  state.phase = "E->C";
  if (state.leaves.length) {
    initializeRoundSampling();
  }
  afterMutation();
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

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function collectHierarchyRows(node, rows = [], parentId = "", level = 0, path = "") {
  const nodePath = path ? `${path} > ${node.label}` : String(node.label || "");
  rows.push({
    id: String(node.id || ""),
    parent_id: String(parentId || ""),
    level: level,
    label: String(node.label || ""),
    description: String(node.description || ""),
    path: nodePath,
    n_children: Number((node.children || []).length),
    subtree_papers: Number(subtreePaperCount(node.id)),
    direct_papers: Number(directPaperCount(node.id)),
    subtree_assigned_leaves: Number(subtreeAssignedCount(node.id)),
    direct_assigned_leaves: Number(directAssignedCount(node.id)),
  });
  for (const child of node.children || []) {
    collectHierarchyRows(child, rows, node.id, level + 1, nodePath);
  }
  return rows;
}

function renderHierarchyText(node, level = 0) {
  const indent = "  ".repeat(level);
  const lines = [];
  const subtreePapers = subtreePaperCount(node.id);
  const directPapers = directPaperCount(node.id);
  const nChildren = (node.children || []).length;
  lines.push(
    `${indent}- ${node.label} | papers=${subtreePapers} (direct=${directPapers}) | children=${nChildren}`,
  );
  for (const child of node.children || []) {
    lines.push(renderHierarchyText(child, level + 1));
  }
  return lines.join("\n");
}

function renderNodeDescriptionsText(node, lines = []) {
  const label = String(node.label || "").replace(/\s+/g, " ").trim();
  const desc = String(node.description || "-").replace(/\s+/g, " ").trim() || "-";
  lines.push(`${label} | ${desc}`);
  for (const child of node.children || []) {
    renderNodeDescriptionsText(child, lines);
  }
  return lines;
}

function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Impossible de generer le PNG liste."));
          return;
        }
        resolve(blob);
      },
      "image/png",
      0.95,
    );
  });
}

async function exportHierarchyListPng(filename = "taxonomy_hierarchy_list.png") {
  const rows = collectHierarchyRows(state.tree);
  const font = "13px Arial, sans-serif";
  const lineHeight = 22;
  const leftPad = 20;
  const topPad = 20;
  const indentStep = 22;
  const footerPad = 20;

  const probe = document.createElement("canvas");
  const probeCtx = probe.getContext("2d");
  probeCtx.font = font;
  let maxWidth = 0;
  for (const row of rows) {
    const text = `${row.label}  |  papiers: ${row.subtree_papers}`;
    const w = probeCtx.measureText(text).width + leftPad * 2 + row.level * indentStep + 24;
    if (w > maxWidth) {
      maxWidth = w;
    }
  }

  const width = Math.max(1100, Math.ceil(maxWidth));
  const height = Math.max(480, topPad + rows.length * lineHeight + footerPad);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  let y = topPad + 8;
  ctx.font = font;
  for (const row of rows) {
    const x = leftPad + row.level * indentStep;
    const text = `${row.label}  |  papiers: ${row.subtree_papers}`;
    ctx.fillStyle = "#111827";
    ctx.fillText("•", x, y);
    ctx.fillStyle = "#1f2937";
    ctx.fillText(text, x + 12, y);
    y += lineHeight;
  }

  const blob = await canvasToPngBlob(canvas);
  downloadBlob(filename, blob);
}

function exportHierarchyFiles(base = "taxonomy_hierarchy") {
  const rows = collectHierarchyRows(state.tree);
  const csv = toCsv(rows, [
    "id",
    "parent_id",
    "level",
    "label",
    "description",
    "path",
    "n_children",
    "subtree_papers",
    "direct_papers",
    "subtree_assigned_leaves",
    "direct_assigned_leaves",
  ]);
  const textNodeDescriptions = renderNodeDescriptionsText(state.tree).join("\n");
  downloadFile(`${base}.csv`, csv, "text/csv;charset=utf-8");
  downloadFile(
    `${base}_labels_descriptions.txt`,
    textNodeDescriptions,
    "text/plain;charset=utf-8",
  );
}

function serializeSvg(svgNode) {
  const xml = new XMLSerializer().serializeToString(svgNode);
  if (!xml.includes('xmlns="http://www.w3.org/2000/svg"')) {
    return xml.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  return xml;
}

function svgToPngBlob(svgNode) {
  return new Promise((resolve, reject) => {
    try {
      const svgText = serializeSvg(svgNode);
      const svgBlob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const svgUrl = URL.createObjectURL(svgBlob);
      const img = new Image();
      img.onload = () => {
        const width = Number(svgNode.viewBox?.baseVal?.width) || svgNode.clientWidth || 1200;
        const height = Number(svgNode.viewBox?.baseVal?.height) || svgNode.clientHeight || 900;
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.floor(width));
        canvas.height = Math.max(1, Math.floor(height));
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(svgUrl);
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error("Impossible de generer le PNG."));
              return;
            }
            resolve(blob);
          },
          "image/png",
          0.95,
        );
      };
      img.onerror = () => {
        URL.revokeObjectURL(svgUrl);
        reject(new Error("Erreur de rendu SVG vers PNG."));
      };
      img.src = svgUrl;
    } catch (err) {
      reject(err);
    }
  });
}

async function exportSunburstPng(filename = "taxonomy_sunburst.png") {
  if (!window.TaxoSunburst) {
    throw new Error("Module sunburst indisponible.");
  }
  const host = document.createElement("div");
  host.style.position = "fixed";
  host.style.left = "-99999px";
  host.style.top = "-99999px";
  host.style.width = "1600px";
  host.style.height = "1100px";
  document.body.appendChild(host);
  try {
    window.TaxoSunburst.render(host, {
      root: state.tree,
      nodePaperCount: subtreePaperCount,
      directPaperCount,
    });
    const svg = host.querySelector("svg");
    if (!svg) {
      throw new Error("Sunburst SVG introuvable.");
    }
    const blob = await svgToPngBlob(svg);
    downloadBlob(filename, blob);
  } finally {
    host.remove();
  }
}

async function exportSupervisorBundle() {
  const stamp = new Date()
    .toISOString()
    .replace(/:/g, "-")
    .replace(/\..+$/, "");
  const base = `superviseur_${stamp}`;
  exportHierarchyFiles(`${base}_hierarchy`);
  await exportHierarchyListPng(`${base}_hierarchy_list.png`);
  await exportSunburstPng(`${base}_sunburst.png`);
}

function matchesNodeSearch(node, query) {
  if (!query) {
    return true;
  }
  const haystack = `${node.label} ${node.description}`.toLowerCase();
  return haystack.includes(query);
}

function nodeOrDescendantMatches(node, query) {
  if (!query) {
    return true;
  }
  if (matchesNodeSearch(node, query)) {
    return true;
  }
  if (state.showLeavesInTree) {
    const leafHit = assignedLeafNodesForTree(node.id).some((leaf) =>
      `${leaf.leaf_id} ${leaf.label} ${leaf.description}`.toLowerCase().includes(query),
    );
    if (leafHit) {
      return true;
    }
  }
  return node.children.some((child) => nodeOrDescendantMatches(child, query));
}

function isAssigned(leafId) {
  return state.assignments.some((row) => String(row.leaf_id) === String(leafId));
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

function openDecisionModal({ title, targetText }) {
  els.decisionDialogTitle.textContent = title;
  els.decisionDialogTarget.textContent = targetText;
  els.decisionJustification.value = "";
  return new Promise((resolve) => {
    pendingDecisionResolver = resolve;
    els.decisionDialog.showModal();
    els.decisionJustification.focus();
  });
}

function closeDecisionModal(result) {
  if (pendingDecisionResolver) {
    pendingDecisionResolver(result);
    pendingDecisionResolver = null;
  }
  els.decisionDialog.close();
}

function addDecision({
  type,
  target,
  justification,
  leavesConcernees,
  memo = "",
  structural = true,
  parentId = "",
  beforeLabel = "",
  afterLabel = "",
}) {
  const round = getCurrentRound();
  state.decisions.push({
    id: nextDecisionId(),
    date: nowIso(),
    type,
    target,
    justification,
    leaves_concernees: leavesConcernees.join("|"),
    memo,
    phase: state.phase,
    round_id: round ? round.id : "",
    session_id: state.sessionId,
    structural: structural ? "1" : "0",
    parent_id: parentId,
    before_label: beforeLabel,
    after_label: afterLabel,
  });
  if (round && structural) {
    round.structural_changes += 1;
  }
}

function withStructuralDecision(actionInfo, applyFn) {
  openDecisionModal({
    title: `Justification requise: ${actionInfo.type}`,
    targetText: actionInfo.target,
  }).then((justification) => {
    if (!justification) {
      return;
    }
    const wc = wordCount(justification);
    if (wc === 0 || wc > 20) {
      alert("La justification est obligatoire et doit contenir 20 mots maximum.");
      return;
    }
    const leavesConcernees = actionInfo.leavesConcernees || [];
    applyFn();
    addDecision({
      type: actionInfo.type,
      target: actionInfo.target,
      justification,
      leavesConcernees,
      structural: actionInfo.structural !== false,
      parentId: actionInfo.parentId || "",
      beforeLabel: actionInfo.beforeLabel || "",
      afterLabel: actionInfo.afterLabel || "",
    });
    afterMutation();
  });
}

function findNodeAndParent(nodeId, current = state.tree, parent = null) {
  if (current.id === nodeId) {
    return { node: current, parent };
  }
  for (const child of current.children) {
    const found = findNodeAndParent(nodeId, child, current);
    if (found) {
      return found;
    }
  }
  return null;
}

function moveAssignments(fromNodeId, toNodeId) {
  for (const row of state.assignments) {
    if (String(row.node_id) === String(fromNodeId)) {
      row.node_id = toNodeId;
      row.date = nowIso();
    }
  }
}

function deleteNode(nodeId) {
  if (nodeId === state.tree.id) {
    alert("Suppression de la racine interdite.");
    return;
  }
  const found = findNodeAndParent(nodeId);
  if (!found || !found.parent) {
    return;
  }
  const { node, parent } = found;
  const idx = parent.children.findIndex((c) => c.id === node.id);
  if (idx < 0) {
    return;
  }
  parent.children.splice(idx, 1, ...node.children);
  moveAssignments(node.id, parent.id);
  delete state.collapsed[node.id];
  if (state.selectedNodeId === node.id) {
    state.selectedNodeId = parent.id;
  }
}

function moveNode(nodeId, newParentId) {
  if (nodeId === state.tree.id) {
    alert("Deplacer la racine est interdit.");
    return false;
  }
  const source = findNodeAndParent(nodeId);
  const target = findNodeAndParent(newParentId);
  if (!source || !source.parent || !target) {
    return false;
  }
  const descendants = new Set(collectDescendantIds(source.node));
  if (descendants.has(newParentId)) {
    alert("Impossible de deplacer un noeud dans son propre sous-arbre.");
    return false;
  }
  source.parent.children = source.parent.children.filter((c) => c.id !== nodeId);
  target.node.children.push(source.node);
  return true;
}

function mergeWithSibling(nodeId, siblingId, newLabel, newDescription) {
  const found = findNodeAndParent(nodeId);
  if (!found || !found.parent) {
    return false;
  }
  const parent = found.parent;
  const sibling = parent.children.find((c) => c.id === siblingId);
  if (!sibling) {
    return false;
  }
  const node = found.node;
  parent.children = parent.children.filter(
    (c) => c.id !== node.id && c.id !== sibling.id,
  );
  const merged = {
    id: generateNodeId("merge"),
    label: newLabel,
    description: newDescription,
    children: [node, sibling],
  };
  parent.children.push(merged);
  return merged.id;
}

function splitNode(nodeId, labelA, labelB) {
  const found = findNodeAndParent(nodeId);
  if (!found) {
    return false;
  }
  const node = found.node;
  const childA = {
    id: generateNodeId("split"),
    label: labelA,
    description: "",
    children: [],
  };
  const childB = {
    id: generateNodeId("split"),
    label: labelB,
    description: "",
    children: [],
  };

  const directLeafRows = state.assignments.filter(
    (r) => String(r.node_id) === String(nodeId),
  );
  directLeafRows.forEach((r, idx) => {
    r.node_id = idx % 2 === 0 ? childA.id : childB.id;
    r.date = nowIso();
  });

  for (let i = 0; i < node.children.length; i += 1) {
    if (i % 2 === 0) {
      childA.children.push(node.children[i]);
    } else {
      childB.children.push(node.children[i]);
    }
  }
  node.children = [childA, childB];
  state.collapsed[node.id] = false;
  return true;
}

function createChild(nodeId, label, description) {
  const found = findNodeAndParent(nodeId);
  if (!found) {
    return null;
  }
  const child = {
    id: generateNodeId("node"),
    label,
    description: description || "",
    children: [],
  };
  found.node.children.push(child);
  state.collapsed[found.node.id] = false;
  return child.id;
}

function assignLeaf(leafId, nodeId) {
  setLeafPending(leafId, false);
  const currentRound = getCurrentRound();
  const hadExisting = state.assignments.some(
    (row) => String(row.leaf_id) === String(leafId),
  );
  const existing = state.assignments.find(
    (row) => String(row.leaf_id) === String(leafId),
  );
  if (existing) {
    existing.node_id = nodeId;
    existing.date = nowIso();
  } else {
    state.assignments.push({
      leaf_id: String(leafId),
      node_id: String(nodeId),
      date: nowIso(),
    });
  }
  if (currentRound && !hadExisting && currentRound.leaf_ids.includes(String(leafId))) {
    currentRound.assignments += 1;
  }
}

function unassignLeaf(leafId) {
  const lid = String(leafId);
  const existing = state.assignments.find((row) => String(row.leaf_id) === lid);
  if (!existing) {
    return null;
  }
  const oldNodeId = String(existing.node_id);
  state.assignments = state.assignments.filter((row) => String(row.leaf_id) !== lid);
  for (const round of state.roundManager.rounds) {
    if (round.leaf_ids.includes(lid) && round.assignments > 0) {
      round.assignments -= 1;
      break;
    }
  }
  return oldNodeId;
}

function handleLeafDrop(leafId, nodeId) {
  const leaf = state.leaves.find((l) => String(l.id) === String(leafId));
  if (!leaf) {
    return;
  }
  if (state.boundaryDecision) {
    withStructuralDecision(
      {
        type: "boundary_assignment",
        target: `${leaf.id} -> ${nodeId}`,
        leavesConcernees: [String(leaf.id)],
      },
      () => assignLeaf(leaf.id, nodeId),
    );
    return;
  }
  assignLeaf(leaf.id, nodeId);
  afterMutation();
}

function buildCoreExports() {
  const treeCanonical = JSON.stringify(state.tree, null, 2);
  const treeWithLeavesCanonical = JSON.stringify(
    buildTreeWithAssignedLeaves(state.tree),
    null,
    2,
  );
  const decisionsCsv = toCsv(state.decisions, [
    "id",
    "date",
    "type",
    "target",
    "justification",
    "leaves_concernees",
    "memo",
    "phase",
    "round_id",
    "session_id",
    "structural",
    "parent_id",
    "before_label",
    "after_label",
  ]);
  const assignmentsCsv = toCsv(state.assignments, ["leaf_id", "node_id", "date"]);
  const pendingCsv = toCsv(
    state.pendingLeafIds.map((leafId) => ({ leaf_id: String(leafId) })),
    ["leaf_id"],
  );
  const roundsCsv = toCsv(state.roundManager.rounds, [
    "id",
    "name",
    "target",
    "status",
    "started_at",
    "completed_at",
    "assignments",
    "structural_changes",
  ]);
  return {
    treeCanonical,
    treeWithLeavesCanonical,
    decisionsCsv,
    assignmentsCsv,
    pendingCsv,
    roundsCsv,
  };
}

function exportAll() {
  const {
    treeCanonical,
    treeWithLeavesCanonical,
    decisionsCsv,
    assignmentsCsv,
    pendingCsv,
    roundsCsv,
  } = buildCoreExports();

  downloadFile("tree.json", treeCanonical, "application/json;charset=utf-8");
  downloadFile(
    "tree_with_leaves.json",
    treeWithLeavesCanonical,
    "application/json;charset=utf-8",
  );
  downloadFile("decisions.csv", decisionsCsv, "text/csv;charset=utf-8");
  downloadFile("assignments.csv", assignmentsCsv, "text/csv;charset=utf-8");
  downloadFile("pending.csv", pendingCsv, "text/csv;charset=utf-8");
  downloadFile("rounds.csv", roundsCsv, "text/csv;charset=utf-8");
  saveSnapshot("auto_export");
}

function exportSessionBundle() {
  const stamp = new Date()
    .toISOString()
    .replace(/:/g, "-")
    .replace(/\..+$/, "");
  const base = `session_${stamp}`;
  const {
    treeCanonical,
    treeWithLeavesCanonical,
    decisionsCsv,
    assignmentsCsv,
    pendingCsv,
    roundsCsv,
  } = buildCoreExports();

  downloadFile(`${base}_tree.json`, treeCanonical, "application/json;charset=utf-8");
  downloadFile(
    `${base}_tree_with_leaves.json`,
    treeWithLeavesCanonical,
    "application/json;charset=utf-8",
  );
  downloadFile(`${base}_decisions.csv`, decisionsCsv, "text/csv;charset=utf-8");
  downloadFile(`${base}_assignments.csv`, assignmentsCsv, "text/csv;charset=utf-8");
  downloadFile(`${base}_pending.csv`, pendingCsv, "text/csv;charset=utf-8");
  downloadFile(`${base}_rounds.csv`, roundsCsv, "text/csv;charset=utf-8");
  exportReviewHtmlNamed(`${base}_taxonomy_review.html`);
  saveSnapshot(`auto_${base}`);
}

function exportTreeWithLeavesOnly() {
  const treeWithLeavesCanonical = JSON.stringify(
    buildTreeWithAssignedLeaves(state.tree),
    null,
    2,
  );
  downloadFile(
    "tree_with_leaves.json",
    treeWithLeavesCanonical,
    "application/json;charset=utf-8",
  );
}

function saveSnapshot(label = null) {
  const snapshots = JSON.parse(localStorage.getItem(SNAPSHOTS_KEY) || "[]");
  const name =
    label ||
    prompt("Nom du snapshot:", `session_${new Date().toISOString().slice(0, 19)}`) ||
    "";
  if (!name) {
    return;
  }
  snapshots.push({
    name,
    date: nowIso(),
    tree: clone(state.tree),
    assignments: clone(state.assignments),
    decisions: clone(state.decisions),
  });
  localStorage.setItem(SNAPSHOTS_KEY, JSON.stringify(snapshots));
  alert(`Snapshot enregistre: ${name}`);
}

function renderComparePanel() {
  const snapshots = JSON.parse(localStorage.getItem(SNAPSHOTS_KEY) || "[]");
  if (snapshots.length < 2) {
    alert("Il faut au moins 2 snapshots.");
    return;
  }
  els.comparePanel.classList.remove("hidden");
  const options = snapshots
    .map(
      (s, i) =>
        `<option value="${i}">${s.name} (${new Date(s.date).toLocaleString()})</option>`,
    )
    .join("");
  els.comparePanel.innerHTML = `
    <h3>Comparer snapshots</h3>
    <div class="controls-stack">
      <select id="snapA">${options}</select>
      <select id="snapB">${options}</select>
      <button id="runCompareBtn" type="button">Comparer</button>
    </div>
    <div id="compareResult"></div>
  `;
  document.getElementById("runCompareBtn").addEventListener("click", () => {
    const iA = Number(document.getElementById("snapA").value);
    const iB = Number(document.getElementById("snapB").value);
    const res = computeSnapshotDiff(snapshots[iA], snapshots[iB]);
    document.getElementById("compareResult").innerHTML = `
      <div><b>Ajoutes:</b> ${res.added.join(", ") || "-"}</div>
      <div><b>Supprimes:</b> ${res.removed.join(", ") || "-"}</div>
      <div><b>Renommes:</b> ${res.renamed.join(", ") || "-"}</div>
      <div><b>Deplaces:</b> ${res.moved.join(", ") || "-"}</div>
    `;
  });
}

function computeSnapshotDiff(a, b) {
  const mapTree = (tree) => {
    const out = new Map();
    function walk(node, parentId = null) {
      out.set(node.id, { label: node.label, parentId });
      for (const child of node.children || []) {
        walk(child, node.id);
      }
    }
    walk(tree);
    return out;
  };
  const A = mapTree(a.tree);
  const B = mapTree(b.tree);
  const added = [];
  const removed = [];
  const moved = [];
  const renamed = [];

  for (const id of B.keys()) {
    if (!A.has(id)) {
      added.push(id);
    }
  }
  for (const id of A.keys()) {
    if (!B.has(id)) {
      removed.push(id);
    }
  }
  for (const [id, nodeA] of A.entries()) {
    if (!B.has(id)) {
      continue;
    }
    const nodeB = B.get(id);
    if (nodeA.label !== nodeB.label) {
      renamed.push(id);
    }
    if (nodeA.parentId !== nodeB.parentId) {
      moved.push(id);
    }
  }
  return { added, removed, renamed, moved };
}

function computeNickerson() {
  const { nodesById, parentById } = flattenTree(state.tree);
  const conciseViolations = [];
  const robustViolations = [];
  const explanatoryViolations = [];

  for (const node of nodesById.values()) {
    if (node.children.length > 7) {
      conciseViolations.push(node.id);
    }
    if (node.id !== state.tree.id) {
      const countLeaves = subtreeAssignedCount(node.id);
      if (countLeaves < 2 && node.children.length < 2) {
        robustViolations.push(node.id);
      }
    }
    if (!/[.!?]/.test((node.description || "").trim())) {
      explanatoryViolations.push(node.id);
    }
    parentById.get(node.id);
  }

  const assignedUnique = new Set(state.assignments.map((r) => String(r.leaf_id))).size;
  const totalLeaves = state.leaves.length;
  const coverage = totalLeaves === 0 ? 0 : Math.round((100 * assignedUnique) / totalLeaves);
  return {
    conciseViolations,
    robustViolations,
    explanatoryViolations,
    assignedUnique,
    totalLeaves,
    coverage,
    stableTwoRounds: computeStabilityTwoRounds(),
  };
}

function statusClass(ok, warn = false) {
  if (ok) {
    return "status-green";
  }
  return warn ? "status-orange" : "status-red";
}

function renderNickerson() {
  const n = computeNickerson();
  const conciseOk = n.conciseViolations.length === 0;
  const robustOk = n.robustViolations.length === 0;
  const explanatoryOk = n.explanatoryViolations.length === 0;
  const comprehensiveOk = n.assignedUnique === n.totalLeaves && n.totalLeaves > 0;

  els.nickersonPanel.innerHTML = `
    <div class="nickerson-item ${statusClass(conciseOk)}">
      <b>Concise</b>: ${conciseOk ? "OK" : "violations"}<br />
      <span>${conciseOk ? "-" : conciseList(n.conciseViolations)}</span>
    </div>
    <div class="nickerson-item ${statusClass(robustOk)}">
      <b>Robust</b>: ${robustOk ? "OK" : "singletons"}<br />
      <span>${robustOk ? "-" : conciseList(n.robustViolations)}</span>
    </div>
    <div class="nickerson-item ${statusClass(comprehensiveOk, n.coverage >= 70)}">
      <b>Comprehensive</b>: ${n.assignedUnique}/${n.totalLeaves} (${n.coverage}%)</b>
    </div>
    <div class="nickerson-item ${statusClass(explanatoryOk)}">
      <b>Explanatory</b>: ${explanatoryOk ? "OK" : "descriptions manquantes"}<br />
      <span>${explanatoryOk ? "-" : conciseList(n.explanatoryViolations)}</span>
    </div>
    <div class="nickerson-item ${
      state.extensibleOk ? "status-green" : "status-orange"
    }">
      <b>Extensible</b>: ${state.extensibleOk ? "valide manuellement" : "a confirmer"}
    </div>
    <div class="nickerson-item ${statusClass(n.stableTwoRounds, true)}">
      <b>Stabilite</b>: ${
        n.stableTwoRounds ? "2 rounds consecutifs sans changement structurel" : "pas encore atteint"
      }
    </div>
  `;
}

function conciseList(ids) {
  return ids.length > 8 ? `${ids.slice(0, 8).join(", ")}...` : ids.join(", ");
}

function renderLeaves() {
  const assignedMap = getAssignedNodeByLeafId();
  const { nodesById } = flattenTree(state.tree);
  const pendingSet = new Set(state.pendingLeafIds.map(String));
  const currentRound = getCurrentRound();
  const roundLeafSet =
    currentRound && state.roundManager.filterCurrentRound
      ? new Set(currentRound.leaf_ids.map(String))
      : null;
  const q = state.leavesSearch.toLowerCase();
  const leaves = state.leaves
    .filter((leaf) => {
      if (roundLeafSet && !roundLeafSet.has(String(leaf.id))) {
        return false;
      }
      if (state.hidePendingLeaves && pendingSet.has(String(leaf.id))) {
        return false;
      }
      if (state.showPendingOnly && !pendingSet.has(String(leaf.id))) {
        return false;
      }
      if (state.showUnassignedOnly && assignedMap.has(String(leaf.id))) {
        return false;
      }
      if (!q) {
        return true;
      }
      const haystack = `${leafLabel(leaf)} ${leafDescription(leaf)} ${leaf.id}`.toLowerCase();
      return haystack.includes(q);
    })
    .sort((a, b) => Number(b.freq || 0) - Number(a.freq || 0));

  const nPending = pendingSet.size;
  els.leavesMeta.textContent = `${leaves.length}/${state.leaves.length} feuilles affichees · ${nPending} en attente${
    roundLeafSet ? ` (round ${currentRound.id})` : ""
  }`;
  els.leavesList.innerHTML = "";
  for (const leaf of leaves) {
    const item = document.createElement("article");
    item.className = "leaf-item";
    item.draggable = true;
    item.dataset.leafId = String(leaf.id);
    const label = leafLabel(leaf);
    const description = leafDescription(leaf);
    const isPending = pendingSet.has(String(leaf.id));
    const assignedNodeId = assignedMap.get(String(leaf.id));
    const assignedNodeLabel = assignedNodeId
      ? nodesById.get(String(assignedNodeId))?.label || assignedNodeId
      : "";
    const shortDesc =
      description.length > 170 ? `${description.slice(0, 169)}…` : description;
    item.innerHTML = `
      <h4>${label}</h4>
      <div class="leaf-description">${shortDesc || "-"}</div>
      <div class="leaf-meta">
        <span>${leaf.id}</span>
        <span class="badge">freq ${Number(leaf.freq || 0)}</span>
        ${isPending ? '<span class="badge badge-pending">en attente</span>' : ""}
        <span>${assignedMap.has(String(leaf.id)) ? "assignee" : "non assignee"}</span>
      </div>
      ${
        assignedNodeId
          ? `<div class="leaf-assigned-node">Noeud: <b>${assignedNodeLabel}</b> <span class="small-muted">(${assignedNodeId})</span></div>`
          : ""
      }
      <div class="leaf-actions">
        <button type="button" data-pending-toggle="${leaf.id}">
          ${isPending ? "Retirer attente" : "Mettre en attente"}
        </button>
        ${
          assignedNodeId
            ? `<button type="button" data-unassign-leaf="${leaf.id}">Retirer classification</button>`
            : ""
        }
      </div>
    `;
    item.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("application/x-taxo-leaf", String(leaf.id));
      item.classList.add("dragging");
    });
    item.addEventListener("dragend", () => item.classList.remove("dragging"));
    item.querySelector("button[data-pending-toggle]")?.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const leafId = ev.currentTarget.dataset.pendingToggle;
      const nextPending = !pendingSet.has(String(leafId));
      setLeafPending(leafId, nextPending);
      addDecision({
        type: nextPending ? "pending_leaf" : "unpending_leaf",
        target: String(leafId),
        justification: "",
        leavesConcernees: [String(leafId)],
        memo: "",
        structural: false,
      });
      afterMutation();
    });
    item.querySelector("button[data-unassign-leaf]")?.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const leafId = ev.currentTarget.dataset.unassignLeaf;
      const previousNodeId = unassignLeaf(leafId);
      if (!previousNodeId) {
        return;
      }
      addDecision({
        type: "unassign_leaf",
        target: `${leafId} <- ${previousNodeId}`,
        justification: "",
        leavesConcernees: [String(leafId)],
        memo: "",
        structural: false,
      });
      afterMutation();
    });
    els.leavesList.appendChild(item);
  }
}

function renderTree() {
  if (state.treeView === "sunburst" && window.TaxoSunburst) {
    renderTreeSunburst();
    return;
  }
  if (state.treeView === "treemap" && window.TaxoTreemap) {
    renderTreeTreemap();
    return;
  }
  if (state.treeView === "d3" && window.TaxoD3) {
    renderTreeD3();
    return;
  }
  renderTreeList();
}

function renderTreeList() {
  const { nodesById } = flattenTree(state.tree);
  const nAssigned = state.assignments.length;
  const q = state.nodesSearch.trim().toLowerCase();
  els.treeMeta.textContent = state.showLeavesInTree
    ? `${nodesById.size} noeuds + ${nAssigned} feuilles assignees`
    : `${nodesById.size} noeuds`;
  els.treeContainer.innerHTML = "";

  const rootUl = document.createElement("ul");
  rootUl.className = "tree-root-list";
  rootUl.appendChild(renderTreeNode(state.tree, q));
  els.treeContainer.appendChild(rootUl);
}

function renderTreeD3() {
  const { nodesById } = flattenTree(state.tree);
  const nAssigned = state.assignments.length;
  const q = state.nodesSearch.trim().toLowerCase();
  els.treeMeta.textContent = state.showLeavesInTree
    ? `${nodesById.size} noeuds + ${nAssigned} feuilles assignees (D3)`
    : `${nodesById.size} noeuds (D3)`;
  window.TaxoD3.render(els.treeContainer, {
    root: state.tree,
    isCollapsed: (nodeId) => Boolean(state.collapsed[nodeId]),
    isSelected: (nodeId) => state.selectedNodeId === nodeId,
    nodeOrDescendantMatches: (node) => nodeOrDescendantMatches(node, q),
    sortChildren: (children) =>
      [...children].sort(
        (a, b) => subtreeAssignedCount(b.id) - subtreeAssignedCount(a.id),
      ),
    query: q,
    leafMatches: (leaf, query) =>
      `${leaf.leaf_id} ${leaf.label} ${leaf.description}`.toLowerCase().includes(query),
    showLeavesInTree: state.showLeavesInTree,
    assignedLeafNodesForTree,
    directAssignedCount,
    subtreeAssignedCount,
    onSelect: (nodeId) => {
      state.selectedNodeId = nodeId;
      renderAll();
    },
    onContextMenu: (x, y, nodeId) => {
      state.selectedNodeId = nodeId;
      openContextMenu(x, y, nodeId);
      renderInspector();
    },
    onToggleCollapse: (nodeId) => {
      if (!findNodeAndParent(nodeId)?.node.children.length) {
        return;
      }
      state.collapsed[nodeId] = !state.collapsed[nodeId];
      renderTree();
    },
    onLeafDrop: (leafId, nodeId) => {
      handleLeafDrop(leafId, nodeId);
    },
    onNodeDrop: (sourceNodeId, targetNodeId) => {
      const leaves = collectLeavesForNode(sourceNodeId);
      withStructuralDecision(
        {
          type: "move_node",
          target: `${sourceNodeId} -> ${targetNodeId}`,
          leavesConcernees: leaves,
        },
        () => moveNode(sourceNodeId, targetNodeId),
      );
    },
  });
}

function renderTreeSunburst() {
  const papers = subtreePaperCount(state.tree.id);
  els.treeMeta.textContent = `Sunburst zoomable · ${state.tree.label} (${state.tree.id}) · ${papers} papiers`;
  window.TaxoSunburst.render(els.treeContainer, {
    root: state.tree,
    nodePaperCount: subtreePaperCount,
    directPaperCount,
  });
}

function renderTreeTreemap() {
  const { nodesById } = flattenTree(state.tree);
  const papers = subtreePaperCount(state.tree.id);
  els.treeMeta.textContent = `Treemap · ${nodesById.size} noeuds · ${papers} papiers`;
  window.TaxoTreemap.render(els.treeContainer, {
    root: state.tree,
    nodePaperCount: subtreePaperCount,
    isSelected: (nodeId) => state.selectedNodeId === nodeId,
    onSelect: (nodeId) => {
      state.selectedNodeId = nodeId;
      renderInspector();
      renderTree();
    },
  });
}

function renderTreeNode(node, query) {
  if (!nodeOrDescendantMatches(node, query)) {
    return document.createElement("li");
  }
  const li = document.createElement("li");
  li.className = "tree-node";
  li.dataset.nodeId = node.id;
  const directAssigned = directAssignedCount(node.id);
  const subtreeAssigned = subtreeAssignedCount(node.id);
  const isSelected = state.selectedNodeId === node.id;
  const isCollapsed = Boolean(state.collapsed[node.id]);
  const hasLeafChildren = state.showLeavesInTree && directAssigned > 0;
  const showChildren = (node.children.length > 0 || hasLeafChildren) && !isCollapsed;

  const content = document.createElement("div");
  content.className = `tree-node-content${isSelected ? " selected" : ""}`;
  content.dataset.nodeId = node.id;
  content.draggable = node.id !== state.tree.id;
  content.innerHTML = `
    ${
      node.children.length || hasLeafChildren
        ? `<button type="button" class="tree-toggle">${isCollapsed ? "+" : "-"}</button>`
        : `<span class="tree-toggle"></span>`
    }
    <span class="tree-node-label">${node.label}</span>
    <span class="tree-node-meta">(direct:${directAssigned} sous-arbre:${subtreeAssigned} enfants:${node.children.length})</span>
  `;
  content.addEventListener("click", () => {
    state.selectedNodeId = node.id;
    renderAll();
  });
  content.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    state.selectedNodeId = node.id;
    openContextMenu(ev.clientX, ev.clientY, node.id);
  });
  content.addEventListener("dragstart", (ev) => {
    ev.stopPropagation();
    ev.dataTransfer.setData("application/x-taxo-node", node.id);
    content.classList.add("dragging");
  });
  content.addEventListener("dragend", () => {
    content.classList.remove("dragging");
    content.classList.remove("drop-target");
  });
  content.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    content.classList.add("drop-target");
  });
  content.addEventListener("dragleave", () => content.classList.remove("drop-target"));
  content.addEventListener("drop", (ev) => {
    ev.preventDefault();
    content.classList.remove("drop-target");
    const leafId = ev.dataTransfer.getData("application/x-taxo-leaf");
    const nodeId = ev.dataTransfer.getData("application/x-taxo-node");
    if (leafId) {
      handleLeafDrop(leafId, node.id);
      return;
    }
    if (nodeId) {
      const leaves = collectLeavesForNode(nodeId);
      withStructuralDecision(
        {
          type: "move_node",
          target: `${nodeId} -> ${node.id}`,
          leavesConcernees: leaves,
        },
        () => moveNode(nodeId, node.id),
      );
    }
  });

  const toggleBtn = content.querySelector(".tree-toggle");
  if (toggleBtn && (node.children.length || hasLeafChildren)) {
    toggleBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      state.collapsed[node.id] = !state.collapsed[node.id];
      renderTree();
    });
  }
  li.appendChild(content);

  if (showChildren) {
    const childrenUl = document.createElement("ul");
    childrenUl.className = "tree-children";
    const sortedChildren = [...node.children].sort(
      (a, b) => subtreeAssignedCount(b.id) - subtreeAssignedCount(a.id),
    );
    for (const child of sortedChildren) {
      childrenUl.appendChild(renderTreeNode(child, query));
    }

    if (state.showLeavesInTree) {
      const leafRows = assignedLeafNodesForTree(node.id).filter((leaf) =>
        !query
          ? true
          : `${leaf.leaf_id} ${leaf.label} ${leaf.description}`
              .toLowerCase()
              .includes(query),
      );
      for (const leaf of leafRows) {
        const leafLi = document.createElement("li");
        leafLi.className = "tree-node";
        const leafDiv = document.createElement("div");
        leafDiv.className = "tree-leaf-node-content";
        leafDiv.draggable = true;
        const shortDesc =
          leaf.description.length > 150
            ? `${leaf.description.slice(0, 149)}…`
            : leaf.description;
        leafDiv.innerHTML = `
          <span class="tree-leaf-label">${leaf.leaf_id} · ${leaf.label}</span>
          <span class="tree-leaf-desc">${shortDesc || leaf.leaf_id}</span>
        `;
        leafDiv.addEventListener("dragstart", (ev) => {
          ev.dataTransfer.setData("application/x-taxo-leaf", leaf.leaf_id);
          leafDiv.classList.add("dragging");
        });
        leafDiv.addEventListener("dragend", () => {
          leafDiv.classList.remove("dragging");
        });
        leafLi.appendChild(leafDiv);
        childrenUl.appendChild(leafLi);
      }
    }
    li.appendChild(childrenUl);
  }
  return li;
}

function renderInspector() {
  const found = findNodeAndParent(state.selectedNodeId);
  if (!found) {
    els.inspector.innerHTML = "<div>Aucun noeud selectionne</div>";
    return;
  }
  const { node, parent } = found;
  const subtreeLeafIds = collectLeavesForNode(node.id);
  const directLeafIds = collectDirectAssignedLeafIds(node.id);
  els.inspector.innerHTML = `
    <h3>Noeud selectionne</h3>
    <div class="inspector-grid">
      <b>ID</b><span>${node.id}</span>
      <b>Label</b><span>${node.label}</span>
      <b>Description</b><span>${node.description || "-"}</span>
      <b>Parent</b><span>${parent ? parent.id : "-"}</span>
      <b>Enfants</b><span>${node.children.length}</span>
      <b>Feuilles directes</b><span>${directLeafIds.length}</span>
      <b>Feuilles sous-arbre</b><span>${subtreeLeafIds.length}</span>
    </div>
    <div class="inspector-assigned-block">
      <h4>Feuilles directes assignees</h4>
      <div class="inspector-leaf-list">${renderLeafPreviewList(directLeafIds, 12)}</div>
    </div>
    <div class="inspector-assigned-block">
      <h4>Feuilles sous-arbre</h4>
      <div class="inspector-leaf-list">${renderLeafPreviewList(subtreeLeafIds, 16)}</div>
    </div>
    <div class="inspector-actions">
      <button type="button" data-action="create-child">Creer enfant</button>
      <button type="button" data-action="rename-node">Renommer</button>
      <button type="button" data-action="edit-description">Editer description</button>
      <button type="button" data-action="merge-sibling">Fusionner frere</button>
      <button type="button" data-action="split-node">Scinder</button>
      <button type="button" data-action="move-node">Deplacer</button>
      <button type="button" data-action="delete-node">Supprimer</button>
    </div>
  `;
  els.inspector.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => runNodeAction(btn.dataset.action, node.id));
  });
}

function renderLog() {
  const rows = [...state.decisions].slice(-12).reverse();
  if (!rows.length) {
    els.decisionLog.innerHTML = "<div>Aucune decision structurelle.</div>";
    return;
  }
  els.decisionLog.innerHTML = rows
    .map(
      (r) => `
      <div class="log-entry">
        <b>${r.id}</b> ${new Date(r.date).toLocaleString()}<br />
        <span>${r.type} / ${r.target}</span> <span class="meta-chip">round ${r.round_id || "-"}</span> <span class="meta-chip">${r.phase || "-"}</span><br />
        <i>${r.justification || ""}</i>${r.memo ? `<br /><span>${r.memo}</span>` : ""}
      </div>
    `,
    )
    .join("");
}

function renderRoundManager() {
  const round = getCurrentRound();
  const options = state.roundManager.rounds
    .map(
      (r) =>
        `<option value="${r.id}" ${r.id === state.roundManager.currentRoundId ? "selected" : ""}>${r.name} (${r.status})</option>`,
    )
    .join("");
  const pendingCount = round
    ? round.leaf_ids.filter((id) => !isAssigned(id)).length
    : 0;
  els.roundManagerCard.innerHTML = `
    <h3>Round manager</h3>
    <div class="inline-row">
      <button id="initSamplingBtn" type="button" class="small-btn">Init sampling 30/60/reste</button>
      <button id="completeRoundBtn" type="button" class="small-btn">Clore round</button>
    </div>
    <div class="inline-row" style="margin-top:6px">
      <label>Round:</label>
      <select id="roundSelect">${options}</select>
      <span class="meta-chip">phase ${state.phase}</span>
    </div>
    <div class="inline-row" style="margin-top:6px">
      <label><input id="filterCurrentRoundCheckbox" type="checkbox" ${
        state.roundManager.filterCurrentRound ? "checked" : ""
      }/> Filtrer round courant</label>
      <label>Phase:
        <select id="phaseSelect">
          <option value="E->C" ${state.phase === "E->C" ? "selected" : ""}>E->C</option>
          <option value="C->E" ${state.phase === "C->E" ? "selected" : ""}>C->E</option>
        </select>
      </label>
    </div>
    <div class="meta-line" style="margin-top:6px">
      ${round ? `${round.name}: ${pendingCount} feuilles restantes dans le round` : "Aucun round"}
    </div>
  `;

  document.getElementById("initSamplingBtn").addEventListener("click", () => {
    initializeRoundSampling();
    afterMutation();
  });
  document.getElementById("completeRoundBtn").addEventListener("click", () => {
    completeCurrentRound();
    afterMutation();
  });
  document.getElementById("roundSelect").addEventListener("change", (ev) => {
    switchRound(Number(ev.target.value));
    afterMutation();
  });
  document
    .getElementById("filterCurrentRoundCheckbox")
    .addEventListener("change", (ev) => {
      state.roundManager.filterCurrentRound = ev.target.checked;
      afterMutation();
    });
  document.getElementById("phaseSelect").addEventListener("change", (ev) => {
    state.phase = ev.target.value;
    saveState();
  });
}

function renderConceptMode() {
  const selected = findNodeAndParent(state.selectedNodeId)?.node;
  const selectedText = selected
    ? `${selected.label}. ${selected.description || ""}`.trim()
    : "";
  const candidateHtml = state.conceptMode.candidates.length
    ? state.conceptMode.candidates
        .map((c) => {
          const leaf = getLeafById(c.leaf_id);
          return `<div class="candidate-item">
            <div class="candidate-item-title">${leafLabel(leaf)} <span class="meta-chip">score ${c.score.toFixed(
              2,
            )}</span> ${
              c.assigned ? `<span class="meta-chip">assignee->${c.assigned}</span>` : ""
            }</div>
            <div class="candidate-item-desc">${leafDescription(leaf) || "-"}</div>
            <button class="small-btn" data-candidate-assign="${c.leaf_id}">Assigner au noeud selectionne</button>
          </div>`;
        })
        .join("")
    : `<div class="small-muted">Aucun candidat pour l'instant.</div>`;

  els.conceptSearchCard.innerHTML = `
    <h3>Mode C→E guide</h3>
    <textarea id="conceptQueryInput" placeholder="Requete conceptuelle">${state.conceptMode.query || selectedText}</textarea>
    <div class="inline-row" style="margin-top:6px">
      <button id="useSelectedConceptBtn" type="button" class="small-btn">Utiliser noeud selectionne</button>
      <button id="runConceptSearchBtn" type="button" class="small-btn">Trouver candidates</button>
    </div>
    <div id="conceptCandidates" class="candidate-list" style="margin-top:6px">${candidateHtml}</div>
  `;

  document.getElementById("useSelectedConceptBtn").addEventListener("click", () => {
    state.conceptMode.query = selectedText;
    renderConceptMode();
  });
  document.getElementById("runConceptSearchBtn").addEventListener("click", () => {
    const query = document.getElementById("conceptQueryInput").value.trim();
    state.conceptMode.query = query;
    state.conceptMode.candidates = suggestConceptCandidates(query, 20);
    renderConceptMode();
    saveState();
  });
  els.conceptSearchCard
    .querySelectorAll("button[data-candidate-assign]")
    .forEach((btn) => {
      btn.addEventListener("click", () => {
        const leafId = btn.dataset.candidateAssign;
        if (!state.selectedNodeId) {
          alert("Selectionne un noeud cible.");
          return;
        }
        handleLeafDrop(leafId, state.selectedNodeId);
      });
    });
}

function renderIterationHistory() {
  const rows = state.roundManager.rounds
    .map(
      (r) => `<tr>
        <td>${r.name}</td>
        <td>${r.status}</td>
        <td>${r.leaf_ids.length}</td>
        <td>${r.assignments}</td>
        <td>${r.structural_changes}</td>
      </tr>`,
    )
    .join("");
  els.iterationHistoryCard.innerHTML = `
    <h3>Historique iterations</h3>
    <table class="history-table">
      <thead><tr><th>Round</th><th>Status</th><th>Feuilles</th><th>Assign.</th><th>Struct.</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="meta-line" style="margin-top:6px">
      Critere "2 rounds stables": <b>${computeStabilityTwoRounds() ? "OK" : "non atteint"}</b>
    </div>
  `;
}

function runNodeAction(action, nodeId) {
  hideContextMenu();
  const found = findNodeAndParent(nodeId);
  if (!found) {
    return;
  }
  const leaves = collectLeavesForNode(nodeId);

  if (action === "create-child") {
    const label = prompt("Label du nouvel enfant:", "new_node");
    if (!label) {
      return;
    }
    const description = prompt("Description du nouvel enfant:", "") || "";
    openDecisionModal({
      title: "Justification requise: create_node",
      targetText: `parent=${nodeId} label=${label}`,
    }).then((justification) => {
      if (!justification) {
        return;
      }
      const wc = wordCount(justification);
      if (wc === 0 || wc > 20) {
        alert("La justification est obligatoire et doit contenir 20 mots maximum.");
        return;
      }
      const childId = createChild(nodeId, label, description);
      let logged = false;
      try {
        addDecision({
          type: "create_node",
          target: `parent=${nodeId} label=${label}`,
          justification,
          leavesConcernees: leaves,
          structural: true,
          parentId: nodeId,
          beforeLabel: "",
          afterLabel: label,
        });
        logged = true;
      } catch (err) {
        console.error("create_node logging failed, fallback row inserted", err);
        const round = getCurrentRound();
        state.decisions.push({
          id: nextDecisionId(),
          date: nowIso(),
          type: "create_node",
          target: `parent=${nodeId} label=${label}`,
          justification,
          leaves_concernees: Array.isArray(leaves) ? leaves.join("|") : "",
          memo: "",
          phase: state.phase,
          round_id: round ? round.id : "",
          session_id: state.sessionId,
          structural: "1",
          parent_id: nodeId,
          before_label: "",
          after_label: label,
        });
        if (round) {
          round.structural_changes += 1;
        }
      }
      afterMutation();
      if (childId) {
        alert(
          logged
            ? `Noeud cree et logge: ${childId}`
            : `Noeud cree: ${childId} (log via fallback)`,
        );
      }
    });
    return;
  }

  if (action === "rename-node") {
    const nextLabel = prompt("Nouveau label:", found.node.label);
    if (!nextLabel || nextLabel === found.node.label) {
      return;
    }
    withStructuralDecision(
      {
        type: "rename_node",
        target: `${nodeId}: ${found.node.label} -> ${nextLabel}`,
        leavesConcernees: leaves,
      },
      () => {
        found.node.label = nextLabel;
      },
    );
    return;
  }

  if (action === "edit-description") {
    const nextDesc = prompt("Nouvelle description:", found.node.description || "");
    if (nextDesc == null || nextDesc === found.node.description) {
      return;
    }
    withStructuralDecision(
      {
        type: "edit_description",
        target: `${nodeId}`,
        leavesConcernees: leaves,
      },
      () => {
        found.node.description = nextDesc;
      },
    );
    return;
  }

  if (action === "merge-sibling") {
    if (!found.parent) {
      alert("La racine ne peut pas fusionner avec un frere.");
      return;
    }
    const siblings = found.parent.children.filter((c) => c.id !== nodeId);
    if (!siblings.length) {
      alert("Aucun frere disponible.");
      return;
    }
    const siblingInput = prompt(
      `ID du frere a fusionner:\n${siblings.map((s) => `${s.id} (${s.label})`).join("\n")}`,
    );
    if (!siblingInput) {
      return;
    }
    const sibling = siblings.find((s) => s.id === siblingInput.trim());
    if (!sibling) {
      alert("Frere introuvable.");
      return;
    }
    const newLabel = prompt("Label du noeud fusionne:", `${found.node.label}_merged`);
    if (!newLabel) {
      return;
    }
    const newDescription = prompt("Description du noeud fusionne:", "") || "";
    const leavesSibling = collectLeavesForNode(sibling.id);
    withStructuralDecision(
      {
        type: "merge_nodes",
        target: `${nodeId} + ${sibling.id}`,
        leavesConcernees: Array.from(new Set([...leaves, ...leavesSibling])),
      },
      () => mergeWithSibling(nodeId, sibling.id, newLabel, newDescription),
    );
    return;
  }

  if (action === "split-node") {
    const a = prompt("Label du sous-noeud A:", `${found.node.label}_a`);
    if (!a) {
      return;
    }
    const b = prompt("Label du sous-noeud B:", `${found.node.label}_b`);
    if (!b) {
      return;
    }
    withStructuralDecision(
      {
        type: "split_node",
        target: nodeId,
        leavesConcernees: leaves,
      },
      () => splitNode(nodeId, a, b),
    );
    return;
  }

  if (action === "move-node") {
    const { nodesById } = flattenTree(state.tree);
    const targetId = prompt(
      `Nouveau parent ID:\n${Array.from(nodesById.keys()).join(", ")}`,
      state.tree.id,
    );
    if (!targetId) {
      return;
    }
    withStructuralDecision(
      {
        type: "move_node",
        target: `${nodeId} -> ${targetId}`,
        leavesConcernees: leaves,
      },
      () => moveNode(nodeId, targetId.trim()),
    );
    return;
  }

  if (action === "delete-node") {
    withStructuralDecision(
      {
        type: "delete_node",
        target: nodeId,
        leavesConcernees: leaves,
      },
      () => deleteNode(nodeId),
    );
  }
}

function openContextMenu(x, y, nodeId) {
  const actions = [
    ["create-child", "Creer enfant"],
    ["rename-node", "Renommer"],
    ["edit-description", "Editer description"],
    ["merge-sibling", "Fusionner avec frere"],
    ["split-node", "Scinder"],
    ["move-node", "Deplacer"],
    ["delete-node", "Supprimer"],
  ];
  els.contextMenu.innerHTML = actions
    .map(
      ([key, label]) =>
        `<button type="button" data-node="${nodeId}" data-action="${key}">${label}</button>`,
    )
    .join("");
  els.contextMenu.style.left = `${x}px`;
  els.contextMenu.style.top = `${y}px`;
  els.contextMenu.classList.remove("hidden");
  els.contextMenu.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () =>
      runNodeAction(btn.dataset.action, btn.dataset.node),
    );
  });
}

function hideContextMenu() {
  els.contextMenu.classList.add("hidden");
  els.contextMenu.innerHTML = "";
}

function renderAll() {
  updateTreeViewToggleLabel();
  updateCenterFocusUI();
  updateLeavesPanelToggleLabel();
  renderLeaves();
  renderTree();
  renderRoundManager();
  renderInspector();
  renderLog();
  renderConceptMode();
  renderIterationHistory();
  renderNickerson();
}

function updateCenterFocusUI() {
  if (els.layoutRoot) {
    els.layoutRoot.classList.toggle("focus-center", state.centerFocusMode);
  }
  if (els.panelLeft && els.panelRight && els.panelCenter) {
    els.panelLeft.style.display = state.centerFocusMode ? "none" : "";
    els.panelRight.style.display = state.centerFocusMode ? "none" : "";
    els.panelCenter.style.gridColumn = state.centerFocusMode ? "1 / -1" : "";
  }
  if (els.toggleCenterFocusBtn) {
    els.toggleCenterFocusBtn.textContent = state.centerFocusMode
      ? "Vue normale"
      : "Focus Arbre";
  }
}

function updateLeavesPanelToggleLabel() {
  if (!els.toggleLeavesPanelBtn || !els.leavesPanelBody) {
    return;
  }
  if (state.leavesPanelCollapsed) {
    els.leavesPanelBody.classList.add("hidden");
    els.toggleLeavesPanelBtn.textContent = "Derouler";
  } else {
    els.leavesPanelBody.classList.remove("hidden");
    els.toggleLeavesPanelBtn.textContent = "Replier";
  }
}

function updateTreeViewToggleLabel() {
  if (!els.treeViewToggleBtn) {
    return;
  }
  if (state.treeView === "d3") {
    els.treeViewToggleBtn.textContent = "Vue: D3";
    return;
  }
  if (state.treeView === "sunburst") {
    els.treeViewToggleBtn.textContent = "Vue: Sunburst";
    return;
  }
  if (state.treeView === "treemap") {
    els.treeViewToggleBtn.textContent = "Vue: Treemap";
    return;
  }
  els.treeViewToggleBtn.textContent = "Vue: Liste";
}

function afterMutation() {
  saveState();
  renderAll();
}

async function loadLeavesFromFile(file) {
  const txt = await readFileAsText(file);
  const parsed = JSON.parse(txt);
  if (!Array.isArray(parsed)) {
    throw new Error("leaves.json doit etre une liste.");
  }
  state.leaves = parsed.map((row) => normalizeLeafRow(row));
  initializeRoundSampling();
  afterMutation();
}

async function importSessionFiles(files) {
  const allFiles = Array.from(files);
  const lowerName = (f) => String(f.name || "").toLowerCase();
  const pick = (predicates) =>
    allFiles.find((f) => predicates.some((p) => p(lowerName(f))));

  const treeFile = pick([
    (n) => n === "tree.json",
    (n) => n.endsWith("_tree.json"),
  ]);
  const decisionsFile = pick([
    (n) => n === "decisions.csv",
    (n) => n.endsWith("_decisions.csv"),
  ]);
  const assignmentsFile = pick([
    (n) => n === "assignments.csv",
    (n) => n.endsWith("_assignments.csv"),
  ]);
  const pendingFile = pick([
    (n) => n === "pending.csv",
    (n) => n.endsWith("_pending.csv"),
  ]);
  if (!treeFile || !decisionsFile || !assignmentsFile) {
    alert(
      "Selectionne les 3 fichiers session: tree + decisions + assignments (avec ou sans horodatage).",
    );
    return;
  }
  if (!confirm("Cette action ecrase l'etat courant. Continuer ?")) {
    return;
  }
  const [treeTxt, decisionsTxt, assignmentsTxt, pendingTxt] = await Promise.all([
    readFileAsText(treeFile),
    readFileAsText(decisionsFile),
    readFileAsText(assignmentsFile),
    pendingFile ? readFileAsText(pendingFile) : Promise.resolve(""),
  ]);
  state.tree = normalizeNode(JSON.parse(treeTxt));
  state.decisions = parseCsv(decisionsTxt).map((r) => ({
    id: r.id || nextDecisionId(),
    date: r.date || nowIso(),
    type: r.type || "",
    target: r.target || "",
    justification: r.justification || "",
    leaves_concernees: r.leaves_concernees || "",
    memo: r.memo || "",
    phase: r.phase || "E->C",
    round_id: r.round_id || "",
    session_id: r.session_id || state.sessionId,
    structural: r.structural || "1",
    parent_id: r.parent_id || "",
    before_label: r.before_label || "",
    after_label: r.after_label || "",
  }));
  state.assignments = parseCsv(assignmentsTxt).map((r) => ({
    leaf_id: String(r.leaf_id || ""),
    node_id: String(r.node_id || ""),
    date: r.date || nowIso(),
  }));
  state.pendingLeafIds = pendingTxt
    ? parseCsv(pendingTxt).map((r) => String(r.leaf_id || "")).filter(Boolean)
    : [];
  state.selectedNodeId = state.tree.id;
  afterMutation();
}

function initEvents() {
  els.loadLeavesBtn.addEventListener("click", () => els.leavesFileInput.click());
  els.exportBtn.addEventListener("click", exportAll);
  els.exportSessionBtn.addEventListener("click", exportSessionBundle);
  els.exportTreeWithLeavesBtn.addEventListener("click", exportTreeWithLeavesOnly);
  els.resetTreeBtn.addEventListener("click", resetTreeToStart);
  els.exportReviewHtmlBtn.addEventListener("click", exportReviewHtml);
  if (els.exportSupervisorBtn) {
    els.exportSupervisorBtn.addEventListener("click", async () => {
      try {
        await exportSupervisorBundle();
        alert("Export superviseur termine (Sunburst PNG + Liste PNG + CSV).");
      } catch (err) {
        console.error(err);
        alert(
          `Export partiel: CSV genere, un export PNG a echoue (${err?.message || err}).`,
        );
      }
    });
  }
  els.importSessionBtn.addEventListener("click", () => els.sessionFilesInput.click());
  els.snapshotBtn.addEventListener("click", () => saveSnapshot());
  els.compareBtn.addEventListener("click", renderComparePanel);
  els.treeViewToggleBtn.addEventListener("click", () => {
    if (state.treeView === "list") {
      state.treeView = "d3";
    } else if (state.treeView === "d3") {
      state.treeView = "sunburst";
    } else if (state.treeView === "sunburst") {
      state.treeView = "treemap";
    } else {
      state.treeView = "list";
      state.sunburstFocusNodeId = null;
    }
    afterMutation();
  });
  els.toggleCenterFocusBtn.addEventListener("click", () => {
    state.centerFocusMode = !state.centerFocusMode;
    afterMutation();
  });

  els.boundaryDecisionToggle.addEventListener("change", (ev) => {
    state.boundaryDecision = ev.target.checked;
    saveState();
  });
  els.leafSearchInput.addEventListener("input", (ev) => {
    state.leavesSearch = ev.target.value;
    renderLeaves();
  });
  els.nodeSearchInput.addEventListener("input", (ev) => {
    state.nodesSearch = ev.target.value;
    renderTree();
  });
  els.showUnassignedOnly.addEventListener("change", (ev) => {
    state.showUnassignedOnly = ev.target.checked;
    renderLeaves();
  });
  els.showPendingOnly.addEventListener("change", (ev) => {
    state.showPendingOnly = ev.target.checked;
    renderLeaves();
    saveState();
  });
  els.hidePendingLeaves.addEventListener("change", (ev) => {
    state.hidePendingLeaves = ev.target.checked;
    renderLeaves();
    saveState();
  });
  els.toggleLeavesPanelBtn.addEventListener("click", () => {
    state.leavesPanelCollapsed = !state.leavesPanelCollapsed;
    afterMutation();
  });
  els.showLeavesInTree.addEventListener("change", (ev) => {
    state.showLeavesInTree = ev.target.checked;
    afterMutation();
  });
  els.extensibleCheckbox.addEventListener("change", (ev) => {
    state.extensibleOk = ev.target.checked;
    afterMutation();
  });
  els.saveMemoBtn.addEventListener("click", () => {
    addMemoDecision(els.memoInput.value);
    afterMutation();
  });

  els.leavesFileInput.addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) {
      return;
    }
    try {
      await loadLeavesFromFile(file);
    } catch (err) {
      alert(`Echec chargement leaves: ${err.message}`);
    } finally {
      ev.target.value = "";
    }
  });

  els.sessionFilesInput.addEventListener("change", async (ev) => {
    const files = ev.target.files;
    if (!files || !files.length) {
      return;
    }
    try {
      await importSessionFiles(files);
    } catch (err) {
      alert(`Echec import session: ${err.message}`);
    } finally {
      ev.target.value = "";
    }
  });

  els.decisionForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const submitter = ev.submitter ? ev.submitter.value : "cancel";
    if (submitter === "confirm") {
      const text = els.decisionJustification.value.trim();
      closeDecisionModal(text);
      return;
    }
    closeDecisionModal(null);
  });

  window.addEventListener("click", (ev) => {
    if (!els.contextMenu.contains(ev.target)) {
      hideContextMenu();
    }
  });
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      hideContextMenu();
    }
  });
}

function bootstrap() {
  loadState();
  if (state.treeView === "d3" && !window.TaxoD3) {
    state.treeView = "list";
  }
  if (state.treeView === "sunburst" && !window.TaxoSunburst) {
    state.treeView = "list";
  }
  if (state.treeView === "treemap" && !window.TaxoTreemap) {
    state.treeView = "list";
  }
  state.selectedNodeId = state.tree.id;
  els.extensibleCheckbox.checked = state.extensibleOk;
  els.showUnassignedOnly.checked = state.showUnassignedOnly;
  els.showPendingOnly.checked = state.showPendingOnly;
  els.hidePendingLeaves.checked = state.hidePendingLeaves;
  els.showLeavesInTree.checked = state.showLeavesInTree;
  els.boundaryDecisionToggle.checked = state.boundaryDecision;
  initEvents();
  renderAll();
}

bootstrap();
