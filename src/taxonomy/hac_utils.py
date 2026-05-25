from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import LabelEncoder

EXCLUDED_LABELS: frozenset[str] = frozenset({"other", "topic_subject"})


def build_linkage(embeddings: np.ndarray, method: str = "average") -> np.ndarray:
    if method == "ward":
        return linkage(embeddings, method="ward", metric="euclidean")
    dist = squareform(cosine_distances(embeddings), checks=False)
    return linkage(dist, method=method)


def make_color_map(
    coarse_values: pd.Series,
    palette: list[str] | None = None,
) -> dict[str, str]:
    if palette is None:
        palette = pc.qualitative.Plotly
    unique = sorted(coarse_values.dropna().unique())
    return {label: palette[i % len(palette)] for i, label in enumerate(unique)}


def plot_dendrogram(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    coarse_col: str,
    label_col: str,
    title: str,
    color_map: dict[str, str] | None = None,
    color_threshold: float = 0.4,
) -> ff.Figure:
    coarse_values = df[coarse_col].fillna("Unknown")
    if color_map is None:
        color_map = make_color_map(coarse_values)

    leaf_labels = [
        f"{i:02d} | {bk[:30]} | [{tt[:35]}]"
        for i, (bk, tt) in enumerate(zip(df["bibkey"], df[label_col].fillna("?")))
    ]
    label_to_coarse = dict(zip(leaf_labels, coarse_values))

    fig = ff.create_dendrogram(
        embeddings,
        orientation="right",
        labels=leaf_labels,
        distfun=lambda x: squareform(cosine_distances(x), checks=False),
        linkagefun=lambda d: linkage(d, method="average"),
        color_threshold=color_threshold,
    )

    ticktext = (
        list(fig.layout.yaxis.ticktext) if fig.layout.yaxis.ticktext is not None else []
    )
    tickvals = (
        list(fig.layout.yaxis.tickvals) if fig.layout.yaxis.tickvals is not None else []
    )

    leaf_annotations = [
        dict(
            x=1.002,
            y=val,
            xref="paper",
            yref="y",
            text=text,
            showarrow=False,
            font=dict(
                color=color_map.get(label_to_coarse.get(text, "Unknown"), "black"),
                size=10,
                family="monospace",
            ),
            xanchor="left",
            align="left",
        )
        for text, val in zip(ticktext, tickvals)
    ]
    legend_annotations = [
        dict(
            x=0.01,
            y=1.0 - i * 0.04,
            xref="paper",
            yref="paper",
            text=f"■ {label}",
            showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left",
        )
        for i, (label, color) in enumerate(color_map.items())
    ]

    fig.update_layout(
        title=title,
        height=max(800, 22 * len(leaf_labels)),
        font=dict(size=11, family="monospace"),
        plot_bgcolor="white",
        autosize=True,
        annotations=leaf_annotations + legend_annotations,
        xaxis=dict(
            title="Cosine Distance",
            showgrid=True,
            gridcolor="lightgray",
            side="bottom",
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            side="right",
            automargin=True,
            fixedrange=False,
        ),
        margin=dict(l=20, r=450, t=80, b=20),
    )
    return fig


def plot_dendrogram_adaptive(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    root: TaxoNode,
    coarse_col: str,
    label_col: str,
    title: str,
    color_threshold: float = 0.4,
) -> ff.Figure:
    """
    Dendrogram where leaves are coloured by their adaptive cluster (from cut_tree_adaptive).

    Each cluster gets a distinct colour; the legend shows the majority coarse label
    and size of each cluster. Papers not assigned to any leaf (outliers skipped by
    the degenerate-split rule) are shown in grey.
    """
    n = len(df)
    flat = root.flat_labels(n=n)  # -1 for unassigned outliers

    # Build cluster display names: "C03 · NER (n=25)"
    leaf_nodes = root._leaf_nodes()
    cluster_names: dict[int, str] = {}
    for cid, node in enumerate(leaf_nodes):
        top = df.iloc[node.leaves][coarse_col].value_counts()
        label = top.index[0] if len(top) else "?"
        cluster_names[cid] = f"C{cid:02d} · {label} (n={len(node.leaves)})"

    coarse_values = pd.Series(
        [cluster_names.get(flat[i], "outlier") for i in range(n)],
        index=df.index,
    )

    palette = pc.qualitative.Alphabet + pc.qualitative.Dark24
    unique_clusters = sorted(cluster_names.values()) + ["outlier"]
    color_map = {
        lbl: (palette[i % len(palette)] if lbl != "outlier" else "#cccccc")
        for i, lbl in enumerate(unique_clusters)
    }

    leaf_labels = [
        f"{i:02d} | {bk[:30]} | [{tt[:35]}]"
        for i, (bk, tt) in enumerate(zip(df["bibkey"], df[label_col].fillna("?")))
    ]
    label_to_coarse = dict(zip(leaf_labels, coarse_values))

    fig = ff.create_dendrogram(
        embeddings,
        orientation="right",
        labels=leaf_labels,
        distfun=lambda x: squareform(cosine_distances(x), checks=False),
        linkagefun=lambda d: linkage(d, method="average"),
        color_threshold=color_threshold,
    )

    ticktext = (
        list(fig.layout.yaxis.ticktext) if fig.layout.yaxis.ticktext is not None else []
    )
    tickvals = (
        list(fig.layout.yaxis.tickvals) if fig.layout.yaxis.tickvals is not None else []
    )

    leaf_annotations = [
        dict(
            x=1.002,
            y=val,
            xref="paper",
            yref="y",
            text=text,
            showarrow=False,
            font=dict(
                color=color_map.get(label_to_coarse.get(text, "outlier"), "#cccccc"),
                size=10,
                family="monospace",
            ),
            xanchor="left",
            align="left",
        )
        for text, val in zip(ticktext, tickvals)
    ]
    legend_annotations = [
        dict(
            x=0.01,
            y=1.0 - i * 0.03,
            xref="paper",
            yref="paper",
            text=f"■ {lbl}",
            showarrow=False,
            font=dict(color=color_map[lbl], size=10),
            xanchor="left",
        )
        for i, lbl in enumerate(sorted(cluster_names.values()))
    ]

    fig.update_layout(
        title=title,
        height=max(800, 22 * len(leaf_labels)),
        font=dict(size=11, family="monospace"),
        plot_bgcolor="white",
        autosize=True,
        annotations=leaf_annotations + legend_annotations,
        xaxis=dict(
            title="Cosine Distance",
            showgrid=True,
            gridcolor="lightgray",
            side="bottom",
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            side="right",
            automargin=True,
            fixedrange=False,
        ),
        margin=dict(l=20, r=500, t=80, b=20),
    )
    return fig


def clustering_metrics(
    embeddings: np.ndarray,
    Z: np.ndarray,
    coarse_values: pd.Series,
    n_clusters: int = 15,
) -> dict:
    hac_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    le = LabelEncoder()
    coarse_encoded = le.fit_transform(coarse_values)

    ari = adjusted_rand_score(coarse_encoded, hac_labels)
    nmi = normalized_mutual_info_score(coarse_encoded, hac_labels)
    sil = silhouette_score(embeddings, hac_labels, metric="cosine")

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, np.nan)
    intra, inter = [], []
    n = len(hac_labels)
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j]
            (intra if hac_labels[i] == hac_labels[j] else inter).append(s)

    intra_mean = float(np.mean(intra))
    inter_mean = float(np.mean(inter))
    return {
        "n_clusters": len(set(hac_labels)),
        "n_clusters_requested": n_clusters,
        "ari": ari,
        "nmi": nmi,
        "silhouette": sil,
        "cosine_intra": intra_mean,
        "cosine_inter": inter_mean,
        "cosine_ratio": intra_mean / inter_mean,
        "hac_labels": hac_labels,
    }


def dendrogram_purity(
    Z: np.ndarray,
    labels: np.ndarray,
    excluded: frozenset[str] = EXCLUDED_LABELS,
) -> float:
    n = len(labels)
    node_leaves: dict[int, set[int]] = {i: {i} for i in range(n)}
    for k, (c1, c2, _, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        node_leaves[n + k] = node_leaves[c1] | node_leaves[c2]

    parent: dict[int, int] = {}
    for k, (c1, c2, _, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        parent[c1] = n + k
        parent[c2] = n + k

    def lca(i: int, j: int) -> int:
        ancestors_i: set[int] = set()
        curr = i
        while curr in parent:
            curr = parent[curr]
            ancestors_i.add(curr)
        curr = j
        while curr in parent:
            curr = parent[curr]
            if curr in ancestors_i:
                return curr
        return n + len(Z) - 1

    label_groups: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl not in excluded:
            label_groups.setdefault(lbl, []).append(i)

    total, count = 0.0, 0
    for lbl, group in label_groups.items():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                node = lca(group[a], group[b])
                leaves = node_leaves[node]
                purity = sum(labels[leave] == lbl for leave in leaves) / len(leaves)
                total += purity
                count += 1

    return total / count if count > 0 else 0.0


@dataclass
class TaxoNode:
    hac_id: int
    leaves: list[int]
    coherence: float
    children: list[TaxoNode] = field(default_factory=list)
    is_outlier: bool = False  # True for asymmetric-split orphan leaves
    # Optional metadata fields (introduced for LLM-merged trees).
    label: str | None = None
    description: str | None = None
    iteration_created: int | None = None
    confidence: float | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flat_labels(self, n: int | None = None) -> np.ndarray:
        """
        Flat cluster labels (0..K-1) for the n original papers.

        Must be called on the root. `n` should be the total number of input
        papers; if omitted, inferred from max leaf index (unsafe if the last
        paper is missing — pass `n` explicitly when possible).
        """
        if n is None:
            n = max(self.leaves) + 1
        labels = np.full(n, -1, dtype=int)
        for cluster_id, node in enumerate(self._leaf_nodes()):
            for idx in node.leaves:
                labels[idx] = cluster_id

        missing = int((labels == -1).sum())
        if missing:
            raise AssertionError(
                f"flat_labels: {missing}/{n} papers not assigned to any leaf — "
                f"cut_tree_adaptive lost papers, check force_split branch."
            )
        return labels

    def _leaf_nodes(self) -> list[TaxoNode]:
        if self.is_leaf:
            return [self]
        result = []
        for child in self.children:
            result.extend(child._leaf_nodes())
        return result

    def all_papers(self) -> set[int]:
        """All paper indices reachable under this node (debugging / audit)."""
        return set(self.leaves)

    def outlier_leaves(self) -> list[TaxoNode]:
        """Convenience: list terminal leaves marked as outliers."""
        return [n for n in self._leaf_nodes() if n.is_outlier]

    def to_dict(self) -> dict:
        """Recursive plain-dict representation (JSON-serializable)."""
        payload = {
            "hac_id": int(self.hac_id),
            "leaves": [int(i) for i in self.leaves],
            "coherence": float(self.coherence),
            "is_outlier": bool(self.is_outlier),
            "children": [c.to_dict() for c in self.children],
        }
        # Emit optional metadata only when present to stay backward compatible
        if self.label is not None:
            payload["label"] = self.label
        if self.description is not None:
            payload["description"] = self.description
        if self.iteration_created is not None:
            payload["iteration_created"] = int(self.iteration_created)
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> TaxoNode:
        return cls(
            hac_id=int(payload["hac_id"]),
            leaves=[int(i) for i in payload["leaves"]],
            coherence=float(payload["coherence"]),
            is_outlier=bool(payload.get("is_outlier", False)),
            children=[cls.from_dict(c) for c in payload.get("children", [])],
            label=payload.get("label"),
            description=payload.get("description"),
            iteration_created=payload.get("iteration_created"),
            confidence=payload.get("confidence"),
        )


def cut_tree_adaptive(
    Z: np.ndarray,
    embeddings: np.ndarray,
    hard_max: int = 50,
    min_gain: float = 0.05,
    min_cluster_size: int = 3,
) -> TaxoNode:
    """
    Three-regime adaptive cut of a HAC dendrogram, returned as a TaxoNode tree.

    Regimes (evaluated top-down on each node):
      - size > hard_max                                           → force split
      - size > 2*min_cluster_size, gain >= min_gain, both children large enough → coherence split
      - otherwise                                                 → leaf

    Coherence gain = weighted_mean(coh_children) - coh_parent.  A split is taken
    only when the gain is positive enough (relative criterion) AND both children
    have at least min_cluster_size leaves.  The minimum splittable size is
    2*min_cluster_size (the smallest size that can yield two viable children).

    Force splits are always taken when size > hard_max, even if the split is
    asymmetric: the smaller side is materialised as an outlier leaf so no paper
    is silently lost.

    Invariant guaranteed: every input paper appears in exactly one terminal leaf.
    """
    n = len(embeddings)

    # Pre-compute leaves and children for every internal node
    node_leaves: dict[int, list[int]] = {i: [i] for i in range(n)}
    for k, (c1, c2, _, _) in enumerate(Z):
        node_leaves[n + k] = node_leaves[int(c1)] + node_leaves[int(c2)]
    node_children: dict[int, tuple[int, int]] = {
        n + k: (int(c1), int(c2)) for k, (c1, c2, _, _) in enumerate(Z)
    }

    # Normalize once; cosine similarity = dot product on unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms

    def _coherence(leaves: list[int]) -> float:
        m = len(leaves)
        if m < 2:
            return 1.0
        sub = unit[leaves]
        sims = sub @ sub.T
        iu = np.triu_indices(m, k=1)
        return float(sims[iu].mean())

    def _make_leaf(
        node: int, leaves: list[int], coh: float, is_outlier: bool = False
    ) -> TaxoNode:
        return TaxoNode(
            hac_id=node,
            leaves=leaves,
            coherence=coh,
            is_outlier=is_outlier,
        )

    def _traverse(node: int) -> TaxoNode:
        leaves = node_leaves[node]
        size = len(leaves)
        coh = _coherence(leaves)
        c1, c2 = node_children.get(node, (None, None))

        # Terminal cases: HAC leaf, or cluster small enough to keep whole
        if c1 is None or size <= min_cluster_size:
            return _make_leaf(node, leaves, coh)

        size_c1, size_c2 = len(node_leaves[c1]), len(node_leaves[c2])
        both_large_enough = min(size_c1, size_c2) >= min_cluster_size

        # Safety split: cluster too broad to keep as a leaf
        if size > hard_max:
            if both_large_enough:
                return TaxoNode(
                    hac_id=node,
                    leaves=leaves,
                    coherence=coh,
                    children=[_traverse(c1), _traverse(c2)],
                )
            # Asymmetric: materialise the smaller side as an outlier leaf
            small, large = (c1, c2) if size_c1 < size_c2 else (c2, c1)
            small_leaves = node_leaves[small]
            small_node = _make_leaf(
                small,
                small_leaves,
                _coherence(small_leaves),
                is_outlier=True,
            )
            return TaxoNode(
                hac_id=node,
                leaves=leaves,
                coherence=coh,
                children=[small_node, _traverse(large)],
            )

        # Coherence-gain split: worthwhile only when both children are viable
        if size > 2 * min_cluster_size and both_large_enough:
            coh_c1 = _coherence(node_leaves[c1])
            coh_c2 = _coherence(node_leaves[c2])
            coh_children = (coh_c1 * size_c1 + coh_c2 * size_c2) / (size_c1 + size_c2)
            if coh_children - coh >= min_gain:
                return TaxoNode(
                    hac_id=node,
                    leaves=leaves,
                    coherence=coh,
                    children=[_traverse(c1), _traverse(c2)],
                )

        # Default: leaf
        return _make_leaf(node, leaves, coh)

    return _traverse(n + len(Z) - 1)


def flat_labels(root: TaxoNode, n: int) -> np.ndarray:
    """
    Assign each of the `n` input papers a terminal-leaf id.

    Walks the tree, collecting only nodes with no children. Asserts every
    paper is covered exactly once (catches regressions of the silent-loss bug).
    """
    labels = np.full(n, -1, dtype=np.int64)

    def _walk(node: TaxoNode) -> None:
        if not node.children:
            for paper_idx in node.leaves:
                labels[paper_idx] = node.hac_id
            return
        for child in node.children:
            _walk(child)

    _walk(root)

    missing = int((labels == -1).sum())
    if missing:
        raise AssertionError(
            f"flat_labels: {missing}/{n} papers not assigned to any leaf — "
            f"cut_tree_adaptive lost papers, check force_split branch."
        )
    return labels


def plot_taxo_tree(
    root: TaxoNode,
    df: pd.DataFrame,
    label_col: str,
    title: str = "Adaptive Taxonomy Tree",
) -> go.Figure:
    """
    Publication-style tree diagram of a TaxoNode tree (vertical, root on top).

    Nodes are displayed as labeled boxes (annotation cards) and connected with
    orthogonal edges for a D3-like hierarchy look. `df` must be indexed 0..n-1.
    """
    # ── 1. Compute top-down tidy layout positions ────────────────────────────
    # x = breadth, y = depth
    x_gap = 2.8
    y_gap = 2.0
    positions: dict[int, tuple[float, float]] = {}
    depths: dict[int, int] = {}
    leaf_counter: list[float] = [0.0]

    def _layout(node: TaxoNode, depth: int) -> None:
        depths[node.hac_id] = depth
        if node.is_leaf:
            positions[node.hac_id] = (leaf_counter[0], depth * y_gap)
            leaf_counter[0] += x_gap
            return
        # Trie les enfants par taille décroissante : gros à gauche, petit à droite
        node.children.sort(key=lambda c: len(c.leaves), reverse=True)
        for child in node.children:
            _layout(child, depth + 1)
        child_x = [positions[c.hac_id][0] for c in node.children]
        positions[node.hac_id] = (sum(child_x) / len(child_x), depth * y_gap)

    _layout(root, 0)

    # ── 2. Build orthogonal edges (elbows) grouped by depth ──────────────────
    palette = ["#7c8cf8", "#a778e8", "#d06ca7", "#e78e58", "#7cb37f", "#5fa7b8"]
    edge_by_depth: dict[int, tuple[list[float | None], list[float | None]]] = {}

    def _add_edges(node: TaxoNode) -> None:
        px, py = positions[node.hac_id]
        d = depths[node.hac_id]
        xs, ys = edge_by_depth.setdefault(d, ([], []))
        for child in node.children:
            cx, cy = positions[child.hac_id]
            mid_y = (py + cy) / 2.0
            # Vertical segment, horizontal segment, vertical segment.
            xs.extend([px, px, cx, cx, None])
            ys.extend([py, mid_y, mid_y, cy, None])
            _add_edges(child)

    _add_edges(root)

    # ── 3. Collect node labels / hover text ───────────────────────────────────
    def _top_label(leaves: list[int]) -> str:
        counts = df.iloc[leaves][label_col].value_counts()
        return str(counts.index[0]) if len(counts) else "?"

    def _fmt_label(raw: str, max_len: int = 20) -> str:
        txt = " ".join(str(raw).split())
        return txt if len(txt) <= max_len else f"{txt[: max_len - 1]}…"

    node_records: list[tuple[TaxoNode, float, float, int, str, str, int]] = []

    def _collect(node: TaxoNode) -> None:
        x, y = positions[node.hac_id]
        d = depths[node.hac_id]
        n = len(node.leaves)
        top = _top_label(node.leaves)
        display = node.label if getattr(node, "label", None) else top
        text = (
            f"<b>{str(display)}</b><br>"
            f"n={n} · coh={node.coherence:.2f}"
            f"{'<br>leaf' if node.is_leaf else ''}"
        )
        box = f"{_fmt_label(display)}<br><span style='font-size:10px'>n={n}</span>"
        node_records.append((node, x, y, d, box, text, len(_fmt_label(display))))
        for child in node.children:
            _collect(child)

    _collect(root)

    # ── 4. Assemble figure (frame + edges + boxed labels) ────────────────────
    n_leaves = max(1, len(root._leaf_nodes()))
    max_depth = max(depths.values()) if depths else 0
    max_x = max(x for x, _ in positions.values()) if positions else 0.0
    max_y = max(y for _, y in positions.values()) if positions else 0.0

    fig = go.Figure()

    # Outer frame for "panel" look
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=-1.2,
        x1=max_x + 1.2,
        y0=-1.0,
        y1=max_y + 1.1,
        line=dict(color="#c7ccd5", width=1.3),
        fillcolor="#f8fafc",
        layer="below",
    )

    for d, (xs, ys) in edge_by_depth.items():
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=palette[d % len(palette)], width=1.15),
                opacity=0.75,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Invisible points only for hover tooltips.
    fig.add_trace(
        go.Scatter(
            x=[x for _, x, _, _, _, _, _ in node_records],
            y=[y for _, _, y, _, _, _, _ in node_records],
            mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            text=[t for _, _, _, _, _, t, _ in node_records],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    for node, x, y, d, box_text, _, _ in node_records:
        color = palette[d % len(palette)]
        fill = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)"
        border = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.65)"
        # Stronger visual hierarchy focused on first meaningful level.
        if d == 1:
            border_width = 1.8
            border_pad = 7
            font_size = 12
        elif d == 0:
            border_width = 1.0
            border_pad = 3
            font_size = 9
        else:
            border_width = 1.2
            border_pad = 4
            font_size = 10
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=box_text,
            showarrow=False,
            align="center",
            font=dict(
                size=font_size,
                color="#0f172a",
                family="Arial",
            ),
            bgcolor=fill,
            bordercolor=border,
            borderwidth=border_width,
            borderpad=border_pad,
        )

    # Estimate width needed to prevent overlap (dense levels + long labels).
    depth_width_px: dict[int, float] = {}
    for _, _, _, d, _, _, label_len in node_records:
        box_px = 26 + min(label_len, 20) * 7.0
        depth_width_px[d] = depth_width_px.get(d, 0.0) + box_px
    depth_counts: dict[int, int] = {}
    for _, _, _, d, _, _, _ in node_records:
        depth_counts[d] = depth_counts.get(d, 0) + 1
    needed_width_px = 260.0
    for d, total_boxes in depth_width_px.items():
        gaps = max(0, depth_counts[d] - 1) * 20.0
        needed_width_px = max(needed_width_px, total_boxes + gaps + 220.0)

    fig.update_layout(
        title=title,
        width=max(1300, int(160 + 120 * n_leaves), int(needed_width_px)),
        height=max(620, int(220 + 125 * (max_depth + 1))),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        font=dict(family="Arial", size=11, color="#0f172a"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.4, max_x + 1.4],
            fixedrange=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[max_y + 1.4, -1.2],  # root on top
            fixedrange=False,
        ),
        margin=dict(l=30, r=30, t=90, b=35),
    )
    return fig


def cophenetic_table(
    Z: np.ndarray,
    labels: np.ndarray,
    excluded: frozenset[str] = EXCLUDED_LABELS,
) -> pd.DataFrame:
    coph_matrix = squareform(cophenet(Z))
    unique_labels = [label for label in np.unique(labels) if label not in excluded]

    rows = []
    for lbl in unique_labels:
        idx_in = np.where(labels == lbl)[0]
        idx_out = np.where((labels != lbl) & ~np.isin(labels, list(excluded)))[0]
        if len(idx_in) < 2 or len(idx_out) == 0:
            continue
        intra_vals = coph_matrix[np.ix_(idx_in, idx_in)][
            np.triu_indices(len(idx_in), k=1)
        ]
        inter_vals = coph_matrix[np.ix_(idx_in, idx_out)].ravel()
        intra_m = float(intra_vals.mean())
        inter_m = float(inter_vals.mean())
        rows.append(
            {
                "family": lbl,
                "intra": round(intra_m, 3),
                "inter": round(inter_m, 3),
                "ratio": round(intra_m / inter_m, 3),
                "n": len(idx_in),
            }
        )

    return pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
