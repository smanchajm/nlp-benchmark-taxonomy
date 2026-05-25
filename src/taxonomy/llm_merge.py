"""LLM-driven pairwise merge for taxonomy tree construction.

Takes the leaf clusters produced by HAC (`cut_tree_adaptive`) and iteratively
asks an LLM to propose N-ary merges among the current pool of nodes. The
process stops when no merge is defensible, producing a forest with variable
depth and branching factor.

Pipeline:
    1. seed_leaves: HAC leaves -> central paraphrases -> LLM label + description
    2. pairwise_merge_loop: iterative merge proposals until stable
    3. save_llm_tree / load_llm_tree: JSON persistence
    4. merge_tree_to_taxonode: bridge to existing plot_taxo_tree
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.taxonomy.hac_utils import TaxoNode
from src.taxonomy.providers import (
    call_structured,
    create_client,
    mistral_batch_fetch,
    mistral_batch_submit,
)
from src.taxonomy.schemas import (
    TASKS,
    LeafCluster,
    LeafLabelGen,
    MergeNode,
    MergeProposal,
    MergeRound,
)

logger = logging.getLogger(__name__)


LEAF_LABEL_TASK = TASKS["taxonomy_leaf_label"]
MERGE_PROPOSAL_TASK = TASKS["taxonomy_merge_proposal"]


# ─────────────────────────────────────────────────────────────────────────────
# Centroid-based paraphrase selection
# ─────────────────────────────────────────────────────────────────────────────


def select_central_paraphrases(
    member_indices: list[int],
    embeddings: np.ndarray,
    paraphrases: list[str],
) -> tuple[list[str], list[int]]:
    """Return top-k paraphrases closest to the cluster centroid (cosine).

    Args:
        member_indices: indices of cluster members in `embeddings` / `paraphrases`.
        embeddings: array of shape [n_total, d]. May be unnormalized.
        paraphrases: full list of paraphrases (one per row of `embeddings`).
        k: number of representatives to return. When None (default), uses an
           adaptive formula: clamp(ceil(log2(n)) + 1, 3, 10).

    Returns:
        (selected_paraphrases, selected_indices) — both length min(k, len(member_indices)),
        ordered by descending cosine similarity to the centroid.
    """
    if not member_indices:
        return [], []

    k = min(10, max(3, math.ceil(math.log2(len(member_indices))) + 1))

    member_embs = embeddings[member_indices]
    centroid = member_embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9

    norms = np.linalg.norm(member_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    member_unit = member_embs / norms

    sims = member_unit @ centroid
    order = np.argsort(-sims)[: min(k, len(member_indices))]
    selected_idx = [member_indices[int(i)] for i in order]
    selected_para = [paraphrases[i] for i in selected_idx]
    return selected_para, selected_idx


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: seed leaves
# ─────────────────────────────────────────────────────────────────────────────


def _format_paraphrases_block(paraphrases: Iterable[str]) -> str:
    return "\n".join(f"- {p.strip()}" for p in paraphrases)


async def seed_leaves(
    root: TaxoNode,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    paraphrase_col: str = "paraphrase_task",
    provider: str = "mistral",
    max_concurrent: int = 5,
    request_interval: float = 0.0,
) -> list[LeafCluster]:
    """For each HAC leaf, pick k central paraphrases and ask the LLM for a label.

    Args:
        root: HAC tree root from `cut_tree_adaptive`.
        df: DataFrame indexed 0..n-1 containing the source paraphrases.
        embeddings: array of shape [n, d]; rows aligned with df.
        paraphrase_col: column in `df` holding the paraphrase strings.
        provider: LLM provider key (must exist in MODEL_MAP).
        k_paraphrases: number of representative paraphrases per leaf. When None
            (default), uses adaptive formula: clamp(ceil(log2(n)) + 1, 3, 10).
        max_concurrent: semaphore size for async LLM calls.

    Returns:
        List of LeafCluster, one per HAC leaf, sorted by hac_id.
    """
    df = df.reset_index(drop=True)
    paraphrases_all = df[paraphrase_col].astype(str).tolist()
    leaf_nodes = root._leaf_nodes()
    logger.info("Seeding %d leaves via %s", len(leaf_nodes), provider)

    client = create_client(provider, async_=True)
    sem = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(leaf_nodes), desc=f"seed_leaves[{provider}]")

    # Pre-compute representative paraphrases per leaf
    rep_per_leaf: dict[int, tuple[list[str], list[int]]] = {}
    for cid, node in enumerate(leaf_nodes):
        rep_per_leaf[cid] = select_central_paraphrases(
            node.leaves, embeddings, paraphrases_all
        )

    async def _process_leaf(cid: int, node: TaxoNode) -> tuple[int, LeafCluster]:
        async with sem:
            if request_interval > 0:
                await asyncio.sleep(request_interval)
            rep_paraphrases, _ = rep_per_leaf[cid]
            gen = None
            for attempt in range(5):
                try:
                    gen = await call_structured(
                        client,
                        provider,
                        LEAF_LABEL_TASK,
                        n=len(node.leaves),
                        paraphrases_block=_format_paraphrases_block(rep_paraphrases),
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        wait = 15 * (2**attempt)
                        logger.warning(
                            "Leaf %d rate-limited, retrying in %ds (attempt %d/5)",
                            cid,
                            wait,
                            attempt + 1,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            "Leaf %d labelling failed: %s — falling back", cid, e
                        )
                        break
            if gen is None:
                gen = LeafLabelGen.model_construct(
                    reasoning="API error",
                    label=f"unlabelled_cluster_{cid:02d}",
                    description=(
                        "Cluster could not be labelled by the LLM (API error). "
                        "Members share an HAC-level proximity but no curated label."
                    ),
                )
            leaf = LeafCluster(
                id=f"leaf_{cid:02d}",
                label=gen.label,
                description=gen.description,
                paraphrases=rep_paraphrases,
                size=len(node.leaves),
                member_indices=list(node.leaves),
                coherence=float(node.coherence),
                hac_id=int(node.hac_id),
            )
            pbar.update(1)
            return cid, leaf

    results = await asyncio.gather(
        *[_process_leaf(cid, node) for cid, node in enumerate(leaf_nodes)]
    )
    pbar.close()

    results.sort(key=lambda x: x[0])
    return [leaf for _, leaf in results]


def _leaf_batch_items(
    root: TaxoNode,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    paraphrase_col: str,
) -> tuple[list[TaxoNode], list[tuple[str, dict]]]:
    """Shared helper: compute (leaf_nodes, batch items) for batch submit/results."""
    df = df.reset_index(drop=True)
    paraphrases_all = df[paraphrase_col].astype(str).tolist()
    leaf_nodes = root._leaf_nodes()
    items = []
    for cid, node in enumerate(leaf_nodes):
        rep_paraphrases, _ = select_central_paraphrases(
            node.leaves, embeddings, paraphrases_all
        )
        items.append(
            (
                str(cid),
                {
                    "n": len(node.leaves),
                    "paraphrases_block": _format_paraphrases_block(rep_paraphrases),
                },
            )
        )
    return leaf_nodes, items


def seed_leaves_batch_submit(
    root: TaxoNode,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    paraphrase_col: str = "paraphrase_task",
    model: str = "mistral-large-latest",
) -> str:
    """Submit a Mistral batch job to label all HAC leaves. Returns the job_id.

    Pair with seed_leaves_batch_results once the job reaches SUCCESS status.
    """
    leaf_nodes, items = _leaf_batch_items(root, df, embeddings, paraphrase_col)
    job_id = mistral_batch_submit(items, task="taxonomy_leaf_label", model=model)
    return job_id


def seed_leaves_batch_results(
    job_id: str,
    root: TaxoNode,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    paraphrase_col: str = "paraphrase_task",
) -> list[LeafCluster]:
    """Retrieve and reconstruct LeafClusters from a completed seed_leaves batch job."""
    df = df.reset_index(drop=True)
    paraphrases_all = df[paraphrase_col].astype(str).tolist()
    leaf_nodes = root._leaf_nodes()
    parsed_map = mistral_batch_fetch(job_id, task="taxonomy_leaf_label")
    leaves = []
    for cid, node in enumerate(leaf_nodes):
        rep_paraphrases, _ = select_central_paraphrases(
            node.leaves, embeddings, paraphrases_all
        )
        gen = parsed_map.get(str(cid))
        if gen is None:
            gen = LeafLabelGen.model_construct(
                reasoning="Batch error: no response",
                label=f"unlabelled_cluster_{cid:02d}",
                description="Cluster could not be labelled (batch error).",
            )
        leaves.append(
            LeafCluster(
                id=f"leaf_{cid:02d}",
                label=gen.label,
                description=gen.description,
                paraphrases=rep_paraphrases,
                size=len(node.leaves),
                member_indices=list(node.leaves),
                coherence=float(node.coherence),
                hac_id=int(node.hac_id),
            )
        )
    return leaves


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: iterative pairwise merge
# ─────────────────────────────────────────────────────────────────────────────


def _render_pool_node(node: MergeNode) -> str:
    paraphrases = "\n".join(f"    - {p.strip()}" for p in node.paraphrases)
    return (
        f"[{node.id}] label={node.label} | size={node.size}\n"
        f"  description: {node.description}\n"
        f"  paraphrases:\n{paraphrases}"
    )


def _render_pool_block(nodes: list[MergeNode]) -> str:
    return "\n\n".join(_render_pool_node(n) for n in nodes)


def _bubble_paraphrases(
    children: list[MergeNode], rng: np.random.Generator, k: int = 5
) -> list[str]:
    """Sample up to k paraphrases evenly from the children."""
    pool: list[str] = []
    per_child = max(1, k // max(len(children), 1))
    for child in children:
        if not child.paraphrases:
            continue
        idx = rng.permutation(len(child.paraphrases))[:per_child]
        pool.extend(child.paraphrases[int(i)] for i in idx)
    if len(pool) < k:
        leftover = [p for child in children for p in child.paraphrases if p not in pool]
        rng.shuffle(leftover)
        pool.extend(leftover[: k - len(pool)])
    return pool[:k]


def _resolve_overlapping_proposals(
    proposals: list[MergeProposal],
) -> tuple[list[MergeProposal], list[MergeProposal]]:
    """When a node appears in multiple proposals, keep the highest-confidence one.

    Returns (kept, rejected_for_overlap).
    """
    sorted_props = sorted(proposals, key=lambda p: -p.confidence)
    used: set[str] = set()
    kept: list[MergeProposal] = []
    rejected: list[MergeProposal] = []
    for p in sorted_props:
        if any(c in used for c in p.children_ids):
            rejected.append(p)
            continue
        kept.append(p)
        used.update(p.children_ids)
    return kept, rejected


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


def _window_density(
    window_ids: list[str], embeddings_by_node_id: dict[str, np.ndarray]
) -> float:
    if len(window_ids) <= 1:
        return 0.0
    mat = np.vstack([embeddings_by_node_id[node_id] for node_id in window_ids])
    sims = mat @ mat.T
    tri_i, tri_j = np.triu_indices(len(window_ids), k=1)
    if len(tri_i) == 0:
        return 0.0
    return float(np.mean(sims[tri_i, tri_j]))


def _build_overlapping_windows(
    pool: list[str],
    embeddings_by_node_id: dict[str, np.ndarray],
    *,
    window_size: int,
) -> list[list[str]]:
    """Build overlapping cosine-neighborhood windows ordered by local density."""
    if len(pool) <= window_size:
        return [list(pool)]

    emb_matrix = np.vstack([embeddings_by_node_id[node_id] for node_id in pool])
    sim = emb_matrix @ emb_matrix.T

    candidate_windows: list[tuple[float, int, list[str]]] = []
    for anchor_idx, _anchor_id in enumerate(pool):
        order = np.argsort(-sim[anchor_idx])
        order = [int(i) for i in order if int(i) != anchor_idx]
        picked = [anchor_idx] + order[: max(0, window_size - 1)]
        window_ids = [pool[i] for i in picked]
        density = _window_density(window_ids, embeddings_by_node_id)
        candidate_windows.append((density, anchor_idx, window_ids))

    candidate_windows.sort(key=lambda x: (-x[0], x[1]))
    selected: list[list[str]] = []
    covered: set[str] = set()
    target_coverage = set(pool)
    for _density, _anchor_idx, window_ids in candidate_windows:
        window_set = set(window_ids)
        if not selected:
            selected.append(window_ids)
            covered.update(window_set)
        elif window_set - covered:
            selected.append(window_ids)
            covered.update(window_set)
        if not (target_coverage - covered):
            break

    # Safety fallback: in degenerate ties/numerics, ensure full coverage.
    if target_coverage - covered:
        uncovered = list(target_coverage - covered)
        for node_id in uncovered:
            idx = pool.index(node_id)
            order = np.argsort(-sim[idx])
            order = [int(i) for i in order if int(i) != idx]
            picked = [idx] + order[: max(0, window_size - 1)]
            selected.append([pool[i] for i in picked])
            covered.add(node_id)

    return selected


def _filter_proposals(
    proposed: list[MergeProposal],
    pool_set: set[str],
    confidence_threshold: float,
) -> tuple[list[MergeProposal], list[MergeProposal], int]:
    """Validate proposals against confidence and current live pool membership."""
    valid = [
        p
        for p in proposed
        if p.confidence >= confidence_threshold
        and len(p.children_ids) >= 2
        and all(c in pool_set for c in p.children_ids)
        and len(set(p.children_ids)) == len(p.children_ids)
    ]
    kept, overlap_rejected = _resolve_overlapping_proposals(valid)
    n_invalid = len(proposed) - len(valid)
    return kept, overlap_rejected, n_invalid


def _apply_merges(
    kept: list[MergeProposal],
    *,
    nodes_by_id: dict[str, MergeNode],
    embeddings_by_node_id: dict[str, np.ndarray],
    pool: list[str],
    embeddings: np.ndarray | None,
    rng: np.random.Generator,
    iteration: int,
    paraphrases_per_node: int,
) -> None:
    """Apply accepted merges to the node graph, pool, and embedding state."""
    for k_idx, prop in enumerate(kept):
        children_nodes = [nodes_by_id[c] for c in prop.children_ids]
        parent_id = f"merge_{iteration:02d}_{k_idx:02d}"
        parent = MergeNode(
            id=parent_id,
            label=prop.parent_label,
            description=prop.parent_description,
            paraphrases=_bubble_paraphrases(
                children_nodes, rng, k=paraphrases_per_node
            ),
            size=sum(c.size for c in children_nodes),
            children=list(prop.children_ids),
            iteration_created=iteration,
            confidence=prop.confidence,
        )
        nodes_by_id[parent_id] = parent
        if embeddings is not None and all(
            c in embeddings_by_node_id for c in prop.children_ids
        ):
            child_weights = np.array([nodes_by_id[c].size for c in prop.children_ids])
            child_embs = np.vstack(
                [embeddings_by_node_id[c] for c in prop.children_ids]
            )
            weighted = np.average(child_embs, axis=0, weights=child_weights)
            embeddings_by_node_id[parent_id] = _normalize_embedding(weighted)
        for c in prop.children_ids:
            pool.remove(c)
        pool.append(parent_id)


async def pairwise_merge_loop(
    leaves: list[LeafCluster],
    *,
    embeddings: np.ndarray | None = None,
    provider: str = "mistral",
    max_iterations: int = 10,
    confidence_threshold: float = 0.6,
    min_pool_size: int = 2,
    paraphrases_per_node: int = 5,
    window_size: int = 18,
    max_concurrent_windows: int = 2,
    seed: int = 42,
) -> tuple[dict[str, MergeNode], list[str], list[dict]]:
    """Iteratively ask the LLM to merge nodes in the pool until no merge is defensible.

    Returns:
        nodes_by_id: every MergeNode created (leaves + internals), keyed by id.
        roots: ids of the final pool (forest roots).
        iteration_log: per-iteration stats (pool_size, n_proposed, n_accepted, mean_conf, stop_reason).
    """
    rng = np.random.default_rng(seed)
    nodes_by_id: dict[str, MergeNode] = {}
    embeddings_by_node_id: dict[str, np.ndarray] = {}
    for leaf in leaves:
        nodes_by_id[leaf.id] = MergeNode(
            id=leaf.id,
            label=leaf.label,
            description=leaf.description,
            paraphrases=list(leaf.paraphrases),
            size=leaf.size,
            children=[],
            iteration_created=0,
            confidence=None,
        )
        if embeddings is not None and leaf.member_indices:
            centroid = embeddings[leaf.member_indices].mean(axis=0)
            embeddings_by_node_id[leaf.id] = _normalize_embedding(centroid)
    pool: list[str] = [leaf.id for leaf in leaves]
    iteration_log: list[dict] = []

    client = create_client(provider, async_=True)

    for iteration in range(1, max_iterations + 1):
        if len(pool) < min_pool_size:
            iteration_log.append(
                {
                    "iteration": iteration,
                    "pool_size": len(pool),
                    "n_proposed": 0,
                    "n_accepted": 0,
                    "mean_confidence": None,
                    "stop_reason": "pool_below_min",
                }
            )
            logger.info("Stop: pool size %d < %d", len(pool), min_pool_size)
            break

        proposed: list[MergeProposal] = []
        if embeddings is not None and all(
            node_id in embeddings_by_node_id for node_id in pool
        ):
            windows = _build_overlapping_windows(
                pool,
                embeddings_by_node_id,
                window_size=window_size,
            )
        else:
            windows = [list(pool)]
        n_windows = len(windows)
        llm_reasoning_by_window: list[dict] = []

        sem = asyncio.Semaphore(max(1, max_concurrent_windows))

        async def _call_window(
            w_idx: int, window: list[str]
        ) -> tuple[int, list[MergeProposal], str, int] | None:
            async with sem:
                try:
                    merge_round: MergeRound = await call_structured(
                        client,
                        provider,
                        MERGE_PROPOSAL_TASK,
                        iteration=iteration,
                        pool_size=len(window),
                        pool_block=_render_pool_block([nodes_by_id[i] for i in window]),
                    )
                except Exception as e:
                    logger.warning(
                        "Merge LLM call failed at iter %d window %d: %s",
                        iteration,
                        w_idx,
                        e,
                    )
                    return None
                return w_idx, merge_round.merges, merge_round.reasoning, len(window)

        window_results = await asyncio.gather(
            *[_call_window(w_idx, window) for w_idx, window in enumerate(windows)]
        )
        ordered_results = sorted(
            [res for res in window_results if res is not None], key=lambda x: x[0]
        )
        for w_idx, merges, reasoning, window_size_local in ordered_results:
            llm_reasoning_by_window.append(
                {
                    "window_index": w_idx,
                    "window_size": window_size_local,
                    "reasoning": reasoning,
                }
            )
            proposed.extend(merges)

        proposal_support = Counter(
            tuple(sorted(prop.children_ids)) for prop in proposed
        )

        pool_set = set(pool)
        kept, overlap_rejected, n_invalid = _filter_proposals(
            proposed, pool_set, confidence_threshold
        )
        kept_merge_justifications = [
            {
                "children_ids": list(prop.children_ids),
                "parent_label": prop.parent_label,
                "justification": prop.justification,
                "confidence": float(prop.confidence),
            }
            for prop in kept
        ]

        if not kept:
            iteration_log.append(
                {
                    "iteration": iteration,
                    "pool_size": len(pool),
                    "n_proposed": len(proposed),
                    "n_accepted": 0,
                    "mean_confidence": None,
                    "n_windows": n_windows,
                    "n_invalid": n_invalid,
                    "n_overlap_rejected": len(overlap_rejected),
                    "llm_reasoning_by_window": llm_reasoning_by_window,
                    "kept_merge_justifications": kept_merge_justifications,
                    "stop_reason": "no_defensible_merge",
                }
            )
            logger.info(
                "Stop at iter %d: %d proposed, 0 accepted (threshold=%.2f)",
                iteration,
                len(proposed),
                confidence_threshold,
            )
            break

        mean_conf = float(np.mean([p.confidence for p in kept]))
        accepted_support_counts = [
            int(proposal_support[tuple(sorted(prop.children_ids))]) for prop in kept
        ]
        n_consensus_merges = sum(1 for count in accepted_support_counts if count > 1)
        n_unique_merges = sum(1 for count in accepted_support_counts if count == 1)

        pool_size_before = len(pool)
        _apply_merges(
            kept,
            nodes_by_id=nodes_by_id,
            embeddings_by_node_id=embeddings_by_node_id,
            pool=pool,
            embeddings=embeddings,
            rng=rng,
            iteration=iteration,
            paraphrases_per_node=paraphrases_per_node,
        )

        iteration_log.append(
            {
                "iteration": iteration,
                "pool_size_before": pool_size_before,
                "pool_size": len(pool),
                "n_proposed": len(proposed),
                "n_accepted": len(kept),
                "mean_confidence": mean_conf,
                "n_windows": n_windows,
                "n_invalid": n_invalid,
                "n_overlap_rejected": len(overlap_rejected),
                "accepted_support_counts": accepted_support_counts,
                "accepted_support_mean": float(np.mean(accepted_support_counts)),
                "accepted_support_min": int(min(accepted_support_counts)),
                "accepted_support_max": int(max(accepted_support_counts)),
                "n_consensus_merges": n_consensus_merges,
                "n_unique_merges": n_unique_merges,
                "llm_reasoning_by_window": llm_reasoning_by_window,
                "kept_merge_justifications": kept_merge_justifications,
                "stop_reason": None,
            }
        )
        logger.info(
            "Iter %d: %d proposed, %d accepted (mean conf %.2f), support mean/min/max %.2f/%d/%d, consensus=%d unique=%d, pool now %d",
            iteration,
            len(proposed),
            len(kept),
            mean_conf,
            float(np.mean(accepted_support_counts)),
            int(min(accepted_support_counts)),
            int(max(accepted_support_counts)),
            n_consensus_merges,
            n_unique_merges,
            len(pool),
        )
    else:
        iteration_log.append(
            {
                "iteration": max_iterations,
                "pool_size": len(pool),
                "stop_reason": "max_iterations",
            }
        )
        logger.info("Stop: max_iterations=%d reached", max_iterations)

    return nodes_by_id, list(pool), iteration_log


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_llm_tree(
    leaves: list[LeafCluster],
    nodes_by_id: dict[str, MergeNode],
    roots: list[str],
    iteration_log: list[dict],
    out_dir: Path,
) -> None:
    """Write leaves.json, tree.json, iterations.jsonl under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "leaves.json").open("w", encoding="utf-8") as f:
        json.dump(
            [leave.model_dump() for leave in leaves], f, ensure_ascii=False, indent=2
        )

    tree_payload = {
        "nodes": {nid: n.model_dump() for nid, n in nodes_by_id.items()},
        "roots": roots,
        "n_iterations": sum(1 for r in iteration_log if r.get("n_accepted")),
    }
    with (out_dir / "tree.json").open("w", encoding="utf-8") as f:
        json.dump(tree_payload, f, ensure_ascii=False, indent=2)

    with (out_dir / "iterations.jsonl").open("w", encoding="utf-8") as f:
        for row in iteration_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Saved LLM tree artefacts to %s", out_dir)


def load_llm_tree(
    in_dir: Path,
) -> tuple[list[LeafCluster], dict[str, MergeNode], list[str], list[dict]]:
    """Inverse of save_llm_tree."""
    in_dir = Path(in_dir)
    with (in_dir / "leaves.json").open(encoding="utf-8") as f:
        leaves = [LeafCluster.model_validate(d) for d in json.load(f)]
    with (in_dir / "tree.json").open(encoding="utf-8") as f:
        tree = json.load(f)
    nodes_by_id = {
        nid: MergeNode.model_validate(payload) for nid, payload in tree["nodes"].items()
    }
    roots = list(tree["roots"])
    with (in_dir / "iterations.jsonl").open(encoding="utf-8") as f:
        iteration_log = [json.loads(line) for line in f if line.strip()]
    return leaves, nodes_by_id, roots, iteration_log


def collapse_label_duplicates(tree: dict) -> list[dict]:
    """Collapse parent-child pairs sharing the exact same label.

    When a parent has a direct child with the identical label, the parent
    is removed: the matching child takes the parent's position, and the
    other siblings become direct children of that dominant child. The
    dominant's size is recomputed from its new descendants.

    Args:
        tree: Serialized tree payload with keys {"nodes", "roots"}.

    Returns:
        List of operations applied, one entry per collapsed parent.
    """
    nodes: dict[str, dict] = tree["nodes"]
    roots: list[str] = tree["roots"]

    def _subtree_size(nid: str) -> int:
        n = nodes[nid]
        if not n.get("children"):
            return n["size"]
        return sum(_subtree_size(c) for c in n["children"])

    parent_of: dict[str, str | None] = {nid: None for nid in nodes}
    for nid, node in nodes.items():
        for child_id in node.get("children", []):
            parent_of[child_id] = nid

    operations: list[dict] = []
    changed = True
    while changed:
        changed = False
        for parent_id, parent in list(nodes.items()):
            children = list(parent.get("children", []))
            if not children:
                continue
            parent_label = parent.get("label")
            dominant = next(
                (
                    cid
                    for cid in children
                    if cid in nodes and nodes[cid].get("label") == parent_label
                ),
                None,
            )
            if dominant is None:
                continue

            siblings = [c for c in children if c != dominant]

            # Dominant absorbs all siblings as direct children
            dom_children = list(nodes[dominant].get("children", []))
            nodes[dominant]["children"] = list(dict.fromkeys(dom_children + siblings))
            for s in siblings:
                parent_of[s] = dominant
            nodes[dominant]["size"] = _subtree_size(dominant)

            # Rewire dominant in place of parent
            grandparent = parent_of.get(parent_id)
            if grandparent is None:
                roots = [dominant if r == parent_id else r for r in roots]
                parent_of[dominant] = None
            else:
                gp_children = list(nodes[grandparent].get("children", []))
                nodes[grandparent]["children"] = [
                    dominant if c == parent_id else c for c in gp_children
                ]
                parent_of[dominant] = grandparent
                nodes[grandparent]["size"] = _subtree_size(grandparent)

            operations.append(
                {
                    "removed_parent": parent_id,
                    "dominant_child": dominant,
                    "absorbed": siblings,
                    "parent_label": parent_label,
                    "parent_confidence": parent.get("confidence"),
                }
            )
            del nodes[parent_id]
            parent_of.pop(parent_id, None)
            changed = True
            break

    tree["roots"] = roots
    return operations


# ─────────────────────────────────────────────────────────────────────────────
# Nested JSON export (for JSON Crack / browser graph viewers)
# ─────────────────────────────────────────────────────────────────────────────


def tree_to_nested(
    tree: dict,
    *,
    include_paraphrases: bool = True,
    drop_none: bool = True,
    sort_by_size: bool = True,
) -> dict:
    """Convert a flat ``{"nodes": {...}, "roots": [...]}`` tree payload into a
    self-nested JSON structure suitable for graph viewers like JSON Crack.

    Each node embeds its descendants directly via the ``children`` field
    (objects, not IDs), so a viewer can render the forest as an actual tree
    without resolving cross-references.

    All node metadata is preserved (``id``, ``label``, ``description``,
    ``size``, ``iteration_created``, ``confidence``, and optionally
    ``paraphrases``).

    When the source has multiple forest roots, a synthetic ``forest_root``
    wrapper is added so the result is a single root object — JSON Crack
    renders single-root payloads more cleanly.

    Args:
        tree: Serialized flat tree, as written by :func:`save_llm_tree`.
        include_paraphrases: Keep the ``paraphrases`` array on each node.
            Set ``False`` to declutter the graph view.
        drop_none: Omit metadata fields whose value is ``None`` to keep the
            rendered nodes compact.
        sort_by_size: Sort children (including the forest roots) by ``size``
            in descending order at every level. Matches the behaviour of
            :func:`src.taxonomy.hac_utils.plot_taxo_tree`.

    Returns:
        Nested JSON-serializable dict.
    """
    nodes: dict[str, dict] = tree["nodes"]
    roots: list[str] = list(tree["roots"])

    def _size_of(nid: str) -> int:
        return int(nodes[nid].get("size") or 0)

    def _node_payload(nid: str) -> dict:
        node = nodes[nid]
        payload: dict = {
            "id": node["id"],
            "label": node.get("label"),
            "description": node.get("description"),
            "size": node.get("size"),
            "iteration_created": node.get("iteration_created"),
            "confidence": node.get("confidence"),
        }
        if include_paraphrases:
            payload["paraphrases"] = node.get("paraphrases", [])
        if drop_none:
            payload = {k: v for k, v in payload.items() if v is not None}

        child_ids = list(node.get("children", []) or [])
        if sort_by_size:
            child_ids.sort(key=_size_of, reverse=True)
        if child_ids:
            payload["children"] = [_node_payload(cid) for cid in child_ids]
        return payload

    if sort_by_size:
        roots = sorted(roots, key=_size_of, reverse=True)

    children = [_node_payload(r) for r in roots]
    if len(children) == 1:
        return children[0]

    return {
        "id": "forest_root",
        "label": "forest_root",
        "description": "Synthetic outer root over the LLM-merge forest.",
        "size": sum(int(c.get("size", 0) or 0) for c in children),
        "children": children,
    }


def save_nested_tree(
    tree: dict,
    out_path: Path,
    *,
    include_paraphrases: bool = True,
    drop_none: bool = True,
    sort_by_size: bool = True,
) -> Path:
    """Write a nested-tree view of ``tree`` to ``out_path`` as pretty JSON.

    Convenience wrapper around :func:`tree_to_nested` for the common case of
    producing a file ready to drop into JSON Crack or similar viewers.
    """
    nested = tree_to_nested(
        tree,
        include_paraphrases=include_paraphrases,
        drop_none=drop_none,
        sort_by_size=sort_by_size,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)
    logger.info("Saved nested tree view to %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Bridge to TaxoNode for plot_taxo_tree reuse
# ─────────────────────────────────────────────────────────────────────────────


def merge_tree_to_taxonode(
    nodes_by_id: dict[str, MergeNode],
    roots: list[str],
    leaves: list[LeafCluster],
) -> TaxoNode:
    """Build a virtual TaxoNode tree mirroring the merge forest.

    Internal-node coherence is set to NaN (LLM-merged, not coherence-driven).
    A synthetic outer root is added if there are multiple forest roots, so
    `plot_taxo_tree` (which expects a single root) can still consume it.

    Returns:
        root: TaxoNode (synthetic if multiple roots).
    """
    leaf_indices = {leaf.id: list(leaf.member_indices) for leaf in leaves}
    leaf_coherence = {leaf.id: leaf.coherence for leaf in leaves}

    counter = {"next": 1_000_000}

    def _alloc_id() -> int:
        nid = counter["next"]
        counter["next"] += 1
        return nid

    def _build(node_id: str) -> TaxoNode:
        node = nodes_by_id[node_id]
        if not node.children:
            # leaf — reuse member indices and coherence
            indices = leaf_indices.get(node_id, [])
            coh = leaf_coherence.get(node_id, float("nan"))
            tn_id = _alloc_id()
            tn = TaxoNode(
                hac_id=tn_id,
                leaves=list(indices),
                coherence=float(coh),
                label=node.label,
                description=node.description,
                iteration_created=node.iteration_created,
                confidence=node.confidence,
            )
            # metadata already attached to `tn` via optional fields
            return tn

        children = [_build(c) for c in node.children]
        all_leaves = [idx for c in children for idx in c.leaves]
        tn_id = _alloc_id()
        tn = TaxoNode(
            hac_id=tn_id,
            leaves=all_leaves,
            coherence=float("nan"),
            children=children,
            label=node.label,
            description=node.description,
            iteration_created=node.iteration_created,
            confidence=node.confidence,
        )
        # metadata already attached to `tn` via optional fields
        return tn

    if len(roots) == 1:
        root = _build(roots[0])
    else:
        children = [_build(r) for r in roots]
        all_leaves = [idx for c in children for idx in c.leaves]
        synthetic_id = _alloc_id()
        root = TaxoNode(
            hac_id=synthetic_id,
            leaves=all_leaves,
            coherence=float("nan"),
            children=children,
        )
        # synthetic root metadata attached on the node itself
        root.label = "forest_root"
        root.description = "Synthetic root over the LLM-merge forest."
        root.iteration_created = -1
        root.confidence = None

    return root


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────


def assert_tree_invariants(
    leaves: list[LeafCluster],
    nodes_by_id: dict[str, MergeNode],
    roots: list[str],
    n_papers: int,
) -> None:
    """Raise AssertionError if the tree violates structural invariants."""
    leaf_ids = {leave.id for leave in leaves}
    # All leaves are in nodes_by_id
    missing_leaves = leaf_ids - set(nodes_by_id)
    assert not missing_leaves, f"leaves missing from nodes_by_id: {missing_leaves}"

    # Reference integrity
    for nid, node in nodes_by_id.items():
        for c in node.children:
            assert c in nodes_by_id, f"{nid} references missing child {c}"

    # No unary internals
    for nid, node in nodes_by_id.items():
        assert not node.children or len(node.children) >= 2, (
            f"{nid} has only {len(node.children)} child"
        )

    # Roots exist
    for r in roots:
        assert r in nodes_by_id, f"root {r} missing"

    # Total paper coverage from roots equals n_papers (each paper exactly once)
    leaf_member_map = {leaf.id: leaf.member_indices for leaf in leaves}
    seen: set[int] = set()

    def _collect(nid: str) -> None:
        node = nodes_by_id[nid]
        if not node.children:
            assert nid in leaf_member_map, f"terminal node {nid} not in leaves list"
            for idx in leaf_member_map[nid]:
                assert idx not in seen, f"paper {idx} appears twice"
                seen.add(idx)
            return
        for c in node.children:
            _collect(c)

    for r in roots:
        _collect(r)
    assert len(seen) == n_papers, f"covered {len(seen)} papers, expected {n_papers}"
