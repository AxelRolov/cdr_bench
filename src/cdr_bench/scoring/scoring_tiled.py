from __future__ import annotations

import heapq
import logging
from typing import Callable

import numpy as np
from numba import njit, prange

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def tanimoto_similarity_matrix_numba(v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Numba-accelerated generalized Tanimoto similarity for dense matrices."""
    num_rows_a = v_a.shape[0]
    num_rows_b = v_b.shape[0]
    similarity_matrix = np.empty((num_rows_a, num_rows_b), dtype=np.float32)

    sum_a_squared = np.sum(np.square(v_a), axis=1)
    sum_b_squared = np.sum(np.square(v_b), axis=1)

    for i in prange(num_rows_a):
        for j in prange(num_rows_b):
            numerator = np.dot(v_a[i], v_b[j])
            denominator = sum_a_squared[i] + sum_b_squared[j] - numerator
            if denominator <= 0.0:
                similarity_matrix[i, j] = 0.0
            else:
                similarity_matrix[i, j] = numerator / denominator

    return similarity_matrix


_DEFAULT_TANIMOTO_BACKEND = "cdr_bench_numba"


def resolve_tanimoto_backend(backend: str | None) -> str:
    """Resolve a backend name to a concrete implementation."""
    name = (backend or "auto").lower()
    if name == "auto":
        return _DEFAULT_TANIMOTO_BACKEND
    if name in {"cdr_bench", "cdr_bench_blas", "cdr_bench_numba", "cdr_bench_fused"}:
        return _DEFAULT_TANIMOTO_BACKEND
    if name in {"gpu", "torch", "torch_cuda"}:
        if torch is None or not torch.cuda.is_available():
            logger.warning(
                "Requested backend '%s' but CUDA torch is unavailable; using numpy.",
                backend,
            )
            return "numpy"
        return "torch_cuda"
    if name in {"numpy", "cpu"}:
        return "numpy"
    raise ValueError(f"Unknown Tanimoto backend: {backend!r}")


def compute_fp_squared_norms(fps: np.ndarray) -> np.ndarray:
    """Precompute ||fp_i||^2 for a fingerprint matrix."""
    return np.einsum("ij,ij->i", fps, fps, dtype=np.float32).astype(np.float32)


def topk_1d(
    scores: np.ndarray,
    rows: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k rows and scores for a single similarity vector."""
    effective_k = min(k, len(scores))
    if effective_k <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    if effective_k == len(scores):
        order = np.argsort(scores)[::-1]
        return rows[order].astype(np.int32), scores[order].astype(np.float32)
    top_local = np.argpartition(scores, -effective_k)[-effective_k:]
    top_local = top_local[np.argsort(scores[top_local])[::-1]]
    return rows[top_local].astype(np.int32), scores[top_local].astype(np.float32)


def topk_matrix(
    scores: np.ndarray,
    rows: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized top-k over rows of a similarity matrix."""
    if scores.ndim != 2:
        raise ValueError("scores must be a 2-D array")
    n_rows, n_cols = scores.shape
    effective_k = min(k, n_cols)
    if effective_k <= 0:
        return (
            np.empty((n_rows, 0), dtype=np.int32),
            np.empty((n_rows, 0), dtype=np.float32),
        )
    if effective_k == n_cols:
        top_local = np.argsort(scores, axis=1)[:, ::-1]
        top_scores = np.take_along_axis(scores, top_local, axis=1)
    else:
        top_local = np.argpartition(scores, -effective_k, axis=1)[:, -effective_k:]
        top_scores = np.take_along_axis(scores, top_local, axis=1)
        order = np.argsort(top_scores, axis=1)[:, ::-1]
        top_local = np.take_along_axis(top_local, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)

    if rows.ndim == 1:
        top_rows = rows[top_local]
    else:
        top_rows = np.take_along_axis(rows, top_local, axis=1)
    return top_rows.astype(np.int32), top_scores.astype(np.float32)


def tanimoto_similarity_matrix_batch(
    queries: np.ndarray,
    candidates: np.ndarray,
    *,
    candidate_sq: np.ndarray | None = None,
    query_sq: np.ndarray | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Compute pairwise generalized Tanimoto between query and candidate blocks."""
    resolved_backend = resolve_tanimoto_backend(backend)

    if resolved_backend == "torch_cuda":
        assert torch is not None  # guarded by resolve_tanimoto_backend
        device = torch.device("cuda")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        candidates = np.ascontiguousarray(candidates, dtype=np.float32)
        queries_t = torch.from_numpy(queries).to(device=device)
        candidates_t = torch.from_numpy(candidates).to(device=device)
        dot = queries_t @ candidates_t.T
        if query_sq is None:
            q_sq_t = (queries_t * queries_t).sum(dim=1, keepdim=True)
        else:
            q_sq_t = torch.from_numpy(
                np.ascontiguousarray(query_sq, dtype=np.float32).reshape(-1, 1)
            ).to(device=device)
        if candidate_sq is None:
            candidate_sq_t = (candidates_t * candidates_t).sum(dim=1)
        else:
            candidate_sq_t = torch.from_numpy(
                np.ascontiguousarray(candidate_sq, dtype=np.float32)
            ).to(device=device)
        denom = torch.clamp(q_sq_t + candidate_sq_t - dot, min=1.0)
        return (dot / denom).to(dtype=torch.float32, device="cpu").numpy()

    if resolved_backend.startswith("cdr_bench"):
        return tanimoto_similarity_matrix_numba(queries, candidates)

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    candidates = np.ascontiguousarray(candidates, dtype=np.float32)
    dot = queries @ candidates.T
    if query_sq is None:
        query_sq = (queries * queries).sum(axis=1, keepdims=False)
    query_sq = np.ascontiguousarray(query_sq, dtype=np.float32).reshape(-1, 1)
    if candidate_sq is None:
        candidate_sq = (candidates * candidates).sum(axis=1, keepdims=False)
    else:
        candidate_sq = np.ascontiguousarray(candidate_sq, dtype=np.float32)
    denom = np.where(query_sq + candidate_sq - dot > 0.0, query_sq + candidate_sq - dot, np.float32(1.0))
    return (dot / denom).astype(np.float32)


def exact_tanimoto_topk_from_block(
    query_fp: np.ndarray,
    cand_fps: np.ndarray,
    candidate_rows: np.ndarray,
    k: int,
    *,
    candidate_sq: np.ndarray | None = None,
    backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k against a preloaded candidate block."""
    if len(candidate_rows) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    query = np.ascontiguousarray(query_fp[np.newaxis, :], dtype=np.float32)
    query_sq = np.asarray([float(np.dot(query[0], query[0]))], dtype=np.float32)
    sims = tanimoto_similarity_matrix_batch(
        query,
        np.ascontiguousarray(cand_fps, dtype=np.float32),
        query_sq=query_sq,
        candidate_sq=candidate_sq,
        backend=backend,
    )[0]
    return topk_1d(sims, candidate_rows, k)


def streaming_exact_topk(
    query_fp: np.ndarray,
    candidate_rows: np.ndarray,
    k: int,
    *,
    block_loader: Callable[[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    tile_rows: int = 10_000,
    log_every_tiles: int = 0,
    log_label: str | None = None,
    backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream exact top-k with O(N log K) merge cost."""
    total_rows = len(candidate_rows)
    effective_k = min(k, total_rows)
    if effective_k <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    tile_rows = max(1, int(tile_rows))
    total_tiles = (total_rows + tile_rows - 1) // tile_rows
    heap: list[tuple[float, int]] = []

    for tile_idx, start in enumerate(range(0, total_rows, tile_rows), start=1):
        end = min(start + tile_rows, total_rows)
        rows_tile, fps_tile, sq_tile = block_loader(start, end)
        sims_tile = tanimoto_similarity_matrix_batch(
            np.ascontiguousarray(query_fp[np.newaxis, :], dtype=np.float32),
            fps_tile,
            query_sq=np.asarray([float(np.dot(query_fp, query_fp))], dtype=np.float32),
            candidate_sq=sq_tile,
            backend=backend,
        )[0]
        local_rows, local_scores = topk_1d(sims_tile, rows_tile, effective_k)
        for row_id, score in zip(local_rows.tolist(), local_scores.tolist()):
            entry = (float(score), int(row_id))
            if len(heap) < effective_k:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)

        if log_every_tiles > 0 and (
            tile_idx == 1 or tile_idx % log_every_tiles == 0 or tile_idx == total_tiles
        ):
            retained_floor = heap[0][0] if heap else float("nan")
            label = f"{log_label}: " if log_label else ""
            logger.info(
                "%sstreaming exact top-k tile %d/%d rows=%d/%d retained_floor=%.4f",
                label,
                tile_idx,
                total_tiles,
                end,
                total_rows,
                retained_floor,
            )

    top_items = sorted(heap, reverse=True)
    top_scores = np.asarray([score for score, _ in top_items], dtype=np.float32)
    top_rows = np.asarray([row_id for _, row_id in top_items], dtype=np.int32)
    return top_rows, top_scores
