from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np

_cp = None
_cupy_error: Exception | None = None

if find_spec("cupy") is not None:
    try:
        import cupy as _cp  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        _cp = None
        _cupy_error = exc


def has_gpu_support() -> bool:
    """Return True when the optional CuPy dependency is importable."""
    return _cp is not None


def require_cupy():
    """Return CuPy or raise a clear runtime error when the gpu extra is absent."""
    if _cp is None:
        raise RuntimeError(
            "GPU scoring requires the optional 'gpu' extra. "
            "Install it with: uv sync --extra gpu"
        ) from _cupy_error
    return _cp


@dataclass(frozen=True)
class CuPyQueryHandle:
    """Prepared query tensors for one streamed search request."""

    query_fp: object
    query_sq: object
    k: int


class CuPyKernelBridge:
    """CuPy-based kernel bridge skeleton for the streaming GPU path."""

    BLOCK_THREADS = 128
    MAX_KERNEL_TOPK = 32
    KERNEL_TEMPLATE = r"""
    #define BLOCK_THREADS %(block_threads)d
    #define MAX_TOPK %(max_topk)d

    extern "C" __device__ inline void insert_topk(
        float score,
        int row_id,
        float* scores,
        int* ids,
        int k
    ) {
        if (k <= 0) {
            return;
        }
        if (score <= scores[0]) {
            return;
        }
        scores[0] = score;
        ids[0] = row_id;
        int pos = 0;
        while (pos + 1 < k && scores[pos] > scores[pos + 1]) {
            float score_tmp = scores[pos];
            scores[pos] = scores[pos + 1];
            scores[pos + 1] = score_tmp;
            int id_tmp = ids[pos];
            ids[pos] = ids[pos + 1];
            ids[pos + 1] = id_tmp;
            ++pos;
        }
    }

    extern "C" __global__ void fused_tanimoto_topk_kernel(
        const float* fps,
        const float* fps_sq,
        const float* query,
        float query_sq,
        int n_rows,
        int fp_dim,
        int k,
        float* out_scores,
        int* out_ids
    ) {
        const int tid = threadIdx.x;
        __shared__ float shared_scores[BLOCK_THREADS * MAX_TOPK];
        __shared__ int shared_ids[BLOCK_THREADS * MAX_TOPK];
        __shared__ float block_scores[MAX_TOPK];
        __shared__ int block_ids[MAX_TOPK];

        float local_scores[MAX_TOPK];
        int local_ids[MAX_TOPK];
        for (int i = 0; i < MAX_TOPK; ++i) {
            local_scores[i] = -3.402823e38f;
            local_ids[i] = -1;
        }

        for (int row = tid; row < n_rows; row += BLOCK_THREADS) {
            const float* cand = fps + ((size_t)row * (size_t)fp_dim);
            float dot = 0.0f;
            for (int d = 0; d < fp_dim; ++d) {
                dot += cand[d] * query[d];
            }
            float denom = query_sq + fps_sq[row] - dot;
            if (denom <= 0.0f) {
                denom = 1.0f;
            }
            float sim = dot / denom;
            insert_topk(sim, row, local_scores, local_ids, k);
        }

        const int base = tid * MAX_TOPK;
        for (int i = 0; i < MAX_TOPK; ++i) {
            shared_scores[base + i] = local_scores[i];
            shared_ids[base + i] = local_ids[i];
        }
        __syncthreads();

        if (tid == 0) {
            for (int i = 0; i < MAX_TOPK; ++i) {
                block_scores[i] = -3.402823e38f;
                block_ids[i] = -1;
            }
            for (int t = 0; t < BLOCK_THREADS; ++t) {
                const int off = t * MAX_TOPK;
                for (int j = 0; j < k; ++j) {
                    int row_id = shared_ids[off + j];
                    if (row_id >= 0) {
                        insert_topk(shared_scores[off + j], row_id, block_scores, block_ids, k);
                    }
                }
            }
            for (int i = 0; i < k; ++i) {
                out_scores[i] = block_scores[k - 1 - i];
                out_ids[i] = block_ids[k - 1 - i];
            }
        }
    }
    """

    def __init__(self, *, rerank_factor: int = 4) -> None:
        self.cp = require_cupy()
        self.rerank_factor = max(1, int(rerank_factor))
        self._module = self.cp.RawModule(
            code=self.KERNEL_TEMPLATE
            % {
                "block_threads": self.BLOCK_THREADS,
                "max_topk": self.MAX_KERNEL_TOPK,
            },
            options=("--std=c++11",),
            name_expressions=("fused_tanimoto_topk_kernel",),
        )
        self._fused_kernel = self._module.get_function("fused_tanimoto_topk_kernel")

    def prepare_query(self, query_fp: np.ndarray, k: int) -> CuPyQueryHandle:
        """Upload one query to the device and precompute its squared norm."""
        cp = self.cp
        query_dev = cp.asarray(np.ascontiguousarray(query_fp, dtype=np.float32))
        query_sq = cp.sum(query_dev * query_dev, dtype=cp.float32)
        return CuPyQueryHandle(query_fp=query_dev, query_sq=query_sq, k=max(1, int(k)))

    def enqueue_tile(
        self,
        slot: object,
        *,
        query_handle: CuPyQueryHandle,
        tile_rows: int,
        rerank_width: int,
    ) -> None:
        """Schedule H2D copy, CuPy scoring, and tiny D2H top-k export for one tile."""
        cp = self.cp
        local_k = min(max(1, int(rerank_width)), tile_rows)
        slot.local_topk = local_k

        with slot.copy_stream:
            slot.dev_fp[:tile_rows].set(slot.host_fp[:tile_rows], stream=slot.copy_stream)
            if slot.dev_sq is not None and slot.host_sq is not None:
                slot.dev_sq[:tile_rows].set(slot.host_sq[:tile_rows], stream=slot.copy_stream)
            slot.copy_done.record(slot.copy_stream)

        with slot.compute_stream:
            slot.compute_stream.wait_event(slot.copy_done)
            if local_k <= self.MAX_KERNEL_TOPK:
                self._fused_kernel(
                    (1,),
                    (self.BLOCK_THREADS,),
                    (
                        slot.dev_fp,
                        slot.dev_sq,
                        query_handle.query_fp,
                        np.float32(query_handle.query_sq.item()),
                        np.int32(tile_rows),
                        np.int32(query_handle.query_fp.shape[0]),
                        np.int32(local_k),
                        slot.dev_topk_scores,
                        slot.dev_topk_ids,
                    ),
                )
            else:
                self._enqueue_tile_fallback(
                    slot,
                    query_handle=query_handle,
                    tile_rows=tile_rows,
                    local_k=local_k,
                )
            slot.compute_done.record(slot.compute_stream)

            cp.asnumpy(slot.dev_topk_ids[:local_k], out=slot.host_topk_ids[:local_k], stream=slot.compute_stream)
            cp.asnumpy(slot.dev_topk_scores[:local_k], out=slot.host_topk_scores[:local_k], stream=slot.compute_stream)
            slot.d2h_done.record(slot.compute_stream)

    def _enqueue_tile_fallback(
        self,
        slot: object,
        *,
        query_handle: CuPyQueryHandle,
        tile_rows: int,
        local_k: int,
    ) -> None:
        """Fallback CuPy path for rerank widths larger than the fused kernel supports."""
        cp = self.cp
        dev_fp = slot.dev_fp[:tile_rows]
        if slot.dev_sq is not None:
            dev_sq = slot.dev_sq[:tile_rows]
        else:
            dev_sq = cp.sum(dev_fp * dev_fp, axis=1, dtype=cp.float32)

        dot = dev_fp @ query_handle.query_fp
        denom = cp.maximum(query_handle.query_sq + dev_sq - dot, cp.float32(1.0))
        sims = dot / denom

        top_idx = cp.argpartition(sims, -local_k)[-local_k:]
        top_scores = sims[top_idx]
        order = cp.argsort(top_scores)[::-1]
        slot.dev_topk_ids[:local_k] = top_idx[order].astype(cp.int32)
        slot.dev_topk_scores[:local_k] = top_scores[order].astype(cp.float32)

    def is_slot_complete(self, slot: object) -> bool:
        """Return True when D2H export for one tile-local top-k is finished."""
        return bool(slot.d2h_done.done)

    def collect_tile_topk(self, slot: object) -> tuple[np.ndarray, np.ndarray]:
        """Map tile-local indices back to global row ids on the host."""
        local_k = int(slot.local_topk)
        local_ids = np.asarray(slot.host_topk_ids[:local_k], dtype=np.int32)
        local_scores = np.asarray(slot.host_topk_scores[:local_k], dtype=np.float32)
        row_ids = np.asarray(slot.row_ids[local_ids], dtype=np.int32)
        return row_ids, local_scores
