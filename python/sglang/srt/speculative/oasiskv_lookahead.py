"""OasisKV normal/draft paired-forward primitives.

OasisKV is look-ahead, not speculative decoding: the odd rows are disposable
target-model probes and can never be accepted as output.  This module keeps the
row/position/state contract in one place so model and attention backends do not
have to infer ownership from tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import torch


@dataclass(frozen=True)
class OasisKVAttentionView:
    """Immutable sparse history selected by the normal query."""

    c4_sparse_locations: Any
    c128_metadata: Any
    swa_metadata: Any
    local_window_metadata: Any
    committed_sequence_lengths: Any


@dataclass(frozen=True)
class OasisKVPrediction:
    layer_id: int
    source_committed_sequence_lengths: Any
    target_token_positions: Any
    c4_entries: Any


@dataclass
class OasisKVScratch:
    """Per-forward storage for draft KV/C4; deliberately has no commit API."""

    kv: dict[int, Any] = field(default_factory=dict)
    c4: dict[int, Any] = field(default_factory=dict)
    released: bool = False

    def put_kv(self, layer_id: int, value: Any) -> None:
        if self.released:
            raise RuntimeError("OasisKV scratch has been released")
        self.kv[layer_id] = value

    def put_c4(self, layer_id: int, value: Any) -> None:
        if self.released:
            raise RuntimeError("OasisKV scratch has been released")
        self.c4[layer_id] = value

    def release(self) -> None:
        self.kv.clear()
        self.c4.clear()
        self.released = True

    def __enter__(self) -> "OasisKVScratch":
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


@dataclass(frozen=True)
class OasisKVPairedBatch:
    """Request-major ``[normal0,draft0,normal1,draft1,...]`` batch metadata."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    normal_rows: torch.Tensor
    draft_rows: torch.Tensor
    draft_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.normal_rows.numel()

    def causal_mask(self) -> torch.Tensor:
        """Inline-token visibility; history is supplied by the sparse backend.

        Rows are independent requests. Normal sees itself only, while draft sees
        normal and itself. No row may see another request.
        """
        mask = torch.zeros(
            (2 * self.batch_size, 2 * self.batch_size),
            dtype=torch.bool,
            device=self.input_ids.device,
        )
        mask[self.normal_rows, self.normal_rows] = True
        mask[self.draft_rows, self.normal_rows] = True
        mask[self.draft_rows, self.draft_rows] = True
        return mask


def build_oasiskv_paired_batch(
    normal_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    committed_lens: torch.Tensor,
    draft_valid: Optional[torch.Tensor] = None,
) -> OasisKVPairedBatch:
    """Interleave lanes without changing committed sequence lengths."""
    if normal_tokens.ndim != 1 or draft_tokens.shape != normal_tokens.shape:
        raise ValueError("normal and draft tokens must be equal-length 1-D tensors")
    if committed_lens.shape != normal_tokens.shape:
        raise ValueError("committed_lens must have one entry per request")
    bsz = normal_tokens.numel()
    rows = torch.arange(2 * bsz, device=normal_tokens.device)
    input_ids = torch.stack((normal_tokens, draft_tokens), dim=1).reshape(-1)
    # The normal token is the current committed position.  The draft is the
    # following position and is never reflected back into committed_lens.
    positions = torch.stack((committed_lens, committed_lens + 1), dim=1).reshape(-1)
    if draft_valid is None:
        draft_valid = torch.ones(bsz, dtype=torch.bool, device=normal_tokens.device)
    if draft_valid.shape != normal_tokens.shape:
        raise ValueError("draft_valid must have one entry per request")
    return OasisKVPairedBatch(input_ids, positions, rows[0::2], rows[1::2], draft_valid)


class OasisKVFeatureStore:
    """Generation-tagged EAGLE feature state safe against scheduler slot reuse."""

    def __init__(self):
        self._features: dict[int, tuple[int, Any]] = {}

    def get(self, slot: int, generation: int) -> Any | None:
        item = self._features.get(slot)
        return item[1] if item is not None and item[0] == generation else None

    def update(self, slot: int, generation: int, feature: Any) -> None:
        self._features[slot] = (generation, feature)

    def finish(self, slot: int, generation: int) -> None:
        if slot in self._features and self._features[slot][0] == generation:
            del self._features[slot]


class OasisKVPairedForward:
    """Execute each real target layer once over the request-major 2B tensor.

    ``layer_forward`` is the model's actual decoder-layer call (not a second
    target callback). ``project_draft_q`` runs the layer's real C4 Q projection.
    The same normal-selected attention view is passed to every layer; draft C4
    is only used to prefetch the *next* step.
    """

    def __init__(self, submit_prefetch: Callable[..., None]):
        self.submit_prefetch = submit_prefetch

    def run(
        self,
        *,
        paired: OasisKVPairedBatch,
        hidden_states: torch.Tensor,
        layers: Iterable[tuple[int, Any]],
        layer_forward: Callable[..., tuple[torch.Tensor, Any]],
        project_draft_q: Callable[[Any, torch.Tensor, torch.Tensor], Any],
        scan_c4: Callable[[Any, OasisKVAttentionView], Any],
        attention_view: OasisKVAttentionView,
        request_metadata: Mapping[str, Any],
    ) -> tuple[torch.Tensor, list[OasisKVPrediction]]:
        if hidden_states.shape[0] != paired.input_ids.shape[0]:
            raise ValueError("paired target hidden state must contain exactly 2B rows")
        predictions: list[OasisKVPrediction] = []
        with OasisKVScratch() as scratch:
            for layer_id, layer in layers:
                # One invocation is essential: it owns QKV, attention and FFN/MoE
                # for both lanes and writes odd-row KV only to scratch.
                hidden_states, draft_kv = layer_forward(
                    layer,
                    hidden_states,
                    paired,
                    attention_view,
                    scratch,
                )
                scratch.put_kv(layer_id, draft_kv)
                draft_q = project_draft_q(layer, hidden_states[paired.draft_rows], paired.positions[paired.draft_rows])
                entries = scan_c4(draft_q, attention_view)
                scratch.put_c4(layer_id, entries)
                prediction = OasisKVPrediction(
                    layer_id,
                    attention_view.committed_sequence_lengths,
                    paired.positions[paired.draft_rows],
                    entries,
                )
                self.submit_prefetch(
                    prediction=prediction,
                    request_metadata=request_metadata,
                    draft_valid=paired.draft_valid,
                )
                predictions.append(prediction)
        return hidden_states[paired.normal_rows], predictions


def submit_oasiskv_prediction(
    coordinator: Any,
    *,
    prediction: OasisKVPrediction,
    request_metadata: Mapping[str, Any],
    draft_valid: torch.Tensor,
) -> None:
    """Submit valid draft C4 rows to PR10's real per-layer prefetch ring.

    Requests without a previous EAGLE feature still execute the normal lane,
    but are omitted from this prediction.  Tensor filtering is applied to all
    identities together so dynamic batching cannot associate a prediction with
    another scheduler slot.
    """
    keep = draft_valid.to(dtype=torch.bool)
    gpu_slots = request_metadata["req_pool_indices"][keep]
    cpu_slots = request_metadata["req_pool_indices_cpu"][keep.cpu()]
    compressed_lens = request_metadata["compressed_seq_lens"][keep]
    source_lens = prediction.source_committed_sequence_lengths[keep.cpu()]
    entries = prediction.c4_entries[keep]
    if gpu_slots.numel() == 0:
        return
    coordinator.submit_oasiskv_prefetch(
        req_pool_indices=gpu_slots,
        req_pool_indices_cpu=cpu_slots,
        compressed_seq_lens=compressed_lens,
        source_committed_lens_cpu=source_lens,
        predicted_c4_entries=entries,
        layer_id=prediction.layer_id,
    )


# Kept as an explicit error instead of silently retaining PR11's two-forward
# callback abstraction. Production callers must use OasisKVPairedForward.
class OasisKVLookaheadLane:
    def __init__(self, **_: Any):
        raise RuntimeError("OasisKVLookaheadLane was replaced by shared 2B paired forward")
