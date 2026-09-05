"""OasisKV normal/draft paired-forward primitives.

OasisKV is look-ahead, not speculative decoding: the odd rows are disposable
target-model probes and can never be accepted as output.  This module keeps the
row/position/state contract in one place so model and attention backends do not
have to infer ownership from tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


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


def configure_oasiskv_forward_batch(
    forward_batch: Any, paired: OasisKVPairedBatch
) -> None:
    """Install paired geometry on the real :class:`ForwardBatch`.

    The batch must already have been prepared as a two-token extend for every
    request.  That existing attention representation supplies the kernel-level
    ``history -> normal -> draft`` causal relation; no detached Python mask is
    used by production.
    """
    if forward_batch.batch_size != paired.batch_size:
        raise ValueError("OasisKV request count differs from ForwardBatch")
    if forward_batch.input_ids.numel() != 2 * paired.batch_size:
        raise ValueError("OasisKV ForwardBatch must contain two tokens per request")
    if list(forward_batch.extend_seq_lens_cpu or ()) != [2] * paired.batch_size:
        raise ValueError("OasisKV paired forward requires a two-token extend layout")
    forward_batch.input_ids = paired.input_ids
    forward_batch.positions = paired.positions
    forward_batch.is_oasiskv_paired = True
    forward_batch.oasiskv_normal_rows = paired.normal_rows
    forward_batch.oasiskv_draft_rows = paired.draft_rows
    forward_batch.oasiskv_draft_valid = paired.draft_valid
