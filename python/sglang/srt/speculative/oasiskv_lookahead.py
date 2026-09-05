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


def paired_batch_from_eagle_verify(
    verify_input: Any, batch_size: int
) -> OasisKVPairedBatch:
    """Adopt EAGLE's real root+one-draft verify tensors as an OasisKV pair."""
    if verify_input.draft_token_num != 2:
        raise ValueError("OasisKV requires exactly one normal and one draft token")
    rows = torch.arange(2 * batch_size, device=verify_input.draft_token.device)
    if verify_input.draft_token.numel() != rows.numel():
        raise ValueError("EAGLE verify tokens do not have request-major 2B layout")
    return OasisKVPairedBatch(
        input_ids=verify_input.draft_token,
        positions=verify_input.positions,
        normal_rows=rows[0::2],
        draft_rows=rows[1::2],
        draft_valid=torch.ones(
            batch_size, dtype=torch.bool, device=verify_input.draft_token.device
        ),
    )


def configure_oasiskv_forward_batch(
    forward_batch: Any, paired: OasisKVPairedBatch
) -> None:
    """Install paired geometry on the real :class:`ForwardBatch`.

    Production decode arrives through ``eagle_prepare_for_verify`` and therefore
    carries tree/position metadata in ``spec_info`` rather than the
    ``extend_prefix_lens``/``extend_start_loc`` fields used by a plain EXTEND.
    Either representation supplies the kernel-level ``history -> normal ->
    draft`` causal relation; no detached Python mask is used by production.
    """
    if forward_batch.batch_size != paired.batch_size:
        raise ValueError("OasisKV request count differs from ForwardBatch")
    if forward_batch.input_ids.numel() != 2 * paired.batch_size:
        raise ValueError("OasisKV ForwardBatch must contain two tokens per request")
    verify_width = getattr(
        getattr(forward_batch, "spec_info", None), "draft_token_num", None
    )
    extend_is_paired = (
        list(forward_batch.extend_seq_lens_cpu or ()) == [2] * paired.batch_size
    )
    if not extend_is_paired and verify_width != 2:
        raise ValueError("OasisKV paired forward requires two tokens per request")
    required_2b = {"out_cache_loc": forward_batch.out_cache_loc}
    if extend_is_paired:
        required_2b.update(
            extend_prefix_lens=getattr(forward_batch, "extend_prefix_lens", None),
            extend_start_loc=getattr(forward_batch, "extend_start_loc", None),
        )
    else:
        spec_info = forward_batch.spec_info
        required_2b.update(
            verify_draft_token=getattr(spec_info, "draft_token", None),
            verify_positions=getattr(spec_info, "positions", None),
        )
    missing = [name for name, value in required_2b.items() if value is None]
    if missing:
        raise ValueError(f"OasisKV ForwardBatch lacks metadata: {', '.join(missing)}")
    if forward_batch.out_cache_loc.numel() != 2 * paired.batch_size:
        raise ValueError("OasisKV requires one normal and one scratch cache location")
    if forward_batch.seq_lens is None or forward_batch.seq_lens_cpu is None:
        raise ValueError("OasisKV requires GPU and CPU sequence lengths")
    forward_batch.input_ids = paired.input_ids
    forward_batch.positions = paired.positions
    forward_batch.is_oasiskv_paired = True
    forward_batch.oasiskv_normal_rows = paired.normal_rows
    forward_batch.oasiskv_draft_rows = paired.draft_rows
    forward_batch.oasiskv_draft_valid = paired.draft_valid
