"""CPU-only correctness tests for DeepSeek-V4 HiSparse chain-EAGLE commits."""

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseDSparkWindow,
    dspark_completed_c4_positions,
    speculative_accepted_c4_indices,
)


def _window(prefix_lens, req_slots, width=4):
    req_offsets = []
    c4_positions = []
    for offset, prefix_len in enumerate(prefix_lens):
        positions = dspark_completed_c4_positions(prefix_len, width)
        req_offsets.extend([offset] * len(positions))
        c4_positions.extend(positions)
    count = len(c4_positions)
    return HiSparseDSparkWindow(
        compressed_locs=torch.arange(count),
        device_locs=torch.arange(100, 100 + count),
        previous_device_mapping=torch.full((count,), -1),
        req_offsets=req_offsets,
        req_pool_indices_cpu=req_slots,
        c4_positions=c4_positions,
        prefix_lens_cpu=prefix_lens,
    )


@pytest.mark.parametrize(
    ("prefix_len", "commit_len", "expected"),
    [
        (8, 1, []),  # all drafts rejected: verified root is not a full C4
        (10, 1, []),  # boundary reached in scratch, but not committed
        (10, 2, [0]),  # partial acceptance completes exactly one C4
        (11, 4, [0]),  # later C4 remains incomplete
        (11, 5, [0, 1]),  # verify spans and commits two C4 groups
    ],
)
def test_eagle_commit_uses_accepted_length_at_c4_boundaries(
    prefix_len, commit_len, expected
):
    window = _window([prefix_len], [7], width=5)
    assert speculative_accepted_c4_indices(window, [commit_len]) == expected


def test_eagle_dynamic_batch_slots_keep_request_local_commit_lengths():
    # Slot order is deliberately non-monotonic and models reuse of scheduler
    # slots after an earlier request finishes.  Commit decisions are indexed by
    # batch offset, never by the reusable global request-slot number.
    window = _window([10, 7, 11], [19, 2, 5], width=4)
    assert speculative_accepted_c4_indices(window, [2, 1, 1]) == [0, 1, 2]


def test_eagle_commit_rejects_mismatched_batch_metadata():
    window = _window([10, 7], [4, 9])
    with pytest.raises(ValueError, match="commit geometry mismatch"):
        speculative_accepted_c4_indices(window, [2])


def test_eagle_cuda_graph_scratch_replay_keeps_addresses_and_changes_identity():
    """EAGLE replay updates values, never graph-captured tensor addresses."""
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator._dspark_scratch_compressed_locs = torch.full(
        (4,), -1, dtype=torch.int64
    )
    coordinator._dspark_scratch_valid_mapping = torch.zeros(32, dtype=torch.bool)
    logical_c4 = torch.tensor([6, 7, 8], dtype=torch.int64)
    resident = torch.tensor([16, 17, 18], dtype=torch.int32)
    scratch = torch.tensor([26, 27, 28], dtype=torch.int32)

    identity_ptr = coordinator._dspark_scratch_compressed_locs.data_ptr()
    validity_ptr = coordinator._dspark_scratch_valid_mapping.data_ptr()
    torch.testing.assert_close(
        coordinator.select_dspark_scratch_locs(logical_c4, resident, scratch),
        resident,
    )

    # Model one replay's prepare phase. The graph sees the same buffers but a
    # different transaction identity, including a non-contiguous C4 selection.
    coordinator._dspark_scratch_compressed_locs[:2].copy_(logical_c4[[0, 2]])
    coordinator._dspark_scratch_valid_mapping[logical_c4[[0, 2]]] = True
    torch.testing.assert_close(
        coordinator.select_dspark_scratch_locs(logical_c4, resident, scratch),
        torch.tensor([26, 17, 28], dtype=torch.int32),
    )
    assert coordinator._dspark_scratch_compressed_locs.data_ptr() == identity_ptr
    assert coordinator._dspark_scratch_valid_mapping.data_ptr() == validity_ptr
