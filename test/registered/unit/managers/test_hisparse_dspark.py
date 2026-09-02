from types import MethodType

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    dspark_completed_c4_positions,
    dspark_fixed_buffer_commit_indices,
)


@pytest.mark.parametrize(
    ("prefix_len", "expected"),
    [(8, [2]), (9, [2]), (10, [2, 3]), (11, [2, 3])],
)
def test_dspark_c4_scratch_crosses_alignment(prefix_len, expected):
    """Every prefix alignment must bind each C4 writer completed by six verify rows."""
    assert dspark_completed_c4_positions(prefix_len, verify_width=6) == expected


def test_dspark_long_commit_has_unique_reserved_destination():
    """Two accepted long C4s from one request must copy only the newest one."""
    assert dspark_fixed_buffer_commit_indices(
        accepted=[0, 1, 2],
        req_offsets=[0, 0, 1],
        req_pool_indices_cpu=[7, 9],
        c4_positions=[4096, 4097, 4098],
        device_buffer_size=4096,
    ) == [1, 2]


def test_dspark_graph_scratch_metadata_updates_in_place():
    """Replay must consume updated metadata without changing captured addresses."""
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator._dspark_scratch_compressed_locs = torch.full(
        (4,), -1, dtype=torch.int64
    )
    coordinator._dspark_scratch_valid_mapping = torch.zeros(16, dtype=torch.bool)
    compressed = torch.tensor([10, 11, 12], dtype=torch.int64)
    swapped = torch.tensor([100, 101, 102], dtype=torch.int32)
    writers = torch.tensor([200, 201, 202], dtype=torch.int32)

    capture_result = coordinator.select_dspark_scratch_locs(
        compressed, swapped, writers
    )
    torch.testing.assert_close(capture_result, swapped)

    metadata_ptr = coordinator._dspark_scratch_compressed_locs.data_ptr()
    coordinator._dspark_scratch_compressed_locs[:2].copy_(compressed[[0, 2]])
    coordinator._dspark_scratch_valid_mapping[compressed[[0, 2]]] = True
    replay_result = coordinator.select_dspark_scratch_locs(
        compressed, swapped, writers
    )

    assert coordinator._dspark_scratch_compressed_locs.data_ptr() == metadata_ptr
    torch.testing.assert_close(replay_result, torch.tensor([200, 101, 202]))


def _coordinator_with_recording_kernel():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.top_k = 2
    coordinator.device = "cpu"
    calls = []

    def run(self, req, seq_len, top_k, layer_id, *, output_buffer, **_kwargs):
        calls.append((req.tolist(), seq_len.tolist(), top_k.tolist(), layer_id))
        output_buffer.copy_(top_k.to(torch.int32) + 100)
        return output_buffer

    coordinator._run_swap_in_kernel = MethodType(run, coordinator)
    return coordinator, calls


@pytest.mark.parametrize(
    ("verify_lens", "seq_lens", "expected_launches"),
    [
        (None, [10, 10, 20, 20], [[7, 9], [7, 9]]),
        ([1, 3], [10, 20], [[7, 9], [9], [9]]),
        (None, [[10, 10], [20, 20]], [[7, 9], [7, 9]]),
    ],
)
def test_dspark_swap_in_is_request_major(verify_lens, seq_lens, expected_launches):
    """A batched kernel would race LRU updates from two steps of one request."""
    coordinator, calls = _coordinator_with_recording_kernel()
    top_k = torch.arange(8, dtype=torch.int64).view(4, 2)
    output = torch.full((4, 2), -1, dtype=torch.int32)

    actual = coordinator.swap_in_selected_pages_spec(
        req_pool_indices=torch.tensor([7, 9]),
        compressed_seq_lens=torch.tensor(seq_lens),
        top_k_result=top_k,
        layer_id=3,
        verify_lens_cpu=verify_lens,
        output_buffer=output,
    )

    assert [call[0] for call in calls] == expected_launches
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, top_k.to(torch.int32) + 100)


def test_dspark_swap_in_ignores_graph_padding_rows():
    """Padding rows must not launch a block or mutate request slot zero."""
    coordinator, calls = _coordinator_with_recording_kernel()
    top_k = torch.arange(12, dtype=torch.int64).view(6, 2)
    output = torch.empty((6, 2), dtype=torch.int32)

    actual = coordinator.swap_in_selected_pages_spec(
        req_pool_indices=torch.tensor([7, 9]),
        compressed_seq_lens=torch.tensor([10, 20]),
        top_k_result=top_k,
        layer_id=3,
        verify_lens_cpu=[1, 3],
        output_buffer=output,
    )

    assert [call[0] for call in calls] == [[7, 9], [9], [9]]
    torch.testing.assert_close(actual[:4], top_k[:4].to(torch.int32) + 100)
    torch.testing.assert_close(actual[4:], torch.full((2, 2), -1, dtype=torch.int32))


def test_dspark_swap_in_rejects_inconsistent_ragged_geometry():
    """Bad compact geometry must fail instead of associating Top-K with another request."""
    coordinator, _ = _coordinator_with_recording_kernel()
    with pytest.raises(ValueError, match="geometry does not match"):
        coordinator.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([7, 9]),
            compressed_seq_lens=torch.tensor([10, 20]),
            top_k_result=torch.zeros((4, 2), dtype=torch.int64),
            layer_id=3,
            verify_lens_cpu=[1, 2],
        )


def test_dspark_swap_in_rejects_inconsistent_sequence_length_geometry():
    """A short seq-len tensor would make the CUDA kernel read past its allocation."""
    coordinator, _ = _coordinator_with_recording_kernel()
    with pytest.raises(ValueError, match="sequence lengths"):
        coordinator.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([7, 9]),
            compressed_seq_lens=torch.tensor([10, 20, 30]),
            top_k_result=torch.zeros((4, 2), dtype=torch.int64),
            layer_id=3,
        )
