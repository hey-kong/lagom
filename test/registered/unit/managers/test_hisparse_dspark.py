from types import MethodType

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator


def _coordinator_with_recording_kernel():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.top_k = 2
    calls = []

    def run(self, req, seq_len, top_k, layer_id, *, output_buffer, **_kwargs):
        calls.append((int(req[0]), int(seq_len[0]), top_k[0].tolist(), layer_id))
        output_buffer.copy_(top_k.to(torch.int32) + 100)
        return output_buffer

    coordinator._run_swap_in_kernel = MethodType(run, coordinator)
    return coordinator, calls


@pytest.mark.parametrize(
    ("verify_lens", "seq_lens", "expected_req_order"),
    [
        (None, [10, 10, 20, 20], [7, 7, 9, 9]),
        ([1, 3], [10, 20], [7, 9, 9, 9]),
    ],
)
def test_dspark_swap_in_is_request_major(
    verify_lens, seq_lens, expected_req_order
):
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

    assert [call[0] for call in calls] == expected_req_order
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, top_k.to(torch.int32) + 100)


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
