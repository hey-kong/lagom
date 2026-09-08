import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_ema_update_forecast_uses_request_indexed_state():
    from sglang.kernels.ops.attention.dsv4.ema import ema_update_forecast

    device = torch.device("cuda")
    levels = torch.zeros((8, 4), dtype=torch.float32, device=device)
    trends = torch.zeros_like(levels)
    state_lens = torch.zeros(8, dtype=torch.int32, device=device)
    prediction_lens = torch.empty(2, dtype=torch.int32, device=device)
    forecast = torch.empty((2, 4), dtype=torch.float32, device=device)
    reqs = torch.tensor([5, 2], dtype=torch.int32, device=device)
    lengths = torch.tensor([4, 3], dtype=torch.int32, device=device)
    first = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 99.0]], device=device)
    first_copy = first.clone()

    result = ema_update_forecast(
        first,
        reqs,
        lengths,
        levels,
        trends,
        state_lens,
        prediction_lens,
        forecast,
        alpha=0.6,
        beta=0.2,
        gamma=0.25,
    )
    torch.testing.assert_close(first, first_copy)
    torch.testing.assert_close(result[0], first[0])
    torch.testing.assert_close(result[1, :3], first[1, :3])
    assert torch.isneginf(result[1, 3])
    assert prediction_lens.tolist() == [0, 0]

    # Reverse rows to prove state follows request ids instead of batch rows.
    second = torch.tensor([[8.0, 7.0, 6.0, 88.0], [2.0, 4.0, 6.0, 8.0]], device=device)
    ema_update_forecast(
        second,
        torch.tensor([2, 5], dtype=torch.int32, device=device),
        torch.tensor([3, 4], dtype=torch.int32, device=device),
        levels,
        trends,
        state_lens,
        prediction_lens,
        forecast,
        alpha=0.6,
        beta=0.2,
        gamma=0.25,
    )
    old_for_req_2 = first[1, :3]
    expected_level = 0.6 * second[0, :3] + 0.4 * old_for_req_2
    expected_trend = 0.2 * (second[0, :3] - old_for_req_2)
    torch.testing.assert_close(forecast[0, :3], expected_level + 0.25 * expected_trend)
    assert prediction_lens.tolist() == [3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_prefetcher_accepts_separate_cpu_and_device_request_indices():
    from sglang.srt.managers.hisparse_prefetcher import EMAPrefetcher

    prefetcher = EMAPrefetcher(logical_entries=2)
    scores = torch.tensor([[1.0, 2.0]], device="cuda")
    assert (
        prefetcher.update(
            scores,
            torch.tensor([2]),
            [5],
            layer_id=0,
            req_pool_indices_device=torch.tensor([5], dtype=torch.int32, device="cuda"),
            seq_lens_device=torch.tensor([2], dtype=torch.int32, device="cuda"),
        )
        is None
    )
