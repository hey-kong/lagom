"""CPU-only lifecycle tests for HiSparse EMA decode CUDA Graph integration."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.hisparse_prefetcher import HiSparsePrefetchStats
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class _Event:
    def __init__(self):
        self.waited_on = None

    def wait(self, stream):
        self.waited_on = stream


def test_graph_replay_joins_previous_ema_writer(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as coordinator_module

    event = _Event()
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.prefetcher_name = "ema"
    coordinator.prefetcher = SimpleNamespace(stats=HiSparsePrefetchStats())
    coordinator._previous_prefetch_event = event
    coordinator._previous_prefetch_pending_entries = 17
    coordinator._previous_prefetch_target_layer = 3
    monkeypatch.setattr(
        coordinator_module.device_module, "current_stream", lambda: "compute"
    )

    coordinator.consume_ema_prefetch()

    assert event.waited_on == "compute"
    assert coordinator._previous_prefetch_pending_entries == 0
    assert coordinator._previous_prefetch_target_layer is None
    assert coordinator.prefetcher.stats.completed_h2d_entries == 17


def test_graph_replay_submits_live_batch_with_captured_scores():
    calls = []
    coordinator = SimpleNamespace(
        prefetcher_name="ema",
        consume_ema_prefetch=lambda: calls.append(("consume",)),
        submit_ema_prefetch=lambda **kwargs: calls.append(("submit", kwargs)),
    )
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(hisparse_coordinator=coordinator)
    runner._replay_graph_key = "bs4"
    scores = torch.arange(24, dtype=torch.float32).view(4, 6)
    compressed_lens = torch.tensor([6, 5, 1, 1])
    runner._ema_graph_prefetch = {
        "bs4": {
            2: {
                "scores": scores,
                "compressed_seq_lens": compressed_lens,
                "layer_id": 2,
            }
        }
    }
    batch = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.tensor([4, 9]),
        req_pool_indices_cpu=torch.tensor([4, 9]),
        seq_lens_cpu=torch.tensor([24, 20]),
    )

    runner._prepare_ema_graph_replay()
    runner._submit_ema_graph_prefetch(batch)

    assert calls[0] == ("consume",)
    kwargs = calls[1][1]
    assert kwargs["req_pool_indices"] is batch.req_pool_indices
    assert kwargs["req_pool_indices_cpu"] is batch.req_pool_indices_cpu
    assert torch.equal(kwargs["scores"], scores[:2])
    assert torch.equal(kwargs["compressed_seq_lens"], compressed_lens[:2])
    assert torch.equal(kwargs["compressed_seq_lens_cpu"], torch.tensor([6, 5]))
    assert kwargs["layer_id"] == 2
