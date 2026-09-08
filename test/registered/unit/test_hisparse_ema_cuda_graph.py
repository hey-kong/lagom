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


def test_graph_replay_accounts_previous_ema_writer():
    event = _Event()
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.prefetcher_name = "ema"
    coordinator.prefetcher = SimpleNamespace(stats=HiSparsePrefetchStats())
    coordinator._previous_prefetch_event = event
    coordinator._previous_prefetch_pending_entries = 17
    coordinator._previous_prefetch_target_layer = 3
    coordinator._ema_graph_work_pending = True
    coordinator.consume_ema_prefetch()

    assert event.waited_on is None
    assert coordinator._previous_prefetch_pending_entries == 0
    assert coordinator._previous_prefetch_target_layer is None
    assert not coordinator._ema_graph_work_pending
    assert coordinator.prefetcher.stats.completed_h2d_entries == 17


def test_captured_layer_waits_only_for_its_ema_writer(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as coordinator_module

    events = [_Event(), _Event()]
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.prefetcher_name = "ema"
    coordinator._ema_layer_events = events
    monkeypatch.setattr(
        coordinator_module.device_module, "current_stream", lambda: "compute"
    )

    coordinator._consume_previous_prefetch(torch.tensor([0]), layer_id=1)

    assert events[0].waited_on is None
    assert events[1].waited_on == "compute"


def test_graph_replay_submits_live_batch_with_captured_scores():
    calls = []
    coordinator = SimpleNamespace(
        prefetcher_name="ema",
        num_real_reqs=torch.zeros(1, dtype=torch.int32),
        consume_ema_prefetch=lambda: calls.append(("consume",)),
        submit_ema_prefetch=lambda **kwargs: calls.append(("submit", kwargs)),
    )
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(hisparse_coordinator=coordinator)
    runner._replay_graph_key = "bs4"
    scores = torch.arange(1200, dtype=torch.float32).view(4, 300)
    compressed_lens = torch.tensor([6, 5, 1, 1])
    page_table = torch.arange(1200, dtype=torch.int32).view(4, 300)
    runner._ema_graph_prefetch = {
        "bs4": {
            2: {
                "scores": scores,
                "compressed_seq_lens": compressed_lens,
                "page_table": page_table,
                "page_size": 1,
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

    # A subsequent replay/batch preparation may immediately overwrite every
    # source tensor; submitted task snapshots must remain unchanged.
    scores.zero_()
    compressed_lens.zero_()
    page_table.zero_()
    batch.req_pool_indices.fill_(99)
    batch.req_pool_indices_cpu.fill_(99)

    assert calls[0] == ("consume",)
    kwargs = calls[1][1]
    assert torch.equal(kwargs["req_pool_indices"], torch.tensor([4, 9]))
    assert torch.equal(kwargs["req_pool_indices_cpu"], torch.tensor([4, 9]))
    assert torch.equal(
        kwargs["scores"], torch.arange(1200, dtype=torch.float32).view(4, 300)[:2, :256]
    )
    assert torch.equal(kwargs["compressed_seq_lens"], torch.tensor([6, 5]))
    assert torch.equal(kwargs["compressed_seq_lens_cpu"], torch.tensor([6, 5]))
    assert torch.equal(
        kwargs["page_table"],
        torch.arange(1200, dtype=torch.int32).view(4, 300)[:2, :256],
    )
    assert kwargs["page_size"] == 1
    assert kwargs["num_real_reqs"].item() == 2
    assert kwargs["layer_id"] == 2
