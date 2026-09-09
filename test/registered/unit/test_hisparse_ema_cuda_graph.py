"""CPU-only lifecycle tests for HiSparse EMA decode CUDA Graph integration."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.hisparse_prefetcher import (
    EMAPrefetcher,
    HiSparsePrefetchStats,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class _Event:
    def __init__(self):
        self.waited_on = None
        self.synchronized = False

    def wait(self, stream):
        self.waited_on = stream

    def synchronize(self):
        self.synchronized = True


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


def test_request_release_drains_ema_before_shared_tables_are_cleared():
    event = _Event()
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.prefetcher_name = "ema"
    coordinator.prefetcher = SimpleNamespace(stats=HiSparsePrefetchStats())
    coordinator._previous_prefetch_event = event
    coordinator._previous_prefetch_pending_entries = 8
    coordinator._previous_prefetch_target_layer = 1
    coordinator._ema_graph_work_pending = True

    coordinator._drain_named_prefetch_before_request_release()

    assert event.synchronized
    assert coordinator._previous_prefetch_pending_entries == 0
    assert coordinator._previous_prefetch_target_layer is None
    assert not coordinator._ema_graph_work_pending


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


def test_previous_graph_replay_joins_last_writer(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as coordinator_module

    event = _Event()
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.prefetcher_name = "previous"
    coordinator.prefetcher = SimpleNamespace(stats=HiSparsePrefetchStats())
    coordinator._previous_prefetch_event = event
    coordinator._previous_prefetch_pending_entries = 12
    coordinator._previous_prefetch_target_layer = 4
    monkeypatch.setattr(
        coordinator_module.device_module, "current_stream", lambda: "compute"
    )

    coordinator.consume_previous_graph_prefetch()

    assert event.waited_on == "compute"
    assert coordinator._previous_prefetch_pending_entries == 0
    assert coordinator._previous_prefetch_target_layer is None
    assert coordinator.prefetcher.stats.completed_h2d_entries == 12


def test_graph_replay_submits_previous_candidates_to_following_layers():
    calls = []
    coordinator = SimpleNamespace(
        prefetcher_name="previous",
        mem_pool_device=SimpleNamespace(layer_num=4),
        _submit_previous_prefetch_to_layer=lambda *args: calls.append(args),
    )
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(hisparse_coordinator=coordinator)
    runner._replay_graph_key = "bs4"
    runner._previous_graph_prefetch = {
        "bs4": {
            1: {
                "candidates": torch.arange(8).view(4, 2),
                "compressed_seq_lens": torch.tensor([8, 7, 1, 1]),
                "source_layer_id": 1,
            },
            # The final layer has no consumer and must not launch useless IO.
            3: {
                "candidates": torch.arange(8).view(4, 2),
                "compressed_seq_lens": torch.tensor([8, 7, 1, 1]),
                "source_layer_id": 3,
            },
        }
    }
    batch = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.tensor([4, 9, 0, 0]),
    )

    runner._submit_previous_graph_prefetch(batch)

    assert len(calls) == 1
    req_pool_indices, seq_lens, candidates, target_layer = calls[0]
    assert torch.equal(req_pool_indices, torch.tensor([4, 9]))
    assert torch.equal(seq_lens, torch.tensor([8, 7]))
    assert torch.equal(candidates, torch.tensor([[0, 1], [2, 3]]))
    assert target_layer == 2


def test_graph_replay_submits_live_batch_with_captured_scores():
    calls = []
    coordinator = SimpleNamespace(
        prefetcher_name="ema",
        prefetcher=EMAPrefetcher(logical_entries=2),
        num_real_reqs=torch.zeros(1, dtype=torch.int32),
        consume_ema_prefetch=lambda: calls.append(("consume",)),
        submit_ema_prefetch=lambda **kwargs: calls.append(("submit", kwargs)),
    )
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(hisparse_coordinator=coordinator)
    runner._replay_graph_key = "bs4"
    scores = torch.arange(1200, dtype=torch.float32).view(4, 300)
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

    # A subsequent replay/batch preparation may immediately overwrite every
    # source tensor; submitted task snapshots must remain unchanged.
    scores.zero_()
    compressed_lens.zero_()
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
    assert kwargs["batch_metadata"].slots == (4, 9)
    assert kwargs["batch_metadata"].lengths == (6, 5)
    assert kwargs["num_real_reqs"].item() == 2
    assert kwargs["layer_id"] == 2

    snapshot_ptrs = {
        name: kwargs[name].data_ptr()
        for name in (
            "req_pool_indices",
            "compressed_seq_lens",
            "scores",
            "num_real_reqs",
        )
    }
    batch.req_pool_indices.copy_(torch.tensor([7, 8]))
    runner._submit_ema_graph_prefetch(batch)
    next_kwargs = calls[-1][1]
    assert all(
        next_kwargs[name].data_ptr() == pointer
        for name, pointer in snapshot_ptrs.items()
    )
