from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.speculative.oasiskv_lookahead import (
    OasisKVAttentionView,
    OasisKVLookaheadLane,
)
from sglang.srt.managers.hisparse_coordinator import OasisKVPrefetchTask
from sglang.srt.managers.hisparse_prefetcher import HiSparsePrefetchStats


def _args(**overrides):
    values = dict(
        hisparse_config='{"prefetcher":"oasiskv"}',
        enable_hisparse=True,
        speculative_draft_model_path="eagle3-checkpoint",
        speculative_algorithm=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        is_oasiskv_lookahead=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_oasiskv_resolves_dedicated_lookahead_mode():
    args = _args()
    _handle_oasiskv_lookahead(args)
    assert args.is_oasiskv_lookahead
    assert args.speculative_algorithm is None
    assert (args.speculative_num_steps, args.speculative_eagle_topk) == (1, 1)


def test_oasiskv_requires_draft_path_and_rejects_spec_verification():
    with pytest.raises(ValueError, match="draft-model-path"):
        _handle_oasiskv_lookahead(_args(speculative_draft_model_path=None))
    with pytest.raises(ValueError, match="LOOKAHEAD_ONLY"):
        _handle_oasiskv_lookahead(_args(speculative_algorithm="EAGLE3"))
    with pytest.raises(ValueError, match="num-steps=1"):
        _handle_oasiskv_lookahead(_args(speculative_num_steps=2))


def test_draft_query_is_target_generated_and_state_is_not_committed():
    state = {"tokens": [7], "kv": [11], "page_table": [3]}
    calls = []
    submitted = []

    def target_no_commit(**kwargs):
        calls.append(kwargs)
        return {5: "target-layer-5-query"}

    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: 42,
        target_no_commit=target_no_commit,
        target_c4_predict=lambda layer, query: [(layer, query)],
        submit_prefetch=lambda **kwargs: submitted.append(kwargs),
        snapshot_committed_state=lambda: repr(state),
    )
    view = OasisKVAttentionView("same-c4", "same-c128", "same-swa", "same-local", 9)
    predictions = lane.run(
        normal_hidden_states="normal",
        attention_view=view,
        request_metadata={"slot": 2, "generation": 8},
    )
    assert calls[0]["token"] == 42
    assert calls[0]["attention_view"] is view
    assert calls[0]["commit"] is False and calls[0]["scratch_kv"] is True
    assert predictions[0].c4_entries == [(5, "target-layer-5-query")]
    assert predictions[0].target_token_positions == 10
    assert state == {"tokens": [7], "kv": [11], "page_table": [3]}
    assert submitted[0]["prediction"].layer_id == 5


def test_mutating_draft_forward_is_rejected():
    state = []
    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: 1,
        target_no_commit=lambda **_: state.append(1) or {},
        target_c4_predict=lambda *_: [],
        submit_prefetch=lambda **_: None,
        snapshot_committed_state=lambda: tuple(state),
    )
    with pytest.raises(RuntimeError, match="mutated committed state"):
        lane.run(
            normal_hidden_states=None,
            attention_view=OasisKVAttentionView(None, None, None, None, 1),
            request_metadata={},
        )


def test_prefetch_identity_rejects_slot_generation_and_position_reuse():
    task = OasisKVPrefetchTask(
        layer_id=3,
        ring_slot=1,
        req_slots=torch.tensor([2, 7]),
        generations=torch.tensor([4, 9]),
        source_lens=torch.tensor([10, 20]),
        target_positions=torch.tensor([11, 21]),
        predicted_entries=None,
        device_locs=None,
        miss_src=None,
        miss_dst=None,
        miss_count=None,
        event=None,
        valid=True,
    )

    def matches(slots=(2, 7), generations=(4, 9), lens=(11, 21)):
        values = torch.tensor(lens)
        return task.matches(
            layer_id=3,
            req_slots=torch.tensor(slots),
            generations=torch.tensor(generations),
            committed_lens=values,
            token_positions=values,
        )

    assert matches()
    assert not matches(slots=(7, 2))  # same batch size, different requests
    assert not matches(generations=(5, 9))  # scheduler slot was reused
    assert not matches(lens=(12, 21))  # decode advanced past the prediction


class _FakeEvent:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def wait(self, _stream):
        self.calls.append(("wait", self.name))

    def synchronize(self):
        self.calls.append(("synchronize", self.name))


def _task(name, calls, *, slots, generations, source_lens, ring_slot):
    return OasisKVPrefetchTask(
        layer_id=0,
        ring_slot=ring_slot,
        req_slots=torch.tensor(slots),
        generations=torch.tensor(generations),
        source_lens=torch.tensor(source_lens),
        target_positions=torch.tensor(source_lens) + 1,
        predicted_entries=None,
        device_locs=None,
        miss_src=None,
        miss_dst=None,
        miss_count=None,
        event=_FakeEvent(name, calls),
        submitted_entries=8,
        valid=True,
    )


def test_consume_finds_second_slot_and_waits_every_layer_writer(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as coordinator_module

    calls = []
    coordinator = object.__new__(coordinator_module.HiSparseCoordinator)
    coordinator.prefetcher_name = "oasiskv"
    coordinator.prefetcher = SimpleNamespace(stats=HiSparsePrefetchStats())
    coordinator._prefetch_generation = torch.tensor([0, 0, 4])
    stale = _task(
        "stale", calls, slots=[1], generations=[0], source_lens=[10], ring_slot=0
    )
    current = _task(
        "current", calls, slots=[2], generations=[4], source_lens=[10], ring_slot=1
    )
    coordinator._oasiskv_ring = [[stale, current]]
    monkeypatch.setattr(
        coordinator_module.device_module, "current_stream", lambda: "compute"
    )

    coordinator._consume_previous_prefetch(
        torch.tensor([2], device="cpu"),
        0,
        req_pool_indices_cpu=torch.tensor([2]),
        committed_lens_cpu=torch.tensor([11]),
    )

    assert calls == [("wait", "stale"), ("wait", "current")]
    assert not stale.valid and not current.valid
    assert coordinator.prefetcher.stats.completed_h2d_entries == 8
    assert coordinator.prefetcher.stats.stale_tasks == 1


def test_ring_rotation_and_request_drain_use_cpu_event_sync():
    import sglang.srt.managers.hisparse_coordinator as coordinator_module

    calls = []
    coordinator = object.__new__(coordinator_module.HiSparseCoordinator)
    coordinator.prefetcher = SimpleNamespace(stats=HiSparsePrefetchStats())
    first = _task(
        "first", calls, slots=[3], generations=[0], source_lens=[1], ring_slot=0
    )
    second = _task(
        "second", calls, slots=[4], generations=[0], source_lens=[1], ring_slot=1
    )
    coordinator._oasiskv_ring = [[first, second]]
    coordinator._oasiskv_next_slot = [0]

    assert coordinator._acquire_oasiskv_ring_task(0) is first
    assert coordinator._acquire_oasiskv_ring_task(0) is second
    coordinator._drain_oasiskv_tasks_for_request(3)

    assert calls == [("synchronize", "first")]
    assert not first.valid and second.valid
