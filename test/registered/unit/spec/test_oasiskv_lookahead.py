from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.speculative.oasiskv_lookahead import (
    OasisKVAttentionView,
    OasisKVFeatureStore,
    OasisKVLayerOutput,
    OasisKVLookaheadLane,
    OasisKVPairedBatch,
    OasisKVPairedOutput,
    OasisKVPairedTargetExecutor,
    OasisKVScratchState,
    OasisKVStateTransaction,
    paired_causal_mask,
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


def test_paired_forward_generates_target_query_and_commits_no_draft_state():
    draft_state = {"scratch": []}
    calls = []
    submitted = []
    shared_pages = object()
    store = OasisKVFeatureStore()
    store.put(2, 8, "prefill-normal-features")

    def paired_forward(batch):
        calls.append(batch)
        assert batch.positions == (9, 10)
        return OasisKVPairedOutput(
            normal_output=43,
            normal_features=["next-normal-features"],
            draft_queries={5: "target-layer-5-draft-query"},
            normal_sparse_locations={5: shared_pages},
            draft_sparse_locations={5: shared_pages},
        )

    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: [42],
        paired_target_forward=paired_forward,
        target_c4_predict=lambda layer, query: [(layer, query)],
        submit_prefetch=lambda **kwargs: submitted.append(kwargs),
        snapshot_draft_state=lambda: repr(draft_state),
        feature_store=store,
    )
    view = OasisKVAttentionView("same-c4", "same-c128", "same-swa", "same-local", 9)
    normal_output, predictions = lane.run(
        normal_tokens=[7],
        attention_view=view,
        request_slots=[2],
        generations=[8],
        request_metadata={"slot": 2, "generation": 8},
    )
    assert normal_output == 43
    assert calls[0].draft_tokens == [42]
    assert calls[0].attention_view is view
    assert predictions[0].c4_entries == [(5, "target-layer-5-draft-query")]
    assert predictions[0].target_token_positions == 10
    assert draft_state == {"scratch": []}
    assert store.get(2, 8) == "next-normal-features"
    assert submitted[0]["prediction"].layer_id == 5
    assert lane.profile.paired_decode_steps == 1
    assert lane.profile.draft_tokens_generated == 1
    assert lane.profile.target_layer_traversals == 1
    assert lane.profile.draft_c4_predictions == 1
    assert lane.profile.speculative_acceptance == 0


def test_leaking_draft_scratch_is_rejected():
    state = []
    store = OasisKVFeatureStore()
    store.put(0, 0, "feature")
    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: [1],
        paired_target_forward=lambda _: state.append(1)
        or OasisKVPairedOutput(None, [None], {}, {}, {}),
        target_c4_predict=lambda *_: [],
        submit_prefetch=lambda **_: None,
        snapshot_draft_state=lambda: tuple(state),
        feature_store=store,
    )
    with pytest.raises(RuntimeError, match="leaked draft scratch"):
        lane.run(
            normal_tokens=[1],
            attention_view=OasisKVAttentionView(None, None, None, None, 1),
            request_slots=[0],
            generations=[0],
            request_metadata={},
        )


def test_missing_feature_skips_draft_and_slot_reuse_clears_stale_feature():
    store = OasisKVFeatureStore()
    store.put(3, 1, "old")
    store.put(3, 2, "new")
    assert store.get(3, 1) is None
    paired_calls = []
    normal_calls = []
    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: pytest.fail("must not synthesize a draft"),
        paired_target_forward=lambda batch: paired_calls.append(batch),
        target_c4_predict=lambda *_: [],
        submit_prefetch=lambda **_: None,
        snapshot_draft_state=lambda: (),
        feature_store=store,
        normal_target_forward=lambda **kwargs: normal_calls.append(kwargs)
        or ("normal-output", ["fresh-3", "fresh-9"]),
    )
    output, predictions = lane.run(
        normal_tokens=[1, 2],
        attention_view=OasisKVAttentionView(
            None, None, None, None, torch.tensor([4, 8])
        ),
        request_slots=[3, 9],
        generations=[2, 0],
        request_metadata={},
    )
    assert output == "normal-output" and predictions == [] and paired_calls == []
    assert len(normal_calls) == 1
    assert store.get(3, 2) == "fresh-3"
    assert store.get(9, 0) == "fresh-9"


def test_different_draft_sparse_pages_are_rejected():
    store = OasisKVFeatureStore()
    store.put(0, 0, "feature")
    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: [2],
        paired_target_forward=lambda _: OasisKVPairedOutput(
            None, ["f"], {0: "q"}, {0: object()}, {0: object()}
        ),
        target_c4_predict=lambda *_: [],
        submit_prefetch=lambda **_: None,
        snapshot_draft_state=lambda: (),
        feature_store=store,
    )
    with pytest.raises(RuntimeError, match="different sparse working set"):
        lane.run(
            normal_tokens=[1],
            attention_view=OasisKVAttentionView(None, None, None, None, 4),
            request_slots=[0],
            generations=[0],
            request_metadata={},
        )


def test_equal_independently_constructed_sparse_tensors_are_accepted():
    store = OasisKVFeatureStore()
    store.put(0, 0, "feature")
    lane = OasisKVLookaheadLane(
        eagle3_draft_one=lambda **_: [2],
        paired_target_forward=lambda _: OasisKVPairedOutput(
            "normal",
            ["next"],
            {0: torch.tensor([[4]])},
            {0: torch.tensor([[1, 3]])},
            {0: torch.tensor([[1, 3]])},
        ),
        target_c4_predict=lambda _, query: query,
        submit_prefetch=lambda **_: None,
        snapshot_draft_state=lambda: (),
        feature_store=store,
    )
    output, predictions = lane.run(
        normal_tokens=[1],
        attention_view=OasisKVAttentionView(None, None, None, None, 4),
        request_slots=[0],
        generations=[0],
        request_metadata={},
    )
    assert output == "normal" and len(predictions) == 1


def test_state_transaction_rejects_authoritative_target_layer_writes():
    authoritative = {"kv": [], "c4": [], "pages": [], "host": [], "lens": [4]}
    transaction = OasisKVStateTransaction(
        scratch=OasisKVScratchState([], [], [], [], [5], []),
        snapshot_authoritative=lambda: repr(authoritative),
        commit_normal=lambda state, **_: state,
    )
    authoritative["kv"].append("draft-write")
    with pytest.raises(RuntimeError, match="authoritative state"):
        transaction.commit_normal("normal", batch=None)


def test_reference_causal_mask_and_single_layer_traversal():
    mask = paired_causal_mask([2, 4])
    assert mask[0, 0].tolist() == [True, True, True, False, False, False]
    assert mask[0, 1].tolist() == [True, True, True, True, False, False]
    calls = []
    shared = [object(), object()]

    def run_layer(*, layer, hidden_states, batch, causal_mask):
        calls.append((layer, hidden_states, causal_mask))
        return OasisKVLayerOutput(hidden_states, f"target-q-{layer}", shared[layer])

    executor = OasisKVPairedTargetExecutor(
        layers=[0, 1],
        run_layer=run_layer,
        commit_normal=lambda hidden, **_: (hidden, ["a", "b"]),
    )
    output = executor(
        OasisKVPairedBatch(
            request_slots=[7, 9],
            generations=[1, 3],
            normal_tokens=[10, 11],
            draft_tokens=[12, 13],
            committed_lens=[2, 4],
            positions=([2, 4], [3, 5]),
            attention_view=OasisKVAttentionView(None, None, None, None, [2, 4]),
        )
    )
    assert len(calls) == executor.layer_calls == 2
    assert output.draft_queries == {0: "target-q-0", 1: "target-q-1"}
    assert output.normal_sparse_locations[0] is output.draft_sparse_locations[0]


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
