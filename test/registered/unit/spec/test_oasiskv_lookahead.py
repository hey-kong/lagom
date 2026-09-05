from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.speculative.oasiskv_lookahead import (
    OasisKVAttentionView,
    OasisKVFeatureStore,
    OasisKVPairedForward,
    OasisKVScratch,
    build_oasiskv_paired_batch,
    submit_oasiskv_prediction,
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


def test_paired_row_mapping_positions_and_causal_relation():
    paired = build_oasiskv_paired_batch(
        torch.tensor([10, 20]), torch.tensor([11, 21]), torch.tensor([7, 13])
    )
    assert paired.input_ids.tolist() == [10, 11, 20, 21]
    assert paired.positions.tolist() == [7, 8, 13, 14]
    assert paired.normal_rows.tolist() == [0, 2]
    assert paired.draft_rows.tolist() == [1, 3]
    assert paired.causal_mask().tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


def test_real_layer_objects_run_once_over_2b_and_draft_is_scratch_only():
    class Layer(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            assert hidden.shape[0] == 4
            return hidden + self.value

    layers = [(3, Layer(1)), (4, Layer(2))]
    submitted = []
    scratch_refs = []
    paired = build_oasiskv_paired_batch(
        torch.tensor([10, 20]), torch.tensor([11, 21]), torch.tensor([7, 13])
    )
    view = OasisKVAttentionView("normal-c4", None, None, None, torch.tensor([7, 13]))

    def layer_forward(layer, hidden, metadata, attention_view, scratch):
        assert attention_view.c4_sparse_locations == "normal-c4"
        assert isinstance(scratch, OasisKVScratch)
        scratch_refs.append(scratch)
        return layer(hidden), hidden[metadata.draft_rows].clone()

    normal, predictions = OasisKVPairedForward(
        lambda **kw: submitted.append(kw)
    ).run(
        paired=paired,
        hidden_states=torch.arange(4.0)[:, None],
        layers=layers,
        layer_forward=layer_forward,
        project_draft_q=lambda layer, hidden, positions: hidden + positions[:, None],
        scan_c4=lambda query, attention_view: query.topk(1, dim=0).indices.T,
        attention_view=view,
        request_metadata={"slots": [2, 7]},
    )
    assert [layer.calls for _, layer in layers] == [1, 1]
    assert normal[:, 0].tolist() == [3, 5]
    assert len(predictions) == len(submitted) == 2
    assert all(ref.released and not ref.kv and not ref.c4 for ref in scratch_refs)


def test_feature_fallback_dynamic_batch_and_slot_reuse():
    store = OasisKVFeatureStore()
    store.update(4, 1, "old")
    assert store.get(4, 1) == "old"
    assert store.get(4, 2) is None  # reused scheduler slot cannot inherit feature
    store.update(4, 2, "new")
    store.update(9, 0, "dynamic-batch-peer")
    store.finish(4, 1)  # stale completion must not delete the new owner
    assert store.get(4, 2) == "new"
    store.finish(4, 2)
    assert store.get(4, 2) is None


def test_draft_c4_is_submitted_to_real_ring_api_with_fallback_filtered():
    coordinator = SimpleNamespace(submit_oasiskv_prefetch=lambda **kw: calls.append(kw))
    calls = []
    prediction = SimpleNamespace(
        layer_id=6,
        source_committed_sequence_lengths=torch.tensor([10, 20]),
        c4_entries=torch.tensor([[1, 2], [3, 4]]),
    )
    submit_oasiskv_prediction(
        coordinator,
        prediction=prediction,
        request_metadata={
            "req_pool_indices": torch.tensor([5, 8]),
            "req_pool_indices_cpu": torch.tensor([5, 8]),
            "compressed_seq_lens": torch.tensor([2, 4]),
        },
        draft_valid=torch.tensor([False, True]),
    )
    assert len(calls) == 1
    assert calls[0]["layer_id"] == 6
    assert calls[0]["req_pool_indices_cpu"].tolist() == [8]
    assert calls[0]["predicted_c4_entries"].tolist() == [[3, 4]]


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
