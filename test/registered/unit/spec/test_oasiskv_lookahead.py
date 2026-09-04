from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.speculative.oasiskv_lookahead import (
    OasisKVAttentionView,
    OasisKVLookaheadLane,
)
from sglang.srt.managers.hisparse_coordinator import OasisKVPrefetchTask


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
