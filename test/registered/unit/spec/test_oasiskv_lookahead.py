from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.speculative.oasiskv_lookahead import (
    build_oasiskv_paired_batch,
    configure_oasiskv_forward_batch,
    paired_batch_from_eagle_verify,
)
from sglang.srt.managers.hisparse_coordinator import (
    OasisKVPrefetchTask,
    is_hisparse_prefetcher_mode_unsupported,
)
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
    assert args.speculative_algorithm == "EAGLE3"
    assert (args.speculative_num_steps, args.speculative_eagle_topk) == (1, 1)
    assert args.speculative_num_draft_tokens == 2


def test_oasiskv_requires_draft_path_and_rejects_spec_verification():
    with pytest.raises(ValueError, match="draft-model-path"):
        _handle_oasiskv_lookahead(_args(speculative_draft_model_path=None))
    with pytest.raises(ValueError, match="LOOKAHEAD_ONLY"):
        _handle_oasiskv_lookahead(_args(speculative_algorithm="EAGLE3"))
    with pytest.raises(ValueError, match="num-steps=1"):
        _handle_oasiskv_lookahead(_args(speculative_num_steps=2))


def test_oasiskv_allows_internal_speculative_scratch_but_still_rejects_pp():
    assert not is_hisparse_prefetcher_mode_unsupported(
        "oasiskv", pp_size=1, is_speculative=True
    )
    assert is_hisparse_prefetcher_mode_unsupported(
        "previous", pp_size=1, is_speculative=True
    )
    assert is_hisparse_prefetcher_mode_unsupported(
        "oasiskv", pp_size=2, is_speculative=True
    )


def test_paired_row_mapping_and_positions():
    paired = build_oasiskv_paired_batch(
        torch.tensor([10, 20]), torch.tensor([11, 21]), torch.tensor([7, 13])
    )
    assert paired.input_ids.tolist() == [10, 11, 20, 21]
    assert paired.positions.tolist() == [7, 8, 13, 14]
    assert paired.normal_rows.tolist() == [0, 2]
    assert paired.draft_rows.tolist() == [1, 3]


def test_real_forward_batch_receives_two_token_extend_geometry():
    paired = build_oasiskv_paired_batch(
        torch.tensor([10, 20]), torch.tensor([11, 21]), torch.tensor([7, 13])
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.empty(4, dtype=torch.long),
        positions=None,
        extend_seq_lens_cpu=[2, 2],
        out_cache_loc=torch.arange(4),
        extend_prefix_lens=torch.tensor([7, 13]),
        extend_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([9, 15]),
        seq_lens_cpu=torch.tensor([9, 15]),
    )
    configure_oasiskv_forward_batch(forward_batch, paired)
    assert forward_batch.is_oasiskv_paired
    assert forward_batch.input_ids.tolist() == [10, 11, 20, 21]
    assert forward_batch.positions.tolist() == [7, 8, 13, 14]


def test_eagle_verify_tensors_are_adopted_without_copy_or_reprojection():
    tokens = torch.tensor([10, 11, 20, 21])
    positions = torch.tensor([7, 8, 13, 14])
    paired = paired_batch_from_eagle_verify(
        SimpleNamespace(draft_token_num=2, draft_token=tokens, positions=positions),
        2,
    )
    assert paired.input_ids is tokens
    assert paired.positions is positions
    assert paired.normal_rows.tolist() == [0, 2]
    assert paired.draft_rows.tolist() == [1, 3]


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

    coordinator.consume_oasiskv_prefetch(
        req_pool_indices=torch.tensor([2], device="cpu"),
        layer_id=0,
        req_pool_indices_cpu=torch.tensor([2]),
        committed_lens_cpu=torch.tensor([11]),
    )

    assert calls == [("wait", "stale"), ("wait", "current")]
    assert not stale.valid and not current.valid
    assert coordinator.prefetcher.stats.completed_h2d_entries == 8
    assert coordinator.prefetcher.stats.prefetch_hits == 1
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
