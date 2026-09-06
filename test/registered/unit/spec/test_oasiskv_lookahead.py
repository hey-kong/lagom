from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.speculative.oasiskv_lookahead import (
    build_oasiskv_commit,
    build_oasiskv_paired_batch,
    compute_oasiskv_logprobs,
    configure_oasiskv_forward_batch,
    paired_batch_from_eagle_verify,
    select_oasiskv_normal_rows,
    submit_oasiskv_layer_prefetch,
    submit_oasiskv_pending_prefetches,
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
        enforce_disable_flashinfer_allreduce_fusion=False,
        disable_cuda_graph=False,
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
    assert args.enforce_disable_flashinfer_allreduce_fusion
    assert not args.disable_cuda_graph


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


def test_eagle_verify_forward_batch_does_not_require_extend_only_metadata():
    tokens = torch.tensor([10, 11, 20, 21])
    positions = torch.tensor([7, 8, 13, 14])
    verify = SimpleNamespace(draft_token_num=2, draft_token=tokens, positions=positions)
    paired = paired_batch_from_eagle_verify(verify, 2)
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=tokens,
        positions=positions,
        extend_seq_lens_cpu=None,
        out_cache_loc=torch.arange(4),
        seq_lens=torch.tensor([7, 13]),
        seq_lens_cpu=torch.tensor([7, 13]),
        spec_info=verify,
    )

    configure_oasiskv_forward_batch(forward_batch, paired)

    assert forward_batch.is_oasiskv_paired
    assert forward_batch.oasiskv_normal_rows.tolist() == [0, 2]


def test_draft_extend_keeps_only_normal_target_features_and_cache_locs():
    features = torch.tensor([[10], [11], [20], [21]])
    cache_locs = torch.tensor([100, 101, 200, 201])

    assert select_oasiskv_normal_rows(features).tolist() == [[10], [20]]
    assert select_oasiskv_normal_rows(cache_locs).tolist() == [100, 200]


def test_commit_is_always_one_request_major_normal_row():
    accept_lens, accept_index = build_oasiskv_commit(
        torch.tensor([0, 2, 4]), batch_size=3, device="cpu"
    )
    assert accept_lens.tolist() == [1, 1, 1]
    assert accept_index.tolist() == [[0], [2], [4]]

    with pytest.raises(ValueError, match="pair roots"):
        build_oasiskv_commit(torch.tensor([0, 1, 4]), batch_size=3, device="cpu")


def test_logits_processor_selects_only_normal_rows_for_oasiskv():
    from sglang.srt.layers.logits_processor import LogitsMetadata

    normal_rows = torch.tensor([0, 2])
    metadata = LogitsMetadata.from_forward_batch(
        SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False,
                is_target_verify=lambda: True,
                is_draft_extend_v2=lambda: False,
            ),
            return_logprob=False,
            capture_hidden_mode=0,
            next_token_logits_buffer=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
            is_prefill_only=False,
            global_num_tokens_gpu=None,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_dp_buffer_len=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            mm_input_embeds=None,
            is_oasiskv_paired=True,
            oasiskv_normal_rows=normal_rows,
        )
    )
    assert metadata.output_select_index is normal_rows


def test_logprobs_use_compacted_normal_rows_not_verify_pair_indices():
    batch = SimpleNamespace(
        seq_lens=torch.tensor([7, 13]),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    logits_output = SimpleNamespace(
        # These are already the compacted normal rows for requests 0 and 1.
        next_token_logits=torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    )

    compute_oasiskv_logprobs(batch, logits_output, torch.tensor([0, 1]))

    expected = torch.log_softmax(logits_output.next_token_logits, dim=-1)[
        torch.arange(2), torch.tensor([0, 1])
    ]
    assert logits_output.next_token_logprobs.shape == (2, 1)
    torch.testing.assert_close(logits_output.next_token_logprobs[:, 0], expected)


def test_pending_prefetch_is_drained_once_after_verify_transaction():
    calls = []
    coordinator = SimpleNamespace(
        submit_oasiskv_prefetch=lambda **kwargs: calls.append(kwargs["layer_id"])
    )
    forward_batch = SimpleNamespace(
        _oasiskv_pending_prefetch={
            3: (coordinator, {"layer_id": 3}),
            7: (coordinator, {"layer_id": 7}),
        }
    )

    submit_oasiskv_pending_prefetches(forward_batch)
    submit_oasiskv_pending_prefetches(forward_batch)

    assert calls == [3, 7]
    assert forward_batch._oasiskv_pending_prefetch == {}


def test_eager_layer_prefetch_launches_immediately_without_c4_transaction():
    calls = []
    coordinator = SimpleNamespace(
        _active_dspark_window=None,
        submit_oasiskv_prefetch=lambda **kwargs: calls.append(kwargs["layer_id"]),
    )
    forward_batch = SimpleNamespace(
        is_oasiskv_graph_capture=False,
        _oasiskv_pending_prefetch={3: (coordinator, {"layer_id": 3})},
    )

    assert submit_oasiskv_layer_prefetch(forward_batch, 3)
    assert calls == [3]
    assert forward_batch._oasiskv_pending_prefetch == {}


def test_layer_prefetch_defers_while_c4_commit_owns_destinations():
    calls = []
    coordinator = SimpleNamespace(
        _active_dspark_window=object(),
        submit_oasiskv_prefetch=lambda **kwargs: calls.append(kwargs),
    )
    pending = {3: (coordinator, {"layer_id": 3})}
    forward_batch = SimpleNamespace(
        is_oasiskv_graph_capture=False, _oasiskv_pending_prefetch=pending
    )

    assert not submit_oasiskv_layer_prefetch(forward_batch, 3)
    assert not calls
    assert forward_batch._oasiskv_pending_prefetch is pending


def test_cuda_graph_replay_joins_all_layers_and_publishes_live_batch_metadata():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    consume_calls = []
    coordinator = SimpleNamespace(
        prefetcher_name="oasiskv",
        mem_pool_device=SimpleNamespace(layer_num=2),
        consume_oasiskv_prefetch=lambda **kwargs: consume_calls.append(kwargs),
    )
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(hisparse_coordinator=coordinator)
    runner._replay_graph_key = "bs2"
    predicted = torch.arange(12).view(3, 4)
    compressed_lens = torch.tensor([7, 13, 0])
    runner._oasiskv_graph_prefetch = {
        "bs2": {
            3: (
                coordinator,
                {
                    "compressed_seq_lens": compressed_lens,
                    "predicted_c4_entries": predicted,
                    "layer_id": 1,
                },
            )
        }
    }
    forward_batch = SimpleNamespace(
        is_oasiskv_paired=True,
        batch_size=2,
        req_pool_indices=torch.tensor([4, 9]),
        req_pool_indices_cpu=torch.tensor([4, 9]),
        seq_lens_cpu=torch.tensor([7, 13]),
    )

    runner._prepare_oasiskv_graph_replay(forward_batch)
    runner._publish_oasiskv_graph_prefetch(forward_batch)

    assert [call["layer_id"] for call in consume_calls] == [0, 1]
    _, submitted = forward_batch._oasiskv_pending_prefetch[3]
    assert submitted["req_pool_indices"] is forward_batch.req_pool_indices
    assert submitted["req_pool_indices_cpu"] is forward_batch.req_pool_indices_cpu
    assert submitted["source_committed_lens_cpu"] is forward_batch.seq_lens_cpu
    assert submitted["compressed_seq_lens"].tolist() == [7, 13]
    assert submitted["predicted_c4_entries"].tolist() == predicted[:2].tolist()


def test_oasiskv_graph_logits_and_hidden_states_have_different_live_widths():
    """Graph padding must use B for logits but 2B for paired hidden states."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    runner = object.__new__(DecodeCudaGraphRunner)
    runner.is_dllm = False
    runner.raw_num_token = 4
    runner.bs = 4
    graph_output = LogitsProcessorOutput(
        next_token_logits=torch.arange(24).view(6, 4),
        hidden_states=torch.arange(32).view(8, 4),
    )
    forward_batch = SimpleNamespace(is_oasiskv_paired=True, batch_size=2)

    # Exercise the same output-compaction block without constructing a CUDA
    # graph backend: the helper is deliberately factored for CPU regression.
    compact = runner._slice_replay_output(graph_output, forward_batch)

    assert compact.next_token_logits.shape == (2, 4)
    assert compact.hidden_states.shape == (4, 4)


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
