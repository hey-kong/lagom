from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.hisparse_prefetcher import (
    EMAPrefetcher,
    OasisKVPrefetcher,
    PreviousPrefetcher,
    create_hisparse_prefetcher,
    supported_hisparse_prefetchers,
)
from sglang.srt.mem_cache.sparsity import (
    parse_hisparse_config,
    resolve_dspark_device_buffer_size,
)


def _config(value):
    return parse_hisparse_config(SimpleNamespace(hisparse_config=value))


def test_legacy_config_is_unchanged():
    config = _config(
        '{"top_k":128,"device_buffer_size":256,"host_to_device_ratio":10,'
        '"swap_in_block_size":128}'
    )
    assert config.prefetcher is None
    assert config.prefetcher_config == {}


def test_dspark_defaults_device_buffer_to_verify_union_upper_bound():
    config = _config('{"top_k":2048,"host_to_device_ratio":5}')
    resolve_dspark_device_buffer_size(
        config,
        raw_hisparse_config='{"top_k":2048,"host_to_device_ratio":5}',
        verify_width=6,
        effective_top_k=512,
    )
    assert config.device_buffer_size == 3072


def test_dspark_preserves_sufficient_explicit_device_buffer():
    config = _config('{"top_k":2048,"device_buffer_size":4096}')
    resolve_dspark_device_buffer_size(
        config,
        raw_hisparse_config='{"top_k":2048,"device_buffer_size":4096}',
        verify_width=6,
        effective_top_k=512,
    )
    assert config.device_buffer_size == 4096


def test_dspark_rejects_insufficient_explicit_device_buffer():
    config = _config('{"top_k":512,"device_buffer_size":2048}')
    with pytest.raises(ValueError, match=r"6 \* 512 = 3072"):
        resolve_dspark_device_buffer_size(
            config,
            raw_hisparse_config='{"top_k":512,"device_buffer_size":2048}',
            verify_width=6,
            effective_top_k=512,
        )


def test_previous_default_size():
    config = _config('{"prefetcher":"previous"}')
    prefetcher = create_hisparse_prefetcher(
        config.prefetcher,
        config.prefetcher_config,
        effective_top_k=37,
        device_buffer_size=64,
    )
    assert isinstance(prefetcher, PreviousPrefetcher)
    assert prefetcher.logical_entries == 37
    assert prefetcher.size == 37


def test_dsv4_size_is_token_coverage():
    prefetcher = create_hisparse_prefetcher(
        "previous",
        {"size": 2048},
        effective_top_k=512,
        device_buffer_size=6144,
        entry_token_span=4,
    )
    assert prefetcher.size == 2048
    assert prefetcher.logical_entries == 512


def test_token_coverage_rounds_up_to_a_logical_entry():
    prefetcher = create_hisparse_prefetcher(
        "previous",
        {"size": 513},
        effective_top_k=512,
        device_buffer_size=1024,
        entry_token_span=4,
    )
    assert prefetcher.logical_entries == 129


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
def test_invalid_size(value):
    with pytest.raises(ValueError, match="size"):
        create_hisparse_prefetcher(
            "previous",
            {"size": value},
            effective_top_k=4,
            device_buffer_size=8,
        )


def test_size_can_exceed_attention_top_k():
    prefetcher = create_hisparse_prefetcher(
        "previous",
        {"size": 9},
        effective_top_k=4,
        device_buffer_size=16,
    )
    assert prefetcher.logical_entries == 9


def test_dsv4_size_can_exceed_attention_top_k():
    prefetcher = create_hisparse_prefetcher(
        "previous",
        {"size": 4096},
        effective_top_k=512,
        device_buffer_size=6144,
        entry_token_span=4,
    )
    assert prefetcher.logical_entries == 1024


def test_size_cannot_exceed_device_buffer():
    with pytest.raises(ValueError, match="device buffer capacity"):
        create_hisparse_prefetcher(
            "previous",
            {"size": 17},
            effective_top_k=4,
            device_buffer_size=16,
        )


def test_unknown_algorithm_and_fields_are_rejected():
    with pytest.raises(ValueError, match="previous"):
        create_hisparse_prefetcher(
            "random", {}, effective_top_k=4, device_buffer_size=8
        )
    with pytest.raises(
        ValueError, match="supported prefetchers: ema, oasiskv, previous"
    ):
        create_hisparse_prefetcher(
            "previous_layer_topk", {}, effective_top_k=4, device_buffer_size=8
        )
    with pytest.raises(ValueError, match="seed"):
        create_hisparse_prefetcher(
            "previous",
            {"seed": 0},
            effective_top_k=4,
            device_buffer_size=8,
        )
    assert supported_hisparse_prefetchers() == ("ema", "oasiskv", "previous")


def test_oasiskv_token_coverage_and_selection():
    prefetcher = create_hisparse_prefetcher(
        "oasiskv",
        {"size": 4096},
        effective_top_k=512,
        device_buffer_size=4096,
        entry_token_span=4,
    )
    assert isinstance(prefetcher, OasisKVPrefetcher)
    assert prefetcher.logical_entries == 1024
    predicted = torch.arange(2048, dtype=torch.int32).view(2, 1024)
    assert torch.equal(prefetcher.select(predicted), predicted)


def test_selects_highest_score_prefix_without_modifying_input():
    prefetcher = PreviousPrefetcher(logical_entries=3, size=3)
    previous = torch.tensor([[9, 4, 7, 2], [8, 1, 6, 3]], dtype=torch.int32)
    original = previous.clone()
    selected = prefetcher.select(previous)
    assert torch.equal(selected, torch.tensor([[9, 4, 7], [8, 1, 6]]))
    assert torch.equal(previous, original)
    assert selected.data_ptr() == previous.data_ptr()


def test_rejects_invalid_or_too_short_previous():
    prefetcher = PreviousPrefetcher(logical_entries=3)
    with pytest.raises(ValueError, match="two-dimensional"):
        prefetcher.select(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="2 entries"):
        prefetcher.select(torch.tensor([[1, 2]]))


def test_ema_defaults_and_c4_token_coverage():
    prefetcher = create_hisparse_prefetcher(
        "ema", {}, effective_top_k=512, device_buffer_size=4096, entry_token_span=4
    )
    assert isinstance(prefetcher, EMAPrefetcher)
    assert prefetcher.logical_entries == 512
    assert prefetcher.size == 2048
    assert (prefetcher.alpha, prefetcher.beta, prefetcher.gamma) == (0.6, 0.2, 0.25)


def test_ema_first_observation_skips_then_updates_level_and_trend():
    prefetcher = EMAPrefetcher(logical_entries=2)
    first = torch.tensor([[1.0, 4.0, 2.0]])
    formal_top_k = torch.tensor([[7, 8]], dtype=torch.int32)
    assert prefetcher.update(first, torch.tensor([3]), [11], 2) is None

    second = torch.tensor([[3.0, 2.0, 5.0]])
    predicted = prefetcher.update(second, torch.tensor([3]), [11], 2)
    # level=[2.2, 2.8, 3.8], trend=[.4, -.4, .6], forecast=[2.3,2.7,3.95]
    assert torch.equal(predicted, torch.tensor([[2, 1]], dtype=torch.int32))
    # Predictions are side data: the current Indexer Top-K stays byte-for-byte intact.
    assert torch.equal(formal_top_k, torch.tensor([[7, 8]], dtype=torch.int32))


def test_ema_enabled_and_disabled_attention_selection_is_identical():
    """EMA warms residency only; attention continues to gather formal Top-K."""
    values = torch.tensor([[2.0, 3.0, 5.0, 7.0]])
    formal = torch.tensor([[3, 1]])
    disabled = torch.gather(values, 1, formal)

    prefetcher = EMAPrefetcher(logical_entries=2)
    prefetcher.update(values, torch.tensor([4]), [0], 0)
    prediction = prefetcher.update(values.flip(1), torch.tensor([4]), [0], 0)
    assert prediction is not None and not torch.equal(prediction.long(), formal)
    enabled = torch.gather(values, 1, formal)
    assert torch.equal(enabled, disabled)


def test_ema_state_isolated_by_request_and_layer_and_survives_batch_reorder():
    prefetcher = EMAPrefetcher(logical_entries=1)
    scores = torch.tensor([[9.0, 1.0], [1.0, 9.0]])
    assert prefetcher.update(scores, torch.tensor([2, 2]), [3, 7], 0) is None
    # Reverse batch rows; request identity, rather than row number, selects history.
    result = prefetcher.update(scores.flip(0), torch.tensor([2, 2]), [7, 3], 0)
    assert torch.equal(result, torch.tensor([[1], [0]], dtype=torch.int32))
    assert prefetcher.update(scores[:1], torch.tensor([2]), [3], 1) is None
    assert set(prefetcher._state) == {(3, 0), (7, 0), (3, 1)}

    prefetcher.release_request(3)
    assert set(prefetcher._state) == {(7, 0)}
    assert prefetcher.update(scores[:1], torch.tensor([2]), [3], 0) is None


@pytest.mark.parametrize("field,value", [("alpha", -0.1), ("beta", 1.1), ("gamma", -1)])
def test_ema_rejects_invalid_smoothing_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        create_hisparse_prefetcher(
            "ema",
            {field: value},
            effective_top_k=4,
            device_buffer_size=8,
        )
