from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.hisparse_prefetcher import (
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
    with pytest.raises(ValueError, match="supported prefetchers: previous"):
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
    assert supported_hisparse_prefetchers() == ("previous",)


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
