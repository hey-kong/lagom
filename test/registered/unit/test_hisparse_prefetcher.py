from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.hisparse_prefetcher import (
    PreviousPrefetcher,
    create_hisparse_prefetcher,
    supported_hisparse_prefetchers,
)
from sglang.srt.mem_cache.sparsity import parse_hisparse_config


def _config(value):
    return parse_hisparse_config(SimpleNamespace(hisparse_config=value))


def test_legacy_config_is_unchanged():
    config = _config(
        '{"top_k":128,"device_buffer_size":256,"host_to_device_ratio":10,'
        '"swap_in_block_size":128}'
    )
    assert config.prefetcher is None
    assert config.prefetcher_config == {}


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
    assert prefetcher.selection_k == 37


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


def test_size_can_expand_previous_selection_to_top_m():
    prefetcher = create_hisparse_prefetcher(
        "previous",
        {"size": 9},
        effective_top_k=4,
        device_buffer_size=16,
    )
    assert prefetcher.logical_entries == 9
    assert prefetcher.selection_k == 9


def test_dsv4_size_can_expand_previous_selection_to_top_m():
    prefetcher = create_hisparse_prefetcher(
        "previous",
        {"size": 4096},
        effective_top_k=512,
        device_buffer_size=6144,
        entry_token_span=4,
    )
    assert prefetcher.logical_entries == 1024
    assert prefetcher.selection_k == 1024


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
