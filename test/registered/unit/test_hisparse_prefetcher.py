import random
from types import SimpleNamespace

import pytest

from sglang.srt.managers.hisparse_prefetcher import (
    RandomHiSparsePrefetcher,
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
    assert config.top_k == 128
    assert config.device_buffer_size == 256
    assert config.prefetcher is None
    assert config.prefetcher_config == {}


def test_random_config_and_effective_top_k_default():
    config = _config('{"prefetcher":"random"}')
    prefetcher = create_hisparse_prefetcher(
        config.prefetcher,
        config.prefetcher_config,
        effective_top_k=37,
        device_buffer_size=64,
    )
    assert isinstance(prefetcher, RandomHiSparsePrefetcher)
    assert prefetcher.logical_entries == 37
    assert prefetcher.size == 37


def test_dsv4_size_is_token_coverage():
    explicit = create_hisparse_prefetcher(
        "random",
        {"size": 2048},
        effective_top_k=512,
        device_buffer_size=6144,
        entry_token_span=4,
    )
    assert explicit.size == 2048
    assert explicit.logical_entries == 512

    defaulted = create_hisparse_prefetcher(
        "random",
        {},
        effective_top_k=512,
        device_buffer_size=6144,
        entry_token_span=4,
    )
    assert defaulted.size == 2048
    assert defaulted.logical_entries == 512


def test_token_coverage_rounds_up_to_a_logical_entry():
    prefetcher = create_hisparse_prefetcher(
        "random",
        {"size": 513},
        effective_top_k=512,
        device_buffer_size=1024,
        entry_token_span=4,
    )
    assert prefetcher.size == 513
    assert prefetcher.logical_entries == 129


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
def test_invalid_size(value):
    with pytest.raises(ValueError, match="size"):
        create_hisparse_prefetcher(
            "random",
            {"size": value},
            effective_top_k=4,
            device_buffer_size=8,
        )


def test_size_cannot_exceed_device_buffer():
    with pytest.raises(ValueError, match="must not exceed"):
        create_hisparse_prefetcher(
            "random",
            {"size": 9},
            effective_top_k=4,
            device_buffer_size=8,
        )


@pytest.mark.parametrize("seed", [1.5, True, "0"])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        create_hisparse_prefetcher(
            "random",
            {"seed": seed},
            effective_top_k=4,
            device_buffer_size=8,
        )


def test_unknown_prefetcher_and_config_fields_are_rejected():
    with pytest.raises(ValueError, match=r"supported prefetchers: random"):
        create_hisparse_prefetcher("typo", {}, effective_top_k=4, device_buffer_size=8)
    with pytest.raises(ValueError, match="typo_field"):
        create_hisparse_prefetcher(
            "random",
            {"typo_field": 1},
            effective_top_k=4,
            device_buffer_size=8,
        )
    with pytest.raises(ValueError, match="num_entries"):
        create_hisparse_prefetcher(
            "random",
            {"num_entries": 512},
            effective_top_k=4,
            device_buffer_size=512,
        )
    assert supported_hisparse_prefetchers() == ("random",)


def test_random_selection_is_unique_bounded_and_sized():
    prefetcher = RandomHiSparsePrefetcher(logical_entries=7, seed=11)
    selected = prefetcher.select(
        request_id=3, layer_id=5, decode_step=8, history_size=100
    )
    assert len(selected) == 7
    assert len(set(selected)) == 7
    assert all(0 <= entry < 100 for entry in selected)
    assert (
        len(prefetcher.select(request_id=3, layer_id=5, decode_step=9, history_size=4))
        == 4
    )


def test_random_selection_is_reproducible_and_call_scoped():
    args = dict(request_id=3, layer_id=5, decode_step=8, history_size=1000)
    first = RandomHiSparsePrefetcher(20, seed=7).select(**args)
    assert first == RandomHiSparsePrefetcher(20, seed=7).select(**args)
    variants = [
        RandomHiSparsePrefetcher(20, seed=7).select(**{**args, key: args[key] + 1})
        for key in ("request_id", "layer_id", "decode_step")
    ]
    assert all(value != first for value in variants)


def test_random_selection_does_not_touch_global_rng_or_allocate_history():
    random.seed(123)
    expected = random.random()
    random.seed(123)
    selected = RandomHiSparsePrefetcher(8, seed=0).select(
        request_id=1, layer_id=2, decode_step=3, history_size=1_000_000
    )
    assert random.random() == expected
    assert len(selected) == 8
