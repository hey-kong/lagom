"""Pluggable candidate selection for the HiSparse device-buffer prefetch path."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional


@dataclass
class HiSparsePrefetchStats:
    selected_entries: int = 0
    submitted_entries: int = 0
    completed_h2d_entries: int = 0


class HiSparsePrefetcher(ABC):
    """Select logical KV entries; cache ownership remains in the coordinator."""

    def __init__(self, logical_entries: int, size: Optional[int] = None):
        # logical_entries is the runtime KV width. size is the public token
        # coverage (they differ for compressed KV models).
        self.logical_entries = logical_entries
        self.size = logical_entries if size is None else size
        self.stats = HiSparsePrefetchStats()

    @abstractmethod
    def select(
        self, *, request_id: int, layer_id: int, decode_step: int, history_size: int
    ) -> list[int]:
        """Return unique logical entries in ``[0, history_size)``."""


_PREFETCHER_REGISTRY: Dict[str, Callable[..., HiSparsePrefetcher]] = {}


def register_hisparse_prefetcher(name: str):
    def decorate(factory: Callable[..., HiSparsePrefetcher]):
        _PREFETCHER_REGISTRY[name] = factory
        return factory

    return decorate


def supported_hisparse_prefetchers() -> tuple[str, ...]:
    return tuple(sorted(_PREFETCHER_REGISTRY))


def _require_int(config: Mapping, key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"prefetcher_config.{key} must be an integer, got {value!r}")
    return value


def create_hisparse_prefetcher(
    name: Optional[str],
    config: Mapping,
    *,
    effective_top_k: int,
    device_buffer_size: int,
    entry_token_span: int = 1,
) -> Optional[HiSparsePrefetcher]:
    resolved = validate_hisparse_prefetcher(
        name,
        config,
        effective_top_k=effective_top_k,
        device_buffer_size=device_buffer_size,
        entry_token_span=entry_token_span,
    )
    if resolved is None:
        return None
    factory, logical_entries, size, seed = resolved
    return factory(logical_entries=logical_entries, size=size, seed=seed)


def validate_hisparse_prefetcher(
    name: Optional[str],
    config: Mapping,
    *,
    effective_top_k: int,
    device_buffer_size: int,
    entry_token_span: int = 1,
):
    """Validate configuration without constructing an algorithm instance."""
    if name is None:
        return None
    normalized = name.lower()
    factory = _PREFETCHER_REGISTRY.get(normalized)
    if factory is None:
        supported = ", ".join(supported_hisparse_prefetchers()) or "(none)"
        raise ValueError(
            f"Unknown HiSparse prefetcher {name!r}; supported prefetchers: {supported}"
        )
    unknown = set(config) - {"size", "seed"}
    if unknown:
        raise ValueError(
            "Unknown random prefetcher_config field(s): " + ", ".join(sorted(unknown))
        )
    if entry_token_span <= 0:
        raise ValueError("entry_token_span must be positive")
    # The public option is token coverage. The model top-k is expressed in the
    # Indexer's logical-entry space, so expand its default by the entry span.
    size = _require_int(config, "size", effective_top_k * entry_token_span)
    if size <= 0:
        raise ValueError("prefetcher_config.size must be positive")
    logical_entries = (size + entry_token_span - 1) // entry_token_span
    if logical_entries > device_buffer_size:
        raise ValueError(
            f"prefetcher_config.size ({size} tokens, "
            f"{logical_entries} logical entries) must not exceed device buffer "
            f"capacity ({device_buffer_size} logical entries, "
            f"{device_buffer_size * entry_token_span} tokens)"
        )
    seed = _require_int(config, "seed", 0)
    return factory, logical_entries, size, seed


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


@register_hisparse_prefetcher("random")
class RandomHiSparsePrefetcher(HiSparsePrefetcher):
    """Stateless, reproducible O(k)-space sampling without replacement.

    Floyd's algorithm draws ``k=min(logical_entries, history_size)`` values in O(k)
    expected time and space, rather than materializing an O(history_size)
    permutation.  A per-call ``random.Random`` keeps model/global RNGs untouched.
    """

    def __init__(self, logical_entries: int, seed: int = 0, size: Optional[int] = None):
        super().__init__(logical_entries, size)
        self.seed = seed

    def select(
        self, *, request_id: int, layer_id: int, decode_step: int, history_size: int
    ) -> list[int]:
        if history_size <= 0:
            return []
        count = min(self.logical_entries, history_size)
        call_seed = _mix64(self.seed & 0xFFFFFFFFFFFFFFFF)
        for component in (request_id, layer_id, decode_step):
            call_seed = _mix64(call_seed ^ (component & 0xFFFFFFFFFFFFFFFF))
        rng = random.Random(call_seed)
        selected: set[int] = set()
        for upper in range(history_size - count, history_size):
            candidate = rng.randrange(upper + 1)
            selected.add(upper if candidate in selected else candidate)
        result = list(selected)
        self.stats.selected_entries += len(result)
        return result
