"""Pluggable candidate selection for the HiSparse device-buffer prefetch path."""

from __future__ import annotations

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
        self.logical_entries = logical_entries
        self.size = logical_entries if size is None else size
        self.stats = HiSparsePrefetchStats()

    @abstractmethod
    def select(self, previous):
        """Select candidates from the preceding sparse layer's scored top-k."""


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
    factory, logical_entries, size = resolved
    return factory(logical_entries=logical_entries, size=size)


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
    unknown = set(config) - {"size"}
    if unknown:
        raise ValueError(
            f"Unknown {normalized} prefetcher_config field(s): "
            + ", ".join(sorted(unknown))
        )
    if entry_token_span <= 0:
        raise ValueError("entry_token_span must be positive")
    size = _require_int(config, "size", effective_top_k * entry_token_span)
    if size <= 0:
        raise ValueError("prefetcher_config.size must be positive")
    logical_entries = (size + entry_token_span - 1) // entry_token_span
    if logical_entries > effective_top_k:
        raise ValueError(
            f"prefetcher_config.size ({size} tokens, {logical_entries} logical "
            f"entries) exceeds the preceding layer top-k ({effective_top_k} "
            "logical entries)"
        )
    if logical_entries > device_buffer_size:
        raise ValueError(
            f"prefetcher_config.size ({size} tokens, {logical_entries} logical "
            f"entries) exceeds device buffer capacity ({device_buffer_size} "
            "logical entries)"
        )
    return factory, logical_entries, size


@register_hisparse_prefetcher("previous")
class PreviousPrefetcher(HiSparsePrefetcher):
    """Use the highest-score positions selected by the preceding sparse layer."""

    def select(self, previous):
        if previous is None or previous.ndim != 2:
            raise ValueError("previous must be a two-dimensional tensor")
        if previous.shape[1] < self.logical_entries:
            raise ValueError(
                f"previous has {previous.shape[1]} entries, but "
                f"{self.logical_entries} are required"
            )
        result = previous[:, : self.logical_entries]
        self.stats.selected_entries += result.numel()
        return result
