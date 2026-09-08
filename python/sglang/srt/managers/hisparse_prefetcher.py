"""Pluggable candidate selection for the HiSparse device-buffer prefetch path."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch


@dataclass
class HiSparsePrefetchStats:
    draft_tokens_generated: int = 0
    selected_entries: int = 0
    submitted_entries: int = 0
    completed_h2d_entries: int = 0
    prediction_hits: int = 0
    prediction_total: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_h2d_bytes: int = 0
    fallback_h2d_bytes: int = 0
    draft_target_forward_seconds: float = 0.0
    # Number of stream-wait dependencies enqueued.  This is deliberately not
    # presented as GPU wall time; event.wait() is asynchronous on the CPU.
    prefetch_wait_submissions: int = 0
    prefetch_wait_seconds: float = 0.0
    stale_tasks: int = 0


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
    factory, logical_entries, size, algorithm_config = resolved
    return factory(logical_entries=logical_entries, size=size, **algorithm_config)


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
    algorithm_fields = {"alpha", "beta", "gamma"} if normalized == "ema" else set()
    unknown = set(config) - ({"size"} | algorithm_fields)
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
    if logical_entries > device_buffer_size:
        raise ValueError(
            f"prefetcher_config.size ({size} tokens, {logical_entries} logical "
            f"entries) exceeds device buffer capacity ({device_buffer_size} "
            "logical entries)"
        )
    algorithm_config = {}
    for key, default in (("alpha", 0.6), ("beta", 0.2), ("gamma", 0.25)):
        if key not in algorithm_fields:
            continue
        value = config.get(key, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"prefetcher_config.{key} must be a number")
        value = float(value)
        if key != "gamma" and not 0.0 <= value <= 1.0:
            raise ValueError(f"prefetcher_config.{key} must be in [0, 1]")
        if key == "gamma" and value < 0.0:
            raise ValueError("prefetcher_config.gamma must be non-negative")
        algorithm_config[key] = value
    return factory, logical_entries, size, algorithm_config


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


@register_hisparse_prefetcher("oasiskv")
class OasisKVPrefetcher(HiSparsePrefetcher):
    """Candidates predicted by a draft-token query in the target C4 indexer."""

    def select(self, predicted):
        if predicted is None or predicted.ndim != 2:
            raise ValueError("OasisKV prediction must be a two-dimensional tensor")
        if predicted.shape[1] < self.logical_entries:
            raise ValueError(
                f"OasisKV prediction has {predicted.shape[1]} entries, but "
                f"{self.logical_entries} are required"
            )
        result = predicted[:, : self.logical_entries]
        self.stats.selected_entries += result.numel()
        return result


@register_hisparse_prefetcher("ema")
class EMAPrefetcher(HiSparsePrefetcher):
    """PRR-style per-request, per-layer EMA prediction over Indexer scores."""

    def __init__(
        self,
        logical_entries: int,
        size: Optional[int] = None,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.25,
    ):
        super().__init__(logical_entries, size)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._state: Dict[
            Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._positions: Optional[torch.Tensor] = None

    def select(self, previous):
        """EMA consumes scores through :meth:`update`; indices are passed through."""
        if previous is None or previous.ndim != 2:
            raise ValueError("EMA candidates must be a two-dimensional tensor")
        # update() already accounts for valid, initialized rows without reading
        # a GPU reduction back to the CPU.
        return previous[:, : self.logical_entries]

    def update(
        self,
        scores: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices,
        layer_id: int,
        out_indices: Optional[torch.Tensor] = None,
        seq_lens_device: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        page_size: int = 1,
        out_page_indices: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Update observations and predict the next decode token's score ranking.

        The first observation for any request/layer initializes state and emits
        no prediction. State follows request-pool identity, not batch row order.
        """
        slots = torch.as_tensor(req_pool_indices, device="cpu").tolist()
        lengths = torch.as_tensor(seq_lens_cpu, device="cpu").tolist()
        if len(slots) != scores.shape[0]:
            raise ValueError("EMA request identities must match score rows")
        if len(lengths) != scores.shape[0]:
            raise ValueError("EMA CPU sequence lengths must match score rows")
        if seq_lens_device is None:
            if scores.device.type != "cpu":
                raise ValueError("EMA CUDA scores require device sequence lengths")
            seq_lens_device = torch.as_tensor(seq_lens_cpu, device="cpu")
        if seq_lens_device.shape[0] != scores.shape[0]:
            raise ValueError("EMA device sequence lengths must match score rows")

        width = scores.shape[1]
        old_levels = []
        old_trends = []
        old_masks = []
        initialized = []
        keys = []
        for slot in slots:
            key = (int(slot), int(layer_id))
            keys.append(key)
            old = self._state.get(key)
            if old is None:
                old_level = old_trend = scores.new_zeros(width, dtype=torch.float32)
                old_mask = torch.zeros(width, dtype=torch.bool, device=scores.device)
                initialized.append(False)
            else:
                old_level, old_trend, old_mask = old
                old_length = min(old_level.numel(), width)
                if old_length < width:
                    old_level = torch.nn.functional.pad(
                        old_level, (0, width - old_length)
                    )
                    old_trend = torch.nn.functional.pad(
                        old_trend, (0, width - old_length)
                    )
                    old_mask = torch.nn.functional.pad(
                        old_mask, (0, width - old_length)
                    )
                else:
                    old_level = old_level[:width]
                    old_trend = old_trend[:width]
                    old_mask = old_mask[:width]
                initialized.append(True)
            old_levels.append(old_level)
            old_trends.append(old_trend)
            old_masks.append(old_mask)

        current = scores.detach().float()
        previous_level = torch.stack(old_levels)
        previous_trend = torch.stack(old_trends)
        positions = self._positions
        if (
            positions is None
            or positions.device != scores.device
            or positions.shape[1] != width
        ):
            positions = torch.arange(width, device=scores.device).unsqueeze(0)
            self._positions = positions
        valid = positions < seq_lens_device.unsqueeze(1)
        has_history = torch.stack(old_masks)
        updated_level = self.alpha * current + (1.0 - self.alpha) * previous_level
        updated_trend = (
            self.beta * (current - previous_level) + (1.0 - self.beta) * previous_trend
        )
        level = torch.where(has_history, updated_level, current).masked_fill(~valid, 0)
        trend = torch.where(
            has_history, updated_trend, torch.zeros_like(updated_trend)
        ).masked_fill(~valid, 0)
        forecast = (level + self.gamma * trend).masked_fill(~valid, float("-inf"))

        for row, key in enumerate(keys):
            # Store views of the batch result rather than cloning level/trend
            # once per request. The next update is functional and never mutates
            # these tensors, so sharing the batch backing storage is safe.
            self._state[key] = (
                level[row],
                trend[row],
                valid[row],
            )

        if not any(initialized):
            return None

        output = out_indices
        if output is None:
            output = torch.full(
                (scores.shape[0], self.logical_entries),
                -1,
                dtype=torch.int32,
                device=scores.device,
            )
        output.fill_(-1)
        actual_k = min(self.logical_entries, width)
        row_initialized = has_history.any(dim=1, keepdim=True)
        if (
            scores.device.type == "cuda"
            and page_table is not None
            and out_page_indices is not None
            and actual_k == self.logical_entries
            and self.logical_entries <= 1024
        ):
            # Reuse DSV4's optimized fused selection kernel. Its raw output is
            # exactly the logical C4 position wanted by HiSparse; physical page
            # output is scratch only. This avoids generic torch.topk, the main
            # remaining EMA cost at long context.
            from sglang.kernels.ops.attention.dsv4 import topk_transform_512

            topk_transform_512(
                forecast,
                seq_lens_device,
                page_table,
                out_page_indices,
                page_size,
                output,
            )
            output.masked_fill_(~row_initialized, -1)
        else:
            # CPU tests and unsupported candidate widths retain a portable path.
            # Prefetch consumes a set, so score ordering is unnecessary.
            values, indices = torch.topk(forecast, actual_k, dim=1, sorted=False)
            selected = indices.to(torch.int32).masked_fill(
                (values == float("-inf")) | ~row_initialized, -1
            )
            output[:, :actual_k].copy_(selected)
        self.stats.selected_entries += sum(
            min(self.logical_entries, length)
            for length, ready in zip(lengths, initialized)
            if ready
        )
        return output

    def release_request(self, req_pool_idx: int) -> None:
        for key in [key for key in self._state if key[0] == req_pool_idx]:
            del self._state[key]
