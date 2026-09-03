# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_prefetcher import (
    create_hisparse_prefetcher,
    validate_hisparse_prefetcher,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


def dspark_completed_c4_positions(prefix_len: int, verify_width: int) -> List[int]:
    """C4 positions whose four-token group completes inside a verify window."""
    return [
        token_pos // 4
        for token_pos in range(prefix_len, prefix_len + verify_width)
        if (token_pos + 1) % 4 == 0
    ]


def dspark_fixed_buffer_commit_indices(
    accepted: List[int],
    req_offsets: List[int],
    req_pool_indices_cpu: List[int],
    c4_positions: List[int],
    device_buffer_size: int,
) -> List[int]:
    """Unique fixed-buffer copies; newest long C4 wins per request."""
    fixed_copy: List[int] = []
    latest_long: Dict[int, int] = {}
    for i in accepted:
        req_idx = req_pool_indices_cpu[req_offsets[i]]
        if c4_positions[i] < device_buffer_size:
            fixed_copy.append(i)
        else:
            latest_long[req_idx] = i
    fixed_copy.extend(latest_long.values())
    return fixed_copy


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


@dataclass
class HiSparseDSparkWindow:
    """Metadata for one target-verify transaction over fixed C4 scratch."""

    compressed_locs: torch.Tensor
    device_locs: torch.Tensor
    previous_device_mapping: torch.Tensor
    req_offsets: List[int]
    req_pool_indices_cpu: List[int]
    c4_positions: List[int]
    prefix_lens_cpu: List[int]


def resolve_shared_index_layers(
    *,
    hf_text_config,
    pp_size: int,
    is_speculative: bool,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (PP, speculative decoding, kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    if pp_size != 1 or is_speculative:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism / speculative decoding; falling back to synchronous "
            "swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
        prefetcher_name: Optional[str] = None,
        prefetcher_config: Optional[Mapping] = None,
        pp_size: int = 1,
        is_speculative: bool = False,
        speculative_verify_width: int = 0,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        # Public Previous size is measured in original-token coverage.
        # DSV4 C4 RESOLVE positions each address one compress_ratio-token entry.
        self.prefetch_entry_token_span = (
            self.compress_ratio if self.is_dsv4_hisparse else 1
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False
        self._active_dspark_window: Optional[HiSparseDSparkWindow] = None
        self._dspark_commit_done_event = device_module.Event()
        self._has_pending_dspark_commit = False
        self._dspark_scratch_compressed_locs: Optional[torch.Tensor] = None
        self._dspark_scratch_device_locs: Optional[torch.Tensor] = None
        self._dspark_scratch_page_locs: Optional[torch.Tensor] = None
        self._dspark_scratch_valid_mapping: Optional[torch.Tensor] = None
        if self.is_dsv4_hisparse and is_speculative and speculative_verify_width > 0:
            # A graph replay cannot change tensor addresses or allocate pages.
            # Reserve enough whole pages for the worst alignment of every
            # request, then update only their logical C4 identities before each
            # replay.  The buffers exist during capture too, where -1 marks all
            # entries invalid and therefore gives model.forward legal storage.
            max_groups_per_req = (
                speculative_verify_width + self.compress_ratio - 2
            ) // self.compress_ratio + 1
            scratch_capacity = max_num_req_slots * max(1, max_groups_per_req)
            scratch_size = (
                (scratch_capacity + self.page_size - 1) // self.page_size
            ) * self.page_size
            scratch_page_locs = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                    scratch_size
                )
            )
            if scratch_page_locs is None:
                raise RuntimeError(
                    "HiSparse DSPARK fixed C4 scratch allocation failed: "
                    f"requested {scratch_size} entries"
                )
            self._dspark_scratch_page_locs = scratch_page_locs
            self._dspark_scratch_device_locs = scratch_page_locs[:scratch_capacity]
            self._dspark_scratch_compressed_locs = torch.full(
                (scratch_capacity,), -1, dtype=torch.int64, device=self.device
            )
            self._dspark_scratch_valid_mapping = torch.zeros_like(
                self.mem_pool_device.full_to_hisparse_device_index_mapping,
                dtype=torch.bool,
            )

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.indexer_prefetch_candidates_buffer = None
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )
        self.prefetcher = None
        # Validate even when a higher-priority mode wins, so misspellings never
        # become silently dormant configuration.
        validate_hisparse_prefetcher(
            prefetcher_name,
            prefetcher_config or {},
            effective_top_k=self.top_k,
            device_buffer_size=self.device_buffer_size,
            entry_token_span=self.prefetch_entry_token_span,
        )
        if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
            logger.info("HiSparse prefetch mode: disabled")
        elif self.enable_prefetch:
            logger.info("HiSparse prefetch mode: legacy shared-index")
            if prefetcher_name is not None:
                logger.info(
                    'HiSparse prefetcher "%s" is not activated because legacy '
                    "shared-index prefetch has higher priority for this model.",
                    prefetcher_name,
                )
        elif prefetcher_name is not None and (pp_size != 1 or is_speculative):
            logger.warning(
                'HiSparse prefetcher "%s" is disabled under pipeline parallelism '
                "or speculative decoding; HiSparse prefetch mode: disabled",
                prefetcher_name,
            )
        else:
            self.prefetcher = create_hisparse_prefetcher(
                prefetcher_name,
                prefetcher_config or {},
                effective_top_k=self.top_k,
                device_buffer_size=self.device_buffer_size,
                entry_token_span=self.prefetch_entry_token_span,
            )
            logger.info(
                "HiSparse prefetch mode: %s",
                prefetcher_name.lower() if prefetcher_name is not None else "disabled",
            )
        if self.prefetcher is not None:
            if not self.is_dsv4_hisparse and (
                self.prefetcher.logical_entries != self.top_k
            ):
                raise ValueError(
                    "generic DSA currently supports prefetcher_config.size only "
                    "when it equals attention Top-k; selecting a smaller or "
                    "larger highest-score candidate set requires score-aware "
                    "Indexer output, which is currently implemented only for "
                    "DeepSeek-V4 C4 HiSparse"
                )
            self.indexer_prefetch_candidates_buffer = torch.full(
                (max_num_req_slots, self.prefetcher.logical_entries),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._prefetch_candidate_buffer = torch.full(
                (max_num_req_slots, self.prefetcher.logical_entries),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._previous_prefetch_device_locs = torch.full_like(
                self._prefetch_candidate_buffer, -1
            )
            self._previous_miss_src = torch.zeros(
                (max_num_req_slots, self.prefetcher.logical_entries),
                dtype=torch.int64,
                device=device,
            )
            self._previous_miss_dst = torch.zeros(
                (max_num_req_slots, self.prefetcher.logical_entries),
                dtype=torch.int32,
                device=device,
            )
            self._previous_miss_count = torch.zeros(
                max_num_req_slots, dtype=torch.int32, device=device
            )
            self._previous_prefetch_stream = device_module.Stream()
            self._previous_prefetch_event = device_module.Event()
            self._previous_prefetch_target_layer = None
            self._previous_prefetch_num_reqs = 0
            self._previous_prefetch_pending_entries = 0
            logger.info(
                "HiSparse Previous Prefetcher: %d-token coverage maps to %d "
                "logical KV entries per request (entry span=%d tokens); Indexer "
                "selection is Top-%d and attention remains Top-%d.",
                self.prefetcher.size,
                self.prefetcher.logical_entries,
                self.prefetch_entry_token_span,
                self.prefetcher.logical_entries,
                self.top_k,
            )

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        self._miss_src = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=self.device
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int32, device=self.device
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        if self.prefetcher is not None:
            self._previous_prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert not self.is_dsv4_hisparse, (
            "naive_load_topk is not implemented for dsv4 hisparse"
        )
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(selected_tokens >= 0), (
                f"Req {req_idx}: selected tokens contain negative positions"
            )
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
        num_top_k: Optional[int] = None,
        output_buffer: Optional[torch.Tensor] = None,
        miss_plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        skip_io: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan into self._miss_{src,dst,count} for the skip layers to replay.
        """
        num_reqs = req_pool_indices.size(0)
        num_top_k = num_top_k or self.top_k
        top_k_indices = (
            self.top_k_device_locs_buffer if output_buffer is None else output_buffer
        )[:num_reqs]

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        if record_plan:
            miss_src, miss_dst, miss_count = miss_plan or (
                self._miss_src,
                self._miss_dst,
                self._miss_count,
            )
            plan = dict(
                miss_src=miss_src[:num_reqs],
                miss_dst=miss_dst[:num_reqs],
                miss_count=miss_count[:num_reqs],
            )
        else:
            plan = {}
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=num_top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io if skip_io is None else skip_io,
            **plan,
        )
        return top_k_indices

    def _consume_previous_prefetch(
        self,
        req_pool_indices: torch.Tensor,
        layer_id: int,
    ) -> None:
        if (
            self.prefetcher is None
            or self._previous_prefetch_target_layer != layer_id
            or self._previous_prefetch_num_reqs != req_pool_indices.size(0)
        ):
            return
        self._previous_prefetch_event.wait(device_module.current_stream())
        self.prefetcher.stats.completed_h2d_entries += (
            self._previous_prefetch_pending_entries
        )
        self._previous_prefetch_pending_entries = 0

    def _submit_previous_prefetch(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        previous: torch.Tensor,
        source_layer_id: int,
    ) -> None:
        """Stage the next sparse layer while the current layer computes."""
        if (
            self.prefetcher is None
            or source_layer_id + 1 >= self.mem_pool_device.layer_num
        ):
            self._previous_prefetch_target_layer = None
            return
        candidates = self.prefetcher.select(previous)
        num_reqs = candidates.size(0)
        self._prefetch_candidate_buffer[:num_reqs].copy_(candidates)
        target_layer = source_layer_id + 1
        # Plan-only RESOLVE updates the target layer's existing resident/LRU
        # metadata and records host->resident-slot copies, but moves no KV bytes.
        self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            self._prefetch_candidate_buffer[:num_reqs],
            target_layer,
            record_plan=True,
            num_top_k=self.prefetcher.logical_entries,
            output_buffer=self._previous_prefetch_device_locs,
            miss_plan=(
                self._previous_miss_src,
                self._previous_miss_dst,
                self._previous_miss_count,
            ),
            skip_io=True,
        )
        self._previous_prefetch_stream.wait_stream(device_module.current_stream())
        with device_module.stream(self._previous_prefetch_stream):
            copy_cache_planned_mla(
                miss_src=self._previous_miss_src[:num_reqs],
                miss_dst=self._previous_miss_dst[:num_reqs],
                miss_count=self._previous_miss_count[:num_reqs],
                num_real_reqs=self.num_real_reqs,
                host_cache=self.mem_pool_host.kv_buffer[target_layer],
                device_buffer=self.mem_pool_device.kv_buffer[target_layer],
                item_size_bytes=self.item_size_bytes,
                num_blocks=4,
                is_dsv4_layout=self.is_dsv4_hisparse,
                skip_io=self.skip_io,
            )
            self._previous_prefetch_event.record(self._previous_prefetch_stream)
        self._previous_prefetch_target_layer = target_layer
        self._previous_prefetch_num_reqs = num_reqs
        submitted_entries = num_reqs * self.prefetcher.logical_entries
        self._previous_prefetch_pending_entries = submitted_entries
        self.prefetcher.stats.submitted_entries += submitted_entries

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        prefetch_candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        self._consume_previous_prefetch(req_pool_indices, layer_id)
        if not self.enable_prefetch:
            result = self._run_swap_in_kernel(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result[:, : self.top_k],
                layer_id,
            )
            self._submit_previous_prefetch(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result if prefetch_candidates is None else prefetch_candidates,
                layer_id,
            )
            return result

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs

    def swap_in_selected_pages_spec(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        *,
        verify_lens_cpu: Optional[List[int]] = None,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Swap DSPARK verify selections in request-major step order.

        ``top_k_result`` is the *real* Indexer output, with either static
        ``[B * W, K]`` rows or compact ``[sum(verify_lens), K]`` rows. LRU and
        resident-slot tables are per request, so two steps of one request must
        never share a launch. Launch step-major batches instead: requests run
        in parallel while each request's later step observes the prior launch.
        Padding rows beyond ``sum(verify_lens)`` remain invalid and never launch.

        This eager synchronous path is intentional: speculative decoding disables
        shared-index/previous-layer prefetch, while preserving the Indexer's
        Top-K unchanged.
        """
        if top_k_result.dim() != 2:
            raise ValueError(
                "HiSparse speculative Top-K must be a 2-D flattened tensor, "
                f"got shape {tuple(top_k_result.shape)}"
            )
        batch_size = req_pool_indices.numel()
        if batch_size == 0:
            return top_k_result.to(torch.int32)
        if output_buffer is None:
            output_buffer = torch.empty_like(top_k_result, dtype=torch.int32)
        result = output_buffer[: top_k_result.size(0)]
        compressed_seq_lens = compressed_seq_lens.reshape(-1)

        if verify_lens_cpu is None:
            if top_k_result.size(0) % batch_size != 0:
                raise ValueError(
                    "Static HiSparse verify rows must be divisible by batch size: "
                    f"rows={top_k_result.size(0)}, batch={batch_size}"
                )
            verify_width = top_k_result.size(0) // batch_size
            top_k_by_req = top_k_result.view(batch_size, verify_width, -1)
            result_by_req = result.view(batch_size, verify_width, -1)
            if compressed_seq_lens.numel() == batch_size:
                seq_lens_by_req = None
            elif compressed_seq_lens.numel() == top_k_result.size(0):
                seq_lens_by_req = compressed_seq_lens.view(batch_size, verify_width)
            else:
                raise ValueError(
                    "Static HiSparse verify sequence lengths must contain one "
                    "value per request or Top-K row: "
                    f"lengths={compressed_seq_lens.numel()}, batch={batch_size}, "
                    f"rows={top_k_result.size(0)}"
                )
            # Static ownership is fixed and request-aligned, including CUDA
            # Graph padding. Launch one full request batch per step: the kernel's
            # num_real_reqs mask now correctly removes padded request blocks.
            for step in range(verify_width):
                self._run_swap_in_kernel(
                    req_pool_indices,
                    (
                        compressed_seq_lens
                        if seq_lens_by_req is None
                        else seq_lens_by_req[:, step]
                    ),
                    top_k_by_req[:, step, : self.top_k],
                    layer_id,
                    output_buffer=result_by_req[:, step, : self.top_k],
                )
            return result

        result.fill_(-1)
        real_rows = sum(verify_lens_cpu)
        if len(verify_lens_cpu) != batch_size or real_rows > top_k_result.size(0):
            raise ValueError(
                "HiSparse ragged verify geometry does not match Top-K rows: "
                f"lens={verify_lens_cpu}, rows={top_k_result.size(0)}"
            )

        num_seq_lens = compressed_seq_lens.numel()
        if num_seq_lens not in (batch_size, real_rows, top_k_result.size(0)):
            raise ValueError(
                "HiSparse verify sequence lengths must contain one value per "
                "request or Top-K row: "
                f"lengths={num_seq_lens}, batch={batch_size}, "
                f"rows={top_k_result.size(0)}"
            )

        # One launch per verify step, with all requests owning that step in the
        # same launch. This preserves per-request ordering while retaining
        # request-level GPU parallelism. Compact rows are request-major, hence
        # offsets identify the row for (request, step).
        offsets = [0]
        for verify_len in verify_lens_cpu:
            offsets.append(offsets[-1] + verify_len)
        for step in range(max(verify_lens_cpu, default=0)):
            active_reqs = [
                i for i, length in enumerate(verify_lens_cpu) if step < length
            ]
            row_ids = [offsets[i] + step for i in active_reqs]
            active_idx = torch.tensor(
                active_reqs, dtype=torch.int64, device=self.device
            )
            row_idx = torch.tensor(row_ids, dtype=torch.int64, device=self.device)
            step_seq_lens = (
                compressed_seq_lens[active_idx]
                if num_seq_lens == batch_size
                else compressed_seq_lens[row_idx]
            )
            step_output = torch.empty(
                (len(active_reqs), self.top_k), dtype=torch.int32, device=self.device
            )
            self._run_swap_in_kernel(
                req_pool_indices[active_idx],
                step_seq_lens,
                top_k_result[row_idx, : self.top_k],
                layer_id,
                output_buffer=step_output,
            )
            result[row_idx, : self.top_k] = step_output
        return result

    def prepare_dspark_verify_window(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        verify_width: int,
    ) -> HiSparseDSparkWindow:
        """Bind temporary physical C4 writer pages for one DSPARK verify."""
        if not self.is_dsv4_hisparse:
            raise RuntimeError("DSPARK HiSparse windows require DeepSeek-V4 C4")
        if self._has_pending_dspark_commit:
            # Commit copies may have been issued by another producer stream.
            # A stream dependency preserves overlap without a CPU-wide barrier.
            self._dspark_commit_done_event.wait(device_module.current_stream())
            self._has_pending_dspark_commit = False
        req_offsets: List[int] = []
        c4_positions: List[int] = []
        full_locs = []
        prefix_cpu = prefix_lens.to("cpu").tolist()
        req_cpu = req_pool_indices.to("cpu").tolist()
        for req_offset, (req_idx, prefix_len) in enumerate(zip(req_cpu, prefix_cpu)):
            for c4_pos in dspark_completed_c4_positions(prefix_len, verify_width):
                token_pos = (c4_pos + 1) * self.compress_ratio - 1
                req_offsets.append(req_offset)
                c4_positions.append(c4_pos)
                full_locs.append(
                    self.req_to_token_pool.req_to_token[req_idx, token_pos]
                )
        if not full_locs:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return HiSparseDSparkWindow(
                empty, empty, empty, [], req_cpu, [], prefix_cpu
            )
        full_locs = torch.stack(full_locs).to(torch.int64)
        compressed_locs = self.hisparse_kvcache.translate_loc_from_full_to_compressed(
            full_locs
        )
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        previous_device_mapping = mapping[compressed_locs].clone()
        if self._dspark_scratch_device_locs is None:
            raise RuntimeError("HiSparse DSPARK fixed C4 scratch is not initialized")
        if compressed_locs.numel() > self._dspark_scratch_device_locs.numel():
            raise RuntimeError(
                "HiSparse DSPARK verify exceeds fixed graph scratch capacity: "
                f"need {compressed_locs.numel()}, have "
                f"{self._dspark_scratch_device_locs.numel()}"
            )
        device_locs = self._dspark_scratch_device_locs[: compressed_locs.numel()]
        try:
            self._dspark_scratch_compressed_locs.fill_(-1)
            self._dspark_scratch_compressed_locs[: compressed_locs.numel()].copy_(
                compressed_locs
            )
            self._dspark_scratch_valid_mapping.fill_(False)
            self._dspark_scratch_valid_mapping[compressed_locs] = True
            mapping[compressed_locs] = device_locs.to(torch.int64)
            window = HiSparseDSparkWindow(
                compressed_locs,
                device_locs,
                previous_device_mapping,
                req_offsets,
                req_cpu,
                c4_positions,
                prefix_cpu,
            )
        except Exception:
            mapping[compressed_locs] = previous_device_mapping
            self._dspark_scratch_compressed_locs.fill_(-1)
            self._dspark_scratch_valid_mapping[compressed_locs] = False
            raise
        self._active_dspark_window = window
        return window

    def select_dspark_scratch_locs(
        self,
        compressed_locs: torch.Tensor,
        swapped_locs: torch.Tensor,
        mapped_device_locs: torch.Tensor,
    ) -> torch.Tensor:
        """Current-window C4 always reads its writer page, even on fast paths."""
        # Always execute this selection when fixed scratch exists. During CUDA
        # graph capture the identities are all -1; replay updates the same
        # device buffer before launch, so no capture-time Python branch or slice
        # address is frozen into the graph.
        if self._dspark_scratch_valid_mapping is None:
            return swapped_locs
        valid_logical_loc = torch.logical_and(
            compressed_locs >= 0,
            compressed_locs < self._dspark_scratch_valid_mapping.numel(),
        )
        safe_locs = compressed_locs.clamp(
            0, self._dspark_scratch_valid_mapping.numel() - 1
        )
        belongs_to_window = torch.logical_and(
            valid_logical_loc, self._dspark_scratch_valid_mapping[safe_locs]
        )
        return torch.where(belongs_to_window, mapped_device_locs, swapped_locs)

    def rollback_dspark_verify_window(self, window: HiSparseDSparkWindow) -> None:
        if window.compressed_locs.numel() == 0:
            return
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            window.compressed_locs
        ] = window.previous_device_mapping
        # Fixed scratch is coordinator-owned for its entire lifetime. Releasing
        # it here would invalidate addresses captured by CUDA Graph.
        self._dspark_scratch_compressed_locs.fill_(-1)
        self._dspark_scratch_valid_mapping[window.compressed_locs] = False
        if self._active_dspark_window is window:
            self._active_dspark_window = None

    def commit_dspark_verify_window(
        self, window: HiSparseDSparkWindow, commit_lens: torch.Tensor
    ) -> None:
        """Back up accepted C4 entries, then reset fixed scratch metadata."""
        if window.compressed_locs.numel() == 0:
            return
        commit_cpu = commit_lens.to("cpu").tolist()
        committed_copies = False
        accepted = [
            i
            for i, (req_offset, c4_pos) in enumerate(
                zip(window.req_offsets, window.c4_positions)
            )
            if (c4_pos + 1) * self.compress_ratio
            <= window.prefix_lens_cpu[req_offset] + int(commit_cpu[req_offset])
        ]
        if accepted:
            accepted_idx = torch.tensor(accepted, dtype=torch.int64, device=self.device)
            host_locs = []
            for i in accepted:
                req_idx = window.req_pool_indices_cpu[window.req_offsets[i]]
                host_locs.append(
                    self.mem_pool_host.alloc_paged_token_slots(
                        self.req_to_host_pool,
                        self.req_to_host_pool_allocated_len,
                        req_idx,
                        window.c4_positions[i],
                        1,
                    )
                )
            host_locs = torch.cat(host_locs)
            # Accepted entries are gathered across requests and are not a
            # contiguous, page-aligned range.  DeepSeekV4PagedHostPool's generic
            # backup dispatch treats a page-size multiple as whole pages, which
            # would retain only the first slot of each arbitrary group.  Always
            # use token-granular C4 transfer for speculative commit.
            transfer_cache_dsv4_mla(
                src_ptrs=self.mem_pool_host.device_ptrs,
                dst_ptrs=self.mem_pool_host.data_ptrs,
                src_indices=window.device_locs[accepted_idx].to(torch.int64),
                dst_indices=host_locs.to(torch.int64),
            )
            # Short-context C4s have distinct fixed slots. Long-context C4s all
            # share the reserved newest slot, so copy only the latest accepted
            # C4 per request to keep transfer destinations unique.
            fixed_copy = dspark_fixed_buffer_commit_indices(
                accepted,
                window.req_offsets,
                window.req_pool_indices_cpu,
                window.c4_positions,
                self.device_buffer_size,
            )
            fixed_idx = torch.tensor(fixed_copy, dtype=torch.int64, device=self.device)
            accepted_req_slots = [
                window.req_pool_indices_cpu[window.req_offsets[i]] for i in fixed_copy
            ]
            accepted_c4_pos = [window.c4_positions[i] for i in fixed_copy]
            fixed_slots = [min(pos, self.device_buffer_size) for pos in accepted_c4_pos]
            dst_locs = self.req_to_device_buffer[
                torch.tensor(accepted_req_slots, dtype=torch.int64, device=self.device),
                torch.tensor(fixed_slots, dtype=torch.int64, device=self.device),
            ]
            transfer_cache_dsv4_mla(
                src_ptrs=self.mem_pool_host.device_ptrs,
                dst_ptrs=self.mem_pool_host.device_ptrs,
                src_indices=window.device_locs[fixed_idx],
                dst_indices=dst_locs,
            )
            for req_idx, c4_pos, fixed_slot in zip(
                accepted_req_slots, accepted_c4_pos, fixed_slots
            ):
                self.req_device_buffer_tokens[:, req_idx, fixed_slot] = c4_pos
            committed_copies = True
        self.rollback_dspark_verify_window(window)
        if committed_copies:
            # Record after mapping rollback too: a subsequent verify may run on
            # another stream and must observe both the copies and metadata reset.
            self._dspark_commit_done_event.record(device_module.current_stream())
            self._has_pending_dspark_commit = True
