"""Fused PRR-style EMA score update for DeepSeek-V4 C4 prefetch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ema_update_forecast_kernel(
    scores,
    req_indices,
    seq_lens,
    state_lens,
    levels,
    trends,
    forecast,
    score_stride,
    forecast_stride,
    state_stride,
    width: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    gamma: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < width
    req = tl.load(req_indices + batch)
    length = tl.load(seq_lens + batch)
    history_len = tl.load(state_lens + req)
    valid = mask & (offsets < length)
    history = valid & (offsets < history_len)

    score = tl.load(scores + batch * score_stride + offsets, mask=mask, other=0.0)
    old_level = tl.load(levels + req * state_stride + offsets, mask=history, other=0.0)
    old_trend = tl.load(trends + req * state_stride + offsets, mask=history, other=0.0)
    new_level = tl.where(history, alpha * score + (1.0 - alpha) * old_level, score)
    new_trend = tl.where(
        history,
        beta * (score - old_level) + (1.0 - beta) * old_trend,
        0.0,
    )
    prediction = tl.where(valid, new_level + gamma * new_trend, -float("inf"))

    tl.store(levels + req * state_stride + offsets, new_level, mask=valid)
    tl.store(trends + req * state_stride + offsets, new_trend, mask=valid)
    tl.store(forecast + batch * forecast_stride + offsets, prediction, mask=mask)


@triton.jit
def _ema_update_lengths_kernel(
    req_indices,
    seq_lens,
    state_lens,
    prediction_lens,
    batch_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = batch < batch_size
    req = tl.load(req_indices + batch, mask=mask, other=0)
    length = tl.load(seq_lens + batch, mask=mask, other=0)
    history_len = tl.load(state_lens + req, mask=mask, other=0)
    # A zero selection length makes the existing DSV4 Top-K emit only -1 for
    # the first observation while mature rows retain their real C4 length.
    tl.store(prediction_lens + batch, tl.where(history_len > 0, length, 0), mask=mask)
    tl.store(state_lens + req, length, mask=mask)


def ema_update_forecast(
    scores: torch.Tensor,
    req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    levels: torch.Tensor,
    trends: torch.Tensor,
    state_lens: torch.Tensor,
    prediction_lens: torch.Tensor,
    forecast: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Update persistent request state and write the next-score forecast."""
    assert scores.is_cuda and scores.dtype == torch.float32 and scores.ndim == 2
    assert req_indices.is_cuda and seq_lens.is_cuda
    assert levels.dtype == trends.dtype == torch.float32
    batch_size, width = scores.shape
    assert forecast.shape == scores.shape and forecast.dtype == torch.float32
    block = 256
    _ema_update_forecast_kernel[(batch_size, triton.cdiv(width, block))](
        scores,
        req_indices,
        seq_lens,
        state_lens,
        levels,
        trends,
        forecast,
        scores.stride(0),
        forecast.stride(0),
        levels.stride(0),
        width,
        alpha,
        beta,
        gamma,
        BLOCK=block,
    )
    length_block = triton.next_power_of_2(batch_size)
    _ema_update_lengths_kernel[(1,)](
        req_indices,
        seq_lens,
        state_lens,
        prediction_lens,
        batch_size,
        BLOCK=length_block,
    )
    return forecast
