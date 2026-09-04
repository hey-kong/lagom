"""OasisKV one-token look-ahead lane.

This module intentionally has no acceptance API.  It coordinates an EAGLE-3
token predictor with a *target-model* no-commit propagation and sends only the
resulting target-layer C4 predictions to HiSparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class OasisKVAttentionView:
    """The immutable history view shared by normal and draft lanes."""

    c4_sparse_locations: Any
    c128_metadata: Any
    swa_metadata: Any
    local_window_metadata: Any
    committed_sequence_lengths: Any


@dataclass(frozen=True)
class OasisKVPrediction:
    layer_id: int
    source_committed_sequence_lengths: Any
    target_token_positions: Any
    c4_entries: Any


class OasisKVLookaheadLane:
    """Correctness-baseline eager target propagation for OasisKV.

    The supplied ``target_no_commit`` callback must return target-model queries
    keyed by layer id.  It receives the exact normal-lane attention view and is
    responsible for scratch/inline KV lifetime.  State snapshots make accidental
    mutation a hard error rather than allowing a prediction to affect output.
    """

    def __init__(
        self,
        *,
        eagle3_draft_one: Callable[..., Any],
        target_no_commit: Callable[..., Mapping[int, Any]],
        target_c4_predict: Callable[[int, Any], Any],
        submit_prefetch: Callable[..., None],
        snapshot_committed_state: Callable[[], Any],
    ):
        self._draft_one = eagle3_draft_one
        self._target_no_commit = target_no_commit
        self._predict = target_c4_predict
        self._submit = submit_prefetch
        self._snapshot = snapshot_committed_state
        self.draft_target_forward_seconds = 0.0

    def run(
        self,
        *,
        normal_hidden_states: Any,
        attention_view: OasisKVAttentionView,
        request_metadata: Mapping[str, Any],
    ) -> list[OasisKVPrediction]:
        before = self._snapshot()
        draft_token = self._draft_one(normal_hidden_states=normal_hidden_states)
        start = perf_counter()
        target_queries = self._target_no_commit(
            token=draft_token,
            attention_view=attention_view,
            scratch_kv=True,
            commit=False,
        )
        self.draft_target_forward_seconds += perf_counter() - start
        if self._snapshot() != before:
            raise RuntimeError(
                "OasisKV no-commit target forward mutated committed state"
            )

        predictions = []
        target_positions = attention_view.committed_sequence_lengths + 1
        for layer_id, target_query in target_queries.items():
            entries = self._predict(layer_id, target_query)
            prediction = OasisKVPrediction(
                layer_id=layer_id,
                source_committed_sequence_lengths=attention_view.committed_sequence_lengths,
                target_token_positions=target_positions,
                c4_entries=entries,
            )
            self._submit(prediction=prediction, request_metadata=request_metadata)
            predictions.append(prediction)
        return predictions
