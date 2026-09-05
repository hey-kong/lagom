"""OasisKV's one-token EAGLE-3 look-ahead execution contract.

OasisKV is not speculative decoding: the draft token is only a prefetch hint.
The target receives ``[normal, draft]`` in one width-two forward and commits
lane zero only.  Keeping this contract in a small, device-independent class
makes the particularly important state and position rules testable without a
DeepSeek checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class OasisKVAttentionView:
    """Immutable history selected by the normal query and shared by both lanes."""

    c4_sparse_locations: Any
    c128_metadata: Any
    swa_metadata: Any
    local_window_metadata: Any
    committed_sequence_lengths: Any


@dataclass(frozen=True)
class OasisKVPairedBatch:
    """Request-major paired input: rows ``2*i``/``2*i+1`` are normal/draft.

    ``committed_lens`` is the history length *before* the normal token.  Hence
    normal position is ``L``, draft position is ``L+1``, and the next decode
    starts with committed length and token position ``L+1``.
    """

    request_slots: Sequence[int]
    generations: Sequence[int]
    normal_tokens: Any
    draft_tokens: Any
    committed_lens: Any
    positions: Any
    attention_view: OasisKVAttentionView


@dataclass(frozen=True)
class OasisKVPairedOutput:
    """Output of exactly one target traversal over a paired batch."""

    normal_output: Any
    normal_features: Any
    draft_queries: Mapping[int, Any]
    normal_sparse_locations: Mapping[int, Any]
    draft_sparse_locations: Mapping[int, Any]


@dataclass(frozen=True)
class OasisKVPrediction:
    layer_id: int
    source_committed_sequence_lengths: Any
    target_token_positions: Any
    c4_entries: Any


@dataclass
class OasisKVProfile:
    paired_decode_steps: int = 0
    draft_tokens_generated: int = 0
    target_layer_traversals: int = 0
    draft_c4_predictions: int = 0
    submitted_prefetch_entries: int = 0
    speculative_acceptance: int = 0


class OasisKVFeatureStore:
    """Generation-qualified EAGLE features, safe across dynamic batch slots."""

    def __init__(self):
        self._features: dict[tuple[int, int], Any] = {}

    def get(self, slot: int, generation: int) -> Any:
        return self._features.get((int(slot), int(generation)))

    def put(self, slot: int, generation: int, features: Any) -> None:
        key = (int(slot), int(generation))
        # Slot reuse invalidates all older generations immediately.
        self.clear_slot(slot)
        if features is not None:
            self._features[key] = features

    def clear_slot(self, slot: int) -> None:
        slot = int(slot)
        for key in [key for key in self._features if key[0] == slot]:
            del self._features[key]

    def seed_prefill(
        self, slots: Sequence[int], generations: Sequence[int], features: Sequence[Any]
    ) -> None:
        """Save normal target features produced by prefill."""
        for slot, generation, feature in zip(slots, generations, features):
            self.put(slot, generation, feature)

    def note_finished(self, slots: Sequence[int]) -> None:
        for slot in slots:
            self.clear_slot(slot)


def paired_causal_mask(history_lens: Sequence[int]) -> torch.Tensor:
    """Build the reference mask for ``history -> normal -> draft``.

    Rows are request-major normal/draft lanes.  This reference implementation
    is intended for eager/extend backends and tests; optimized backends may
    encode the same geometry in their sequence metadata.
    """
    lens = [int(length) for length in history_lens]
    width = max(lens, default=0) + 2
    mask = torch.zeros((len(lens), 2, width), dtype=torch.bool)
    for request, length in enumerate(lens):
        mask[request, 0, : length + 1] = True
        mask[request, 1, : length + 2] = True
    return mask


@dataclass(frozen=True)
class OasisKVLayerOutput:
    hidden_states: Any
    draft_query: Any
    shared_sparse_locations: Any


class OasisKVPairedTargetExecutor:
    """Reference width-two target traversal used by eager integrations.

    ``run_layer`` is called once per target layer with the two lanes together.
    A DeepSeek-V4 adapter performs QKV, attention and FFN/MoE inside that one
    call and returns the target Q-projection for draft C4 indexing.  It must
    direct lane-one cache writes to scratch; only ``commit_normal`` may mutate
    scheduler-visible state.
    """

    def __init__(
        self,
        *,
        layers: Sequence[Any],
        run_layer: Callable[..., OasisKVLayerOutput],
        commit_normal: Callable[..., Any],
    ):
        self.layers = layers
        self.run_layer = run_layer
        self.commit_normal = commit_normal
        self.layer_calls = 0

    def __call__(self, batch: OasisKVPairedBatch) -> OasisKVPairedOutput:
        hidden = (batch.normal_tokens, batch.draft_tokens)
        queries = {}
        normal_locations = {}
        draft_locations = {}
        mask = paired_causal_mask(batch.committed_lens)
        for layer_id, layer in enumerate(self.layers):
            result = self.run_layer(
                layer=layer, hidden_states=hidden, batch=batch, causal_mask=mask
            )
            self.layer_calls += 1
            hidden = result.hidden_states
            queries[layer_id] = result.draft_query
            # Both dictionary values deliberately alias the same selection.
            normal_locations[layer_id] = result.shared_sparse_locations
            draft_locations[layer_id] = result.shared_sparse_locations
        normal_output, normal_features = self.commit_normal(hidden[0], batch=batch)
        return OasisKVPairedOutput(
            normal_output=normal_output,
            normal_features=normal_features,
            draft_queries=queries,
            normal_sparse_locations=normal_locations,
            draft_sparse_locations=draft_locations,
        )


def submit_oasiskv_prediction(
    *, prediction: OasisKVPrediction, request_metadata: Mapping[str, Any]
) -> None:
    """Bridge a paired target C4 prediction to PR #10's prefetch ring."""
    coordinator = request_metadata["hisparse_coordinator"]
    coordinator.submit_oasiskv_prefetch(
        req_pool_indices=request_metadata["req_pool_indices"],
        req_pool_indices_cpu=request_metadata["req_pool_indices_cpu"],
        compressed_seq_lens=request_metadata["compressed_seq_lens"],
        source_committed_lens_cpu=prediction.source_committed_sequence_lengths,
        predicted_c4_entries=prediction.c4_entries,
        layer_id=prediction.layer_id,
    )


class OasisKVLookaheadLane:
    """Run EAGLE once, then one paired target forward, committing lane zero.

    ``paired_target_forward`` is the production integration point.  Its target
    implementation must perform one QKV projection, attention call and FFN/MoE
    call per layer over both rows, use a ``history -> normal -> draft`` mask,
    and write draft K/V and C4 to scratch storage.  The returned per-layer
    locations are checked to prevent a draft-only sparse working set.
    """

    def __init__(
        self,
        *,
        eagle3_draft_one: Callable[..., Any],
        paired_target_forward: Callable[[OasisKVPairedBatch], OasisKVPairedOutput],
        target_c4_predict: Callable[[int, Any], Any],
        submit_prefetch: Callable[..., None],
        snapshot_draft_state: Callable[[], Any],
        feature_store: OasisKVFeatureStore | None = None,
        profile: OasisKVProfile | None = None,
    ):
        self._draft_one = eagle3_draft_one
        self._paired_forward = paired_target_forward
        self._predict = target_c4_predict
        self._submit = submit_prefetch
        self._snapshot = snapshot_draft_state
        self.features = feature_store or OasisKVFeatureStore()
        self.profile = profile or OasisKVProfile()

    def run(
        self,
        *,
        normal_tokens: Any,
        attention_view: OasisKVAttentionView,
        request_slots: Sequence[int],
        generations: Sequence[int],
        request_metadata: Mapping[str, Any],
    ) -> tuple[Any, list[OasisKVPrediction]]:
        if len(request_slots) != len(generations):
            raise ValueError("OasisKV slot/generation batch sizes differ")
        features = [
            self.features.get(slot, generation)
            for slot, generation in zip(request_slots, generations)
        ]
        # Missing first-round/retired features disable look-ahead for the whole
        # dynamic batch; normal HiSparse remains authoritative and no stale or
        # synthetic draft is ever used.
        if any(feature is None for feature in features):
            return None, []

        draft_tokens = self._draft_one(
            normal_hidden_states=features,
            normal_tokens=normal_tokens,
            request_slots=request_slots,
        )
        self.profile.draft_tokens_generated += len(request_slots)
        committed_lens = attention_view.committed_sequence_lengths
        paired = OasisKVPairedBatch(
            request_slots=request_slots,
            generations=generations,
            normal_tokens=normal_tokens,
            draft_tokens=draft_tokens,
            committed_lens=committed_lens,
            positions=(committed_lens, committed_lens + 1),
            attention_view=attention_view,
        )
        before = self._snapshot()
        output = self._paired_forward(paired)
        self.profile.paired_decode_steps += 1
        self.profile.target_layer_traversals += len(output.draft_queries)
        if self._snapshot() != before:
            raise RuntimeError("OasisKV paired forward leaked draft scratch state")

        predictions = []
        target_positions = committed_lens + 1
        for layer_id, draft_query in output.draft_queries.items():
            if (
                output.normal_sparse_locations[layer_id]
                is not output.draft_sparse_locations[layer_id]
            ):
                raise RuntimeError(
                    "OasisKV draft lane selected a different sparse working set"
                )
            entries = self._predict(layer_id, draft_query)
            prediction = OasisKVPrediction(
                layer_id=layer_id,
                source_committed_sequence_lengths=committed_lens,
                target_token_positions=target_positions,
                c4_entries=entries,
            )
            self._submit(prediction=prediction, request_metadata=request_metadata)
            self.profile.draft_c4_predictions += len(request_slots)
            self.profile.submitted_prefetch_entries += getattr(
                entries, "numel", lambda: len(entries)
            )()
            predictions.append(prediction)

        # The target callback owns the atomic normal commit.  Only its normal
        # features are retained for the following EAGLE invocation.
        for slot, generation, feature in zip(
            request_slots, generations, output.normal_features
        ):
            self.features.put(slot, generation, feature)
        return output.normal_output, predictions
