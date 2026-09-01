from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v4_speculative,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM


def _model(num_layers=12):
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.config = SimpleNamespace(num_hidden_layers=num_layers)
    model.model = SimpleNamespace(
        eagle3_layers_to_capture=None, dspark_layers_to_capture=None
    )
    model.capture_aux_hidden_states = False
    return model


def test_eagle3_explicit_layer_ids_are_output_ids_in_checkpoint_order():
    model = _model()

    model.set_eagle3_layers_to_capture([0, 5, 11])

    assert model.model.eagle3_layers_to_capture == [0, 5, 11]
    assert model.capture_aux_hidden_states


def test_eagle3_default_layer_ids_are_valid_completed_outputs():
    model = _model()

    model.set_eagle3_layers_to_capture()

    assert model.model.eagle3_layers_to_capture == [1, 5, 8]
    assert all(0 <= layer_id < 12 for layer_id in model.model.eagle3_layers_to_capture)


@pytest.mark.parametrize(
    "layer_ids", [[], [1, 1, 2], [5, 1, 8], [-1, 2, 3], [1, 2, 12]]
)
def test_eagle3_rejects_invalid_layer_ids(layer_ids):
    with pytest.raises(ValueError, match="EAGLE3 layer IDs"):
        _model().set_eagle3_layers_to_capture(layer_ids)


def test_eagle3_and_dspark_capture_are_mutually_exclusive():
    model = _model()
    model.set_eagle3_layers_to_capture([1, 5, 8])

    with pytest.raises(ValueError, match="cannot be enabled together"):
        model.set_dspark_layers_to_capture([1, 5, 8])


def test_capture_setup_rejects_eagle3_and_dspark_together():
    with pytest.raises(ValueError, match="cannot be enabled at the same time"):
        configure_aux_hidden_state_capture(
            model=object(),
            eagle_use_aux_hidden_state=True,
            eagle_aux_hidden_state_layer_ids=[1, 5, 8],
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=[1, 5, 8],
            is_dspark=True,
        )


@pytest.mark.parametrize("algorithm", [None, "EAGLE", "EAGLE3", "DSPARK"])
def test_deepseek_v4_supported_speculative_algorithms(algorithm):
    args = SimpleNamespace(
        speculative_algorithm=algorithm, speculative_eagle_topk=1, pp_size=1
    )

    validate_deepseek_v4_speculative(args, "DeepseekV4ForCausalLM")


def test_deepseek_v4_eagle3_rejects_topk_greater_than_one():
    args = SimpleNamespace(
        speculative_algorithm="EAGLE3", speculative_eagle_topk=2, pp_size=1
    )

    with pytest.raises(ValueError, match="speculative-eagle-topk 1"):
        validate_deepseek_v4_speculative(args, "DeepseekV4ForCausalLM")


def test_deepseek_v4_eagle3_rejects_pipeline_parallelism():
    args = SimpleNamespace(
        speculative_algorithm="EAGLE3", speculative_eagle_topk=1, pp_size=2
    )

    with pytest.raises(ValueError, match="cross-stage capture is not implemented"):
        validate_deepseek_v4_speculative(args, "DeepseekV4ForCausalLM")
