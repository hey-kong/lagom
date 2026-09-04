from types import SimpleNamespace

from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens


def _args(*, algorithm="EAGLE3", enable_hisparse=False):
    return SimpleNamespace(
        enable_two_batch_overlap=False,
        enable_hisparse=enable_hisparse,
        speculative_algorithm=algorithm,
    )


def test_hisparse_eagle3_requires_cpu_seq_lens():
    backend = SimpleNamespace(needs_cpu_seq_lens=False)

    assert decide_needs_cpu_seq_lens(
        _args(enable_hisparse=True), (backend, backend)
    )


def test_eagle3_without_hisparse_keeps_backend_cpu_seq_lens_policy():
    backend = SimpleNamespace(needs_cpu_seq_lens=False)

    assert not decide_needs_cpu_seq_lens(_args(), (backend, backend))
