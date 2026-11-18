import pytest

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"  # random weights, real Qwen2.5 tokenizer + template


@pytest.fixture(scope="session")
def engine():
    from quickverdict.engine import Engine

    return Engine(TINY, device="cpu", dtype="float32")
