import pytest

from afterai.inference import generate_reply


def test_generate_rejects_empty_prompt():
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        generate_reply("Qwen/Qwen3-0.6B", "   ")


def test_generate_rejects_invalid_device():
    with pytest.raises(ValueError, match="device must be one of"):
        generate_reply("Qwen/Qwen3-0.6B", "Olá", device="tpu")


def test_generate_rejects_zero_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_reply("Qwen/Qwen3-0.6B", "Olá", max_new_tokens=0)
