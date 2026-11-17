"""Small, model-agnostic chat inference for AFTERAI."""

from __future__ import annotations


def generate_reply(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "auto",
) -> str:
    """Generate one reply using a compatible causal language model."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than zero.")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Runtime dependencies are missing. Install them with: "
            "pip install -e '.[runtime]'"
        ) from exc

    cuda_available = torch.cuda.is_available()
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
    selected_device = (
        "cuda" if device == "cuda" or (device == "auto" and cuda_available) else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "The selected tokenizer has no chat template. Choose a compatible chat model."
        )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if selected_device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(selected_device)
    model.eval()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(selected_device)

    generation_options = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    if tokenizer.pad_token_id is not None:
        generation_options["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**generation_options)

    reply_ids = generated_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
