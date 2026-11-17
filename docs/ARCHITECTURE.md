# AFTERAI architecture (initial)

This is the first implementation boundary, not a claim that a complete AI system already exists.

## Layers

1. **Data** — documented chat JSONL format, validation, and dataset provenance.
2. **Model adapter** — model-agnostic loading of a compatible causal language model.
3. **Inference** — single-prompt chat generation through the AFTERAI CLI.
4. **Fine-tuning** — optional supervised fine-tuning with LoRA in a GPU notebook.
5. **Runtime services** — future conversation/session management and model-independent APIs.
6. **Memory and tools** — future opt-in modules with explicit storage and permission boundaries.
7. **Evaluation** — tests and curated evaluations before claiming a capability.

## Current implementation

- Python package and CLI with info, dataset-validation, and single-prompt inference commands.
- JSONL chat dataset validator with automated tests.
- Tiny synthetic sample and Colab quickstart notebook.
- Optional LoRA training script for the Qwen3-0.6B baseline candidate.
- CPU-only GitHub Actions quality workflow.

## Design constraints

- Review the base-model license and hardware requirements before use.
- Local utilities must not require cloud APIs.
- Training must never run automatically in CI.
- Do not commit model weights or private training data.
- Distinguish project code from third-party model weights and datasets.
