# AFTERAI architecture (initial)

This is the first implementation boundary, not a claim that a complete AI system already exists.

## Layers

1. Data — documented chat JSONL format, validation, and dataset provenance.
2. Model adapter — load a user-selected compatible pretrained causal language model.
3. Fine-tuning — optional supervised fine-tuning with LoRA in a GPU notebook.
4. Runtime — future inference interface, conversation handling, and model-independent APIs.
5. Memory and tools — future opt-in modules with explicit storage and permission boundaries.
6. Evaluation — tests and curated evaluations before claiming a capability.

## Current implementation

- Python package and CLI.
- JSONL chat dataset validator.
- Tiny synthetic sample and tests.
- Optional LoRA training script.
- CPU-only GitHub Actions checks.

## Design constraints

- Review the base-model license and hardware requirements before use.
- Local utilities must not require cloud APIs.
- Training must never run automatically in CI.
- Distinguish project code from third-party model weights and datasets.
