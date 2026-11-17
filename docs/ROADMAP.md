# AFTERAI roadmap

## Phase 0 — repository foundation
- [x] Create project README and repository layout.
- [x] Add Python package and command-line interface.
- [x] Define and validate chat JSONL format.
- [x] Add CPU-only CI workflow.
- [x] Add an opt-in LoRA training starter.
- [ ] Confirm CI passes on GitHub.
- [ ] Choose a base model after checking its license, size, tokenizer, and hardware needs.

## Phase 1 — baseline assistant
- [ ] Add a documented model adapter for one selected base model.
- [ ] Add a small inference command.
- [ ] Create a hand-reviewed evaluation set in Portuguese and English.
- [ ] Record baseline results before fine-tuning.

## Phase 2 — data and fine-tuning
- [ ] Create a small, permission-cleared dataset.
- [ ] Run a first LoRA experiment in a free GPU notebook when available.
- [ ] Save training configuration, logs, and evaluation results.
- [ ] Compare the adapted model with the untouched baseline.

## Phase 3 — independent capabilities
- [ ] Conversation/session API.
- [ ] Pluggable local memory with user-controlled retention.
- [ ] Tool interface with explicit allowlists and confirmation for consequential actions.
- [ ] Model-independent evaluation and regression suite.

## Phase 4 — original research
- [ ] Implement a small educational decoder-only transformer.
- [ ] Train it on a tiny, properly licensed corpus.
- [ ] Publish limitations and benchmark results honestly.

## Non-goals for the zero-budget phase

- Training a frontier-scale model from scratch.
- Assuming continuous access to a GPU.
- Downloading large model weights onto the phone.
- Automatically publishing datasets or checkpoints.
