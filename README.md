# AFTERAI

**An independent, open-source AI research and assistant project.**

AFTERAI is being built with a phone-first workflow: Android/Termux for local editing, GitHub for source control and automation, and free cloud notebooks for occasional model experiments.

> **Current status:** bootstrap / Phase 0. This repository does not contain a trained model yet. No paid cloud services are required or assumed.

## Project goals

- Build our own application, data pipeline, evaluation suite, and assistant runtime.
- Start from a small openly licensed pretrained model, then experiment with parameter-efficient fine-tuning (LoRA).
- Keep model choice, license review, datasets, and evaluation results explicit and reproducible.
- Research a small model architecture of our own in parallel, without pretending it is competitive before testing.

## What AFTERAI is — and is not

AFTERAI is not a renamed API wrapper. The repository will own the surrounding software, data formats, training/evaluation code, memory and tool interfaces, and eventually experimental model weights if we can train them.

The initial pretrained model is a replaceable dependency. Its license and usage conditions must be checked before distribution. Fine-tuning a model does not make its original architecture or base weights ours.

## Repository map

```text
.
├── .github/workflows/ci.yml    # CPU-only checks on GitHub Actions
├── config/                     # Example settings
├── data/                       # Dataset format and tiny synthetic example
├── docs/                       # Architecture, roadmap and training guide
├── scripts/                    # Dataset validation and LoRA training
├── src/afterai/                # Python package and CLI
└── tests/                      # Automated tests
```

## Quick start (Android / Linux)

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
afterai info
afterai validate-data data/examples/tiny_sample.jsonl
pytest
```

On Termux, install Python and Git using the Termux package manager first. A virtual environment is recommended; the training dependencies are intentionally not installed by the normal development setup.

## Training

Training is optional and is **not** run by the normal CI workflow. Use a free GPU notebook such as Google Colab when GPU access is available. Free GPU availability and session duration are limited and can change.

See [docs/TRAINING.md](docs/TRAINING.md). The starter script uses supervised fine-tuning with LoRA and expects a JSONL dataset in the format described in [data/README.md](data/README.md). Do not upload private conversations, credentials, or personal data.

## Principles

1. **No-cost first:** no paid GPU or service is assumed.
2. **Small and measurable:** run tiny tests before expensive experiments.
3. **License-aware:** verify licenses for code, models, and datasets independently.
4. **Reproducible:** record model ID, revision, dataset version, and training settings.
5. **Honest evaluation:** distinguish a working prototype from a capable AI system.

## License

The project code is released under the MIT License. Pretrained model weights and external datasets retain their own licenses and terms; the project license does not override them.
