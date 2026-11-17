# Training AFTERAI with a phone-only setup

The Android phone is for editing, repository control, and small checks. Do not try to train a language model on the phone.

## First baseline model

The initial candidate is Qwen/Qwen3-0.6B. Its model card lists 0.6 billion parameters and an Apache 2.0 license. Use a recent Transformers release (4.51.0 or newer) because older releases do not recognize the Qwen3 model type.

Review the current model card and its terms before use: https://huggingface.co/Qwen/Qwen3-0.6B

## Open Colab on Android

Open the [AFTERAI Colab quickstart notebook](https://colab.research.google.com/github/huzeifaumar34/AFTERAI/blob/main/notebooks/AFTERAI_Colab_Quickstart.ipynb). The notebook clones the repository, installs the optional dependencies, checks whether a GPU was assigned, and lets you upload one reviewed JSONL dataset.

Training is disabled by default. Set RUN_TRAINING to True only after checking the dataset and GPU availability. The notebook also leaves adapter download disabled until you explicitly enable it.

## Manual free-first workflow

1. Open a Google Colab notebook in the browser.
2. Select a GPU runtime only when one is offered. Free GPU access is not guaranteed.
3. Clone this public repository in the notebook.
4. Install the optional training dependencies.
5. Upload or create a small dataset that you have permission to use.
6. Run dataset validation and then the training script.
7. Download the adapter output and logs before the temporary session ends.
8. Record the base model ID and revision, dataset provenance, settings, and evaluation results.

Example Colab cells:

    !git clone https://github.com/huzeifaumar34/AFTERAI.git
    %cd AFTERAI
    !python -m pip install -e ".[training]"
    !afterai validate-data data/examples/tiny_sample.jsonl

The synthetic sample is only for checking the pipeline. It is far too small to produce a useful assistant.

## Start a LoRA experiment

The reviewed starter configuration uses Qwen/Qwen3-0.6B. Check the latest model card and available GPU memory before running:

    !python scripts/train_lora.py --model Qwen/Qwen3-0.6B --dataset data/your_reviewed_dataset.jsonl --output /content/afterai-lora-output --epochs 1

The script defaults to LoRA attention modules named q_proj and v_proj, which must be confirmed against the selected model architecture before training.

## Important limits

- Colab free resources are shared, variable, and subject to usage limits and disconnections.
- A GPU may not be available. Do not switch to a paid runtime without an explicit budget decision.
- The initial script is a minimal supervised LoRA baseline, not a production training system.
- Checkpoint and dataset storage is temporary unless you explicitly save it somewhere.
- Never put access tokens, passwords, or private datasets into Git.
