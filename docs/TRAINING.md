# Training AFTERAI with a phone-only setup

The Android phone is for editing, repository control, and small checks. Do not try to train a language model on the phone.

## Free-first workflow

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

Choose a small causal language model whose license permits your intended use. Verify its model card, license, hardware needs, and tokenizer chat template before downloading it.

    !python scripts/train_lora.py --model YOUR_APPROVED_MODEL_ID --dataset data/your_reviewed_dataset.jsonl --output /content/afterai-lora-output --epochs 1

Replace the placeholder with the exact model ID you reviewed. The script defaults to attention modules named q_proj,v_proj; some architectures use different names. Check the model architecture and pass --target-modules when necessary.

## Important limits

- Colab free resources are shared, variable, and subject to usage limits and disconnections.
- A GPU may not be available. Do not switch to a paid runtime without an explicit budget decision.
- The initial script is a minimal supervised LoRA baseline, not a production training system.
- Checkpoint and dataset storage is temporary unless you explicitly save it somewhere.
- Never put access tokens, passwords, or private datasets into Git.
