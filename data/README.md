# AFTERAI dataset format

Training data uses JSON Lines (.jsonl): one JSON object per line.

Each object has a messages array using a simple chat format:

    {"messages":[{"role":"user","content":"Olá!"},{"role":"assistant","content":"Olá! Como posso ajudar?"}]}

Allowed roles are system, user, and assistant. Every record must have at least one user message and one assistant message. Content must be non-empty text.

## Data safety and quality

- Start with small, manually reviewed examples.
- Do not include passwords, API keys, private conversations, personal records, or information collected without permission.
- Do not copy datasets or books unless their license permits the intended use.
- Remove duplicates and check examples for factual errors, harmful instructions, and personally identifying information.
- Keep a record of the source, license, date, and transformations for every dataset.
- A successful format check does not prove that a dataset is safe or high quality.

The file in examples/ is synthetic demonstration data only; it is not a meaningful training corpus.
