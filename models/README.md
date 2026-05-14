# Confidence estimators

Four methods, organized by the family they belong to.

| method | family | files |
|---|---|---|
| **P(True)** | prompt-based self-evaluation | `ptrue.py` — runs the LLM, asks the Supp Table 8 prompt, extracts softmax over `a`/`b` tokens. No training. |
| **P(IK)** | static hidden-state probe | `pik/extract_hidden_states.py` (final input-token state) + `pik/train.py` (MLP probe). |
| **SAPLMA** | static post-answer probe | `saplma/extract_hidden_states.py` (final answer-token state at three layers: final / upper_middle / middle) + `saplma/train.py`. Each layer trains an independent classifier; the paper reports SAPLMA-F by default. |
| **CCPS** | representational stability | `ccps/extract_hidden_states.py` (original + perturbed hidden states and logits — PEI radius 20, 5 steps) → `ccps/extract_features.py` (the 75-dim feature vector documented in Supp Table 9) → `ccps/contrastive_train.py` (max-margin contrastive encoder) → `ccps/classifier_train.py` (cross-entropy head). |

`ccps/nets.py` holds the encoder architecture and loss definitions.

The shell launchers in `scripts/02_extract_representations.sh` and `scripts/03_train_confidence_estimators.sh` wire these together with the model list (Qwen2.5 0.5B/1.5B/3B-Instruct, Llama-3.2 1B/3B-Instruct) and the round-robin GPU assignment used in the paper.
