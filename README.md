# gpt2-classifier

A from-scratch GPT-2 (book-style architecture) fine-tuned as a text
classifier. It ships with SMS spam and email spam datasets out of the box,
but the dataset pipeline is source-agnostic — point it at any two-column
text/label CSV (local or remote) and it works without touching the code.

## Setup

### Local (pixi)

Requires [pixi](https://pixi.sh/) installed.

```bash
pixi install       # installs torch, pandas, tiktoken, safetensors, tqdm,
                    # matplotlib, requests, wandb, pytest
pixi run pytest    # verify the install (56 fast, CPU-only unit tests, no network)
```

No GPU is required — `torch` resolves to a CPU wheel on a machine without
one. On a CUDA-enabled machine (e.g. Google Colab, below), the same code
picks up the GPU automatically; no flags or code changes needed.

### Google Colab

```python
!git clone https://github.com/paymantohidifar/gpt2-text-classifier-from-scratch.git --branch main gpt2_classifier
%cd gpt2-classifier
!pip install -e .
```

Then, in the Colab menu: **Runtime → Change runtime type → GPU**. All
device selection in this project uses `torch.cuda.is_available()`, so
training and inference automatically use the GPU once one is attached —
run the same `python -m gpt2_classifier ...` commands as below.

## CLI usage

All commands are run as `python -m gpt2_classifier <subcommand> ...` (or
`pixi run python -m gpt2_classifier ...` locally).

| Command | Description | Example |
|---|---|---|
| `download-weights` | Fetch a pretrained GPT-2 checkpoint from HuggingFace. | `python -m gpt2_classifier download-weights --model-name "gpt2-small (124M)"` |
| `train` | Fine-tune a classifier on a built-in or custom dataset (see flags below). | `python -m gpt2_classifier train --dataset sms-spam --num-epochs 5` |
| `predict` | Classify a piece of text with a fine-tuned checkpoint. | `python -m gpt2_classifier predict --checkpoint models/sms-spam_classifier.pt --text "..."` |
| `generate` | Raw GPT-2 text generation (debug helper to sanity-check pretrained weights; not classification). | `python -m gpt2_classifier generate --prompt "Every effort moves"` |

### `train` flags

| Flag | Description |
|---|---|
| `--dataset {sms-spam,email-spam}` | Use a built-in registered dataset (default `sms-spam`). |
| `--data-url` / `--data-path` | Bring your own dataset from a remote URL or local file instead. |
| `--text-column` / `--label-column` / `--label-map` | Required with `--data-url`/`--data-path` — describe the raw columns and label encoding, e.g. `--label-map "ham=0,spam=1"`. |
| `--num-epochs` | Number of fine-tuning epochs (default `5`). |
| `--use-wandb` | Enable Weights & Biases experiment tracking. |
| `--ddp` | Multi-GPU training via `torchrun`. |

## Examples

**Quickstart: SMS spam classifier end to end**

```bash
python -m gpt2_classifier download-weights --model-name "gpt2-small (124M)"
python -m gpt2_classifier train --model-name "gpt2-small (124M)" --dataset sms-spam --num-epochs 5
python -m gpt2_classifier predict --checkpoint models/sms-spam_classifier.pt \
  --text "Congratulations! You've won a free prize, call now to claim!!!"
# -> spam
```

**Using the built-in email spam dataset instead**

```bash
python -m gpt2_classifier train --model-name "gpt2-small (124M)" --dataset email-spam --num-epochs 5
python -m gpt2_classifier predict --checkpoint models/email-spam_classifier.pt --text "Hey, lunch tomorrow?"
# -> ham
```

**Bringing your own dataset**

```bash
python -m gpt2_classifier train \
  --data-path my_reviews.csv \
  --text-column review_text \
  --label-column sentiment \
  --label-map "negative=0,positive=1" \
  --num-epochs 5
```

## Licensing

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Portions of this software are derived from work by Sebastian Raschka, originally licensed under the Apache License, Version 2.0. A copy of the Apache License is included in [LICENSE-APACHE](LICENSE-APACHE).

## Acknowledgments & Citations

This repository builds upon the implementations and concepts from the book **Build A Large Language Model (From Scratch)** by Sebastian Raschka. 

If you use this software or derivations of it in your research or project, please cite the original work using the following formats:

### APA Style
Raschka, S. (2024). *Build a Large Language Model (from scratch)*. Manning Publications. https://www.manning.com/books/build-a-large-language-model-from-scratch

### BibTeX
```bibtex
@book{raschka2024build,
  author    = {Raschka, Sebastian},
  title     = {Build A Large Language Model (From Scratch)},
  publisher = {Manning Publications},
  year      = {2024},
  month     = {September},
  isbn      = {978-1633437166},
  url       = {[https://www.manning.com/books/build-a-large-language-model-from-scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)}
}
