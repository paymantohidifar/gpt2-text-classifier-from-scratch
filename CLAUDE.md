# gpt2-classifier

A from-scratch GPT-2 fine-tuned as a text classifier (spam/ham).
Not limited to SMS — the dataset pipeline is
source-agnostic (see `datasets_registry.py`).

## Layout (`src/gpt2_classifier/`)

- `paths.py` — `PROJECT_ROOT`/`DATA_DIR`/`MODELS_DIR`. Always import these instead of hardcoding paths.
- `config.py` — GPT-2 size presets. Use `get_model_config(name)`, never mutate `BASE_CONFIG` directly.
- `datasets_registry.py` — `DatasetSpec` + `DATASET_REGISTRY` (`sms-spam`, `email-spam`). Add a new dataset source here, not in `data.py`.
- `model.py` — the GPT-2 architecture (attention/layernorm/GELU/transformer block/`GPTModel`).
- `weights.py` — downloads + maps HuggingFace's flat safetensors GPT-2 checkpoint into `GPTModel` (HF uses Conv1D-style `[in, out]` weights, hence the `.T` transposes — don't "simplify" these away).
- `data.py` — spec-driven download/prepare/`TextClassificationDataset`/`create_data_loaders`.
- `evaluate.py` — loss/accuracy helpers (last-token logits).
- `train.py` — single-device fine-tuning (the default path): freezes the backbone, swaps in a classification head, unfreezes the last transformer block + final norm, trains, saves a checkpoint (`model_state_dict` + `model_config` + `num_classes` + `label_names`).
- `train_ddp.py` — optional multi-GPU path (`--ddp`), requires `torchrun`/CUDA. Not unit-tested; smoke-checked by import only.
- `logging_utils.py` — `RunLogger`, a wandb wrapper that must degrade to a no-op (never raise) if `wandb.init` fails.
- `inference.py` — `load_classifier()` + `classify_text()`, used by the `predict` CLI subcommand.
- `cli.py` / `__main__.py` — `python -m gpt2_classifier {download-weights,train,predict,generate}`.

## Running things

```
pixi run pytest                          # full test suite
python -m gpt2_classifier download-weights --model-name "gpt2-small (124M)"
python -m gpt2_classifier train --dataset sms-spam --num-epochs 5 [--use-wandb] [--ddp]
python -m gpt2_classifier train --data-url <url> --text-column ... --label-column ... --label-map "ham=0,spam=1"
python -m gpt2_classifier predict --checkpoint models/sms-spam_classifier.pt --text "..."
python -m gpt2_classifier generate --prompt "..."   # raw LM generation, debug only
```

## Conventions to preserve

- No hardcoded cwd-relative paths — always go through `paths.py`.
- No CUDA-specific dependency pins; device selection always via `torch.cuda.is_available()` so the same code runs on this CPU-only dev machine and a CUDA machine (e.g. Colab) unchanged.
- Adding a dataset source = one `DatasetSpec` entry in `datasets_registry.py`, not new pipeline branches in `data.py`.
- New functions get type hints and a Google-style docstring.
- `train_ddp.py` is secondary/opt-in — don't make it the default path or a hard dependency for single-device use.

## Testing

`tests/` has one file per module plus `conftest.py` (shared fixtures: `tiny_gpt_config`, `tiny_gpt_config_real_vocab` for tests that tokenize with the real `tiktoken` "gpt2" encoding, `tiny_gpt_model`, `synthetic_labeled_dataframe`). Tests use tiny configs and mock network calls — no real downloads or full training runs in the suite.

See `docs/refactor-summary.md` for the full history of what was fixed and why.
