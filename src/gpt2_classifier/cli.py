"""Command-line entry point: ``python -m gpt2_classifier <subcommand> ...``.

Subcommands:
    download-weights: Fetch a pretrained GPT-2 checkpoint from HuggingFace.
    train: Fine-tune a GPT-2 classifier on a registered or ad-hoc dataset.
    predict: Classify a piece of text with a fine-tuned checkpoint.
    generate: Raw GPT-2 text generation (debug helper, not classification).
"""

import argparse
import shutil
from pathlib import Path

from gpt2_classifier import paths
from gpt2_classifier.config import URL_DIR, get_model_config
from gpt2_classifier.data import create_data_loaders, prepare_dataset
from gpt2_classifier.datasets_registry import ArchiveFormat, DATASET_REGISTRY, DatasetSpec, get_dataset_spec
from gpt2_classifier.inference import run_prediction
from gpt2_classifier.logging_utils import RunLogger
from gpt2_classifier.model import GPTModel
from gpt2_classifier.train import finetune_model
from gpt2_classifier.utils import generate_response, plot_results
from gpt2_classifier.weights import download_and_load_gpt2, load_weights_into_gpt


def _parse_label_map(raw: str) -> dict[str, int]:
    """Parse a ``"ham=0,spam=1"``-style string into a label map dict."""
    label_map: dict[str, int] = {}
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        if not key or not value:
            raise argparse.ArgumentTypeError(
                f"Invalid --label-map entry {pair!r}; expected format like 'ham=0,spam=1'"
            )
        label_map[key.strip()] = int(value.strip())
    return label_map


def _infer_archive_format(source: str) -> ArchiveFormat:
    """Guess an archive format from a URL/path's extension."""
    lowered = source.lower()
    if lowered.endswith(".zip"):
        return "zip"
    if lowered.endswith((".tar", ".tar.gz", ".tgz")):
        return "tar"
    return "none"


def _build_adhoc_spec(args: argparse.Namespace) -> DatasetSpec:
    """Build a one-off DatasetSpec from --data-url/--data-path CLI flags."""
    if not args.text_column or not args.label_column:
        raise argparse.ArgumentTypeError(
            "--text-column and --label-column are required when using --data-url/--data-path"
        )
    if not args.label_map:
        raise argparse.ArgumentTypeError(
            "--label-map is required when using --data-url/--data-path (e.g. 'ham=0,spam=1')"
        )

    source = args.data_url or args.data_path
    archive_format = args.archive_format or _infer_archive_format(source)

    if archive_format == "none":
        raw_filename = args.raw_filename or Path(source).name
    else:
        if not args.raw_filename:
            raise argparse.ArgumentTypeError(
                "--raw-filename is required when the source is a zip/tar archive "
                "(the filename to read once extracted)"
            )
        raw_filename = args.raw_filename

    name = args.dataset_name or Path(source).stem.replace(" ", "-").lower()

    return DatasetSpec(
        name=name,
        url=args.data_url or f"file://{Path(args.data_path).resolve()}",
        raw_filename=raw_filename,
        text_column=args.text_column,
        label_column=args.label_column,
        label_map=_parse_label_map(args.label_map),
        archive_format=archive_format,
        separator=args.separator,
        has_header=not args.no_header,
        column_names=args.column_names.split(",") if args.column_names else None,
    )


def _resolve_dataset_spec(args: argparse.Namespace) -> DatasetSpec:
    """Resolve the DatasetSpec to use for a `train` invocation.

    Prefers an ad-hoc spec built from ``--data-url``/``--data-path`` when
    given; otherwise looks up ``--dataset`` in the registry.
    """
    if args.data_url and args.data_path:
        raise argparse.ArgumentTypeError("--data-url and --data-path are mutually exclusive")

    if args.data_url or args.data_path:
        spec = _build_adhoc_spec(args)
        if args.data_path:
            # Local file: pre-place it where download_and_extract expects the
            # raw file so no network access is attempted.
            dataset_dir = paths.DATA_DIR / spec.name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            target = dataset_dir / spec.raw_filename
            if not target.exists():
                shutil.copy(args.data_path, target)
        return spec

    return get_dataset_spec(args.dataset)


def _run_download_weights(args: argparse.Namespace) -> None:
    model_config = get_model_config(args.model_name)
    model_url = f"https://huggingface.co/openai-community/{URL_DIR[args.model_name]}/resolve/main/model.safetensors"
    model_destination = paths.MODELS_DIR / f"{URL_DIR[args.model_name]}.safetensors"

    state_dict = download_and_load_gpt2(model_url, model_destination)
    model = GPTModel(model_config)
    load_weights_into_gpt(model, state_dict)
    print(f"Downloaded and verified weights for {args.model_name!r} at {model_destination}")


def _run_train(args: argparse.Namespace) -> None:
    spec = _resolve_dataset_spec(args)
    model_config = get_model_config(args.model_name)

    model_url = f"https://huggingface.co/openai-community/{URL_DIR[args.model_name]}/resolve/main/model.safetensors"
    model_destination = paths.MODELS_DIR / f"{URL_DIR[args.model_name]}.safetensors"
    state_dict = download_and_load_gpt2(model_url, model_destination)
    model = GPTModel(model_config)
    load_weights_into_gpt(model, state_dict)

    prepare_dataset(spec, data_dir=paths.DATA_DIR)
    train_loader, val_loader, _test_loader = create_data_loaders(
        dataset_name=spec.name,
        data_dir=paths.DATA_DIR,
        batch_size=args.batch_size,
        use_ddp=args.ddp,
    )

    label_names = {v: k for k, v in spec.label_map.items()}
    logger = RunLogger(enabled=args.use_wandb, config=vars(args))
    checkpoint_path = args.checkpoint_path or paths.MODELS_DIR / f"{spec.name}_classifier.pt"

    finetune_model(
        train_loader,
        val_loader,
        model,
        model_config,
        num_classes=len(spec.label_map),
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        use_ddp=args.ddp,
        logger=logger,
        plot_metrics=args.plot_metrics,
        checkpoint_path=checkpoint_path,
        label_names=label_names,
    )


def _run_predict(args: argparse.Namespace) -> None:
    label = run_prediction(args.checkpoint, args.text)
    print(label)


def _run_generate(args: argparse.Namespace) -> None:
    model_config = get_model_config(args.model_name)
    model_url = f"https://huggingface.co/openai-community/{URL_DIR[args.model_name]}/resolve/main/model.safetensors"
    model_destination = paths.MODELS_DIR / f"{URL_DIR[args.model_name]}.safetensors"
    state_dict = download_and_load_gpt2(model_url, model_destination)
    model = GPTModel(model_config)
    load_weights_into_gpt(model, state_dict)

    response = generate_response(args.prompt, model, max_new_tokens=args.max_new_tokens)
    print(response)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(description="Fine-tune and use a GPT-2 text classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-weights", help="Download a pretrained GPT-2 checkpoint")
    download_parser.add_argument("--model-name", default="gpt2-small (124M)", choices=list(URL_DIR))
    download_parser.set_defaults(func=_run_download_weights)

    train_parser = subparsers.add_parser("train", help="Fine-tune a classifier")
    train_parser.add_argument("--model-name", default="gpt2-small (124M)", choices=list(URL_DIR))
    train_parser.add_argument("--dataset", default="sms-spam", choices=list(DATASET_REGISTRY))
    train_parser.add_argument("--data-url", default=None, help="Ad-hoc remote dataset URL (zip/tar/plain CSV)")
    train_parser.add_argument("--data-path", default=None, help="Ad-hoc local dataset file")
    train_parser.add_argument("--dataset-name", default=None, help="Name to use for an ad-hoc dataset")
    train_parser.add_argument("--archive-format", default=None, choices=["zip", "tar", "none"])
    train_parser.add_argument("--raw-filename", default=None, help="Filename to read (required for zip/tar sources)")
    train_parser.add_argument("--separator", default=",")
    train_parser.add_argument("--no-header", action="store_true", help="Raw file has no header row")
    train_parser.add_argument("--column-names", default=None, help="Comma-separated column names when --no-header")
    train_parser.add_argument("--text-column", default=None)
    train_parser.add_argument("--label-column", default=None)
    train_parser.add_argument("--label-map", default=None, help="e.g. 'ham=0,spam=1'")
    train_parser.add_argument("--num-epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--eval-freq", type=int, default=50)
    train_parser.add_argument("--eval-iter", type=int, default=5)
    train_parser.add_argument("--use-wandb", action="store_true")
    train_parser.add_argument("--plot-metrics", action="store_true")
    train_parser.add_argument("--ddp", action="store_true", help="Use multi-GPU DDP training (requires torchrun)")
    train_parser.add_argument("--checkpoint-path", type=Path, default=None)
    train_parser.set_defaults(func=_run_train)

    predict_parser = subparsers.add_parser("predict", help="Classify a piece of text")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--text", required=True)
    predict_parser.set_defaults(func=_run_predict)

    generate_parser = subparsers.add_parser("generate", help="Raw GPT-2 text generation (debug helper)")
    generate_parser.add_argument("--model-name", default="gpt2-small (124M)", choices=list(URL_DIR))
    generate_parser.add_argument("--prompt", default="I'm a GPT2 model")
    generate_parser.add_argument("--max-new-tokens", type=int, default=20)
    generate_parser.set_defaults(func=_run_generate)

    return parser


def main() -> None:
    """Parse CLI args and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
