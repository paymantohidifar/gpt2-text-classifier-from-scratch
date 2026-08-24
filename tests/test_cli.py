import argparse

import pytest

from gpt2_classifier.cli import (
    _build_adhoc_spec,
    _infer_archive_format,
    _parse_label_map,
    _resolve_dataset_spec,
    build_parser,
)


def test_parse_label_map_parses_pairs():
    assert _parse_label_map("ham=0,spam=1") == {"ham": 0, "spam": 1}


def test_parse_label_map_rejects_malformed_entry():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_label_map("ham0,spam=1")


@pytest.mark.parametrize(
    "source,expected",
    [
        ("https://example.com/data.zip", "zip"),
        ("https://example.com/data.tar.gz", "tar"),
        ("https://example.com/data.csv", "none"),
    ],
)
def test_infer_archive_format(source, expected):
    assert _infer_archive_format(source) == expected


def test_train_subcommand_parses_expected_args():
    parser = build_parser()
    args = parser.parse_args(
        ["train", "--model-name", "gpt2-small (124M)", "--dataset", "email-spam", "--num-epochs", "3"]
    )
    assert args.command == "train"
    assert args.model_name == "gpt2-small (124M)"
    assert args.dataset == "email-spam"
    assert args.num_epochs == 3
    assert args.func.__name__ == "_run_train"


def test_train_subcommand_lora_flags_default_off():
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.lora_enabled is False
    assert args.lora_rank == 16
    assert args.lora_alpha == 16


def test_train_subcommand_parses_lora_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["train", "--lora-enabled", "--lora-rank", "8", "--lora-alpha", "32"]
    )
    assert args.lora_enabled is True
    assert args.lora_rank == 8
    assert args.lora_alpha == 32


def test_predict_subcommand_requires_checkpoint_and_text():
    parser = build_parser()
    args = parser.parse_args(["predict", "--checkpoint", "models/x.pt", "--text", "hello"])
    assert args.checkpoint == "models/x.pt"
    assert args.text == "hello"
    assert args.func.__name__ == "_run_predict"


def test_generate_subcommand_parses_prompt():
    parser = build_parser()
    args = parser.parse_args(["generate", "--prompt", "once upon a time"])
    assert args.prompt == "once upon a time"
    assert args.func.__name__ == "_run_generate"


def test_download_weights_subcommand_defaults():
    parser = build_parser()
    args = parser.parse_args(["download-weights"])
    assert args.model_name == "gpt2-small (124M)"
    assert args.func.__name__ == "_run_download_weights"


def test_resolve_dataset_spec_uses_registry_by_default():
    parser = build_parser()
    args = parser.parse_args(["train", "--dataset", "sms-spam"])
    spec = _resolve_dataset_spec(args)
    assert spec.name == "sms-spam"


def test_build_adhoc_spec_requires_text_and_label_columns():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--data-url",
            "https://example.com/custom.csv",
            "--label-map",
            "ham=0,spam=1",
        ]
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _build_adhoc_spec(args)


def test_build_adhoc_spec_requires_label_map():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--data-url",
            "https://example.com/custom.csv",
            "--text-column",
            "body",
            "--label-column",
            "is_spam",
        ]
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _build_adhoc_spec(args)


def test_build_adhoc_spec_builds_valid_spec_from_plain_csv_url():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--data-url",
            "https://example.com/custom.csv",
            "--text-column",
            "body",
            "--label-column",
            "is_spam",
            "--label-map",
            "not_spam=0,spam=1",
        ]
    )
    spec = _build_adhoc_spec(args)
    assert spec.text_column == "body"
    assert spec.label_column == "is_spam"
    assert spec.label_map == {"not_spam": 0, "spam": 1}
    assert spec.archive_format == "none"
    assert spec.raw_filename == "custom.csv"


def test_build_adhoc_spec_requires_raw_filename_for_archives():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--data-url",
            "https://example.com/custom.zip",
            "--text-column",
            "body",
            "--label-column",
            "is_spam",
            "--label-map",
            "ham=0,spam=1",
        ]
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _build_adhoc_spec(args)


def test_resolve_dataset_spec_rejects_both_data_url_and_data_path():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--data-url",
            "https://example.com/a.csv",
            "--data-path",
            "b.csv",
            "--text-column",
            "body",
            "--label-column",
            "is_spam",
            "--label-map",
            "ham=0,spam=1",
        ]
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _resolve_dataset_spec(args)
