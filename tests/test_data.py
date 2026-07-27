import pandas as pd
import tiktoken

from gpt2_classifier.data import (
    TextClassificationDataset,
    create_balanced_dataset,
    prepare_dataset,
    random_split,
)
from gpt2_classifier.datasets_registry import DatasetSpec


def test_create_balanced_dataset_equalizes_class_counts(synthetic_labeled_dataframe):
    df = synthetic_labeled_dataframe.copy()
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

    balanced = create_balanced_dataset(df)

    counts = balanced["Label"].value_counts()
    assert counts[0] == counts[1]
    assert counts[0] == 2  # minority class size in the fixture


def test_random_split_proportions():
    df = pd.DataFrame({"Text": [f"t{i}" for i in range(100)], "Label": [0] * 50 + [1] * 50})

    train_df, val_df, test_df = random_split(df, train_frac=0.7, validation_frac=0.1, test_frac=0.2)

    assert len(train_df) == 70
    assert len(val_df) == 10
    assert len(test_df) == 20
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_text_classification_dataset_pads_and_truncates(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"Text": ["short", "a much longer piece of text here"], "Label": [0, 1]}
    ).to_csv(csv_path, index=False)

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextClassificationDataset(csv_path, tokenizer=tokenizer, max_length=5, pad_token_id=999)

    assert len(dataset) == 2
    token_ids, label = dataset[0]
    assert token_ids.shape == (5,)
    assert label.item() == 0


def test_prepare_dataset_normalizes_headerless_source(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sms-like"
    raw_dir.mkdir()
    raw_file = raw_dir / "raw.tsv"
    raw_file.write_text(
        "ham\thello there\n"
        "spam\twin a prize now\n"
        "ham\thow are you\n"
        "spam\tclaim your reward\n"
        "ham\tGood night\n"
        "spam\tAmazing trophy for you\n"
        "ham\thello there\n"
        "spam\twin a prize now\n"
        "ham\thow are you\n"
        "spam\tclaim your reward\n"
        "ham\tGood night\n"
        "spam\tAmazing trophy for you\n"
        "ham\thello there\n"
        "spam\twin a prize now\n"
        "ham\thow are you\n"
        "spam\tclaim your reward\n"
        "ham\tGood night\n"
        "spam\tAmazing trophy for you\n"
    )

    spec = DatasetSpec(
        name="sms-like",
        url="https://example.invalid/does-not-matter.zip",
        raw_filename="raw.tsv",
        text_column="Text",
        label_column="Label",
        label_map={"ham": 0, "spam": 1},
        archive_format="none",
        separator="\t",
        has_header=False,
        column_names=["Label", "Text"],
    )

    # Pre-place the raw file so download_and_extract's "already exists" branch
    # is taken and no network access is attempted.
    monkeypatch.setattr(
        "gpt2_classifier.data.download_and_extract",
        lambda spec, data_dir=tmp_path: raw_file,
    )

    output_dir = prepare_dataset(spec, data_dir=tmp_path, dataset_split=[0.6, 0.2, 0.2])

    train_df = pd.read_csv(output_dir / "train.csv")
    val_df = pd.read_csv(output_dir / "validation.csv")
    test_df = pd.read_csv(output_dir / "test.csv")
    combined = pd.concat([train_df, val_df, test_df])

    assert set(combined.columns) == {"Text", "Label"}
    assert set(combined["Label"].unique()) <= {0, 1}


def test_prepare_dataset_normalizes_headered_source(tmp_path, monkeypatch):
    raw_file = tmp_path / "email.csv"
    pd.DataFrame(
        {
            "Message ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Message": ["hi there", "win money now", "lunch tomorrow?", "claim prize today",
                        "hi there", "win money now", "lunch tomorrow?", "claim prize today",
                        "hi there", "win money now", "lunch tomorrow?", "claim prize today"],
            "Spam/Ham": ["ham", "spam", "ham", "spam",
                         "ham", "spam", "ham", "spam",
                         "ham", "spam", "ham", "spam"],
        }
    ).to_csv(raw_file, index=False)

    spec = DatasetSpec(
        name="email-like",
        url="https://example.invalid/does-not-matter.csv",
        raw_filename="email.csv",
        text_column="Message",
        label_column="Spam/Ham",
        label_map={"ham": 0, "spam": 1},
        archive_format="none",
        separator=",",
    )

    monkeypatch.setattr(
        "gpt2_classifier.data.download_and_extract",
        lambda spec, data_dir=tmp_path: raw_file,
    )

    output_dir = prepare_dataset(spec, data_dir=tmp_path, dataset_split=[0.6, 0.2, 0.2])
    train_df = pd.read_csv(output_dir / "train.csv")
    assert set(train_df.columns) >= {"Text", "Label"}
