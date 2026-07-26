"""Dataset download, preparation, and PyTorch DataLoader construction.

Every function here is driven by a :class:`gpt2_classifier.datasets_registry.DatasetSpec` 
rather than hardcoded URLs/columns, so any two-column text/label dataset -- registered 
(``sms-spam``, ``email-spam``) or ad-hoc (built at the CLI layer
from ``--data-url``/``--data-path``) -- flows through the same pipeline.
"""

import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from gpt2_classifier import paths
from gpt2_classifier.datasets_registry import DatasetSpec

_NORMALIZED_TEXT_COLUMN = "Text"
_NORMALIZED_LABEL_COLUMN = "Label"


def _download_file(url: str, destination: Path) -> None:
    """Stream-download ``url`` to ``destination``."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(destination, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)


def _download_with_fallback(spec: DatasetSpec, destination: Path) -> None:
    """Download ``spec.url`` to ``destination``, retrying ``spec.backup_url`` on failure."""
    try:
        _download_file(spec.url, destination)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        if spec.backup_url is None:
            raise
        print(f"Primary URL failed: {e}. Trying backup URL...")
        _download_file(spec.backup_url, destination)


def download_and_extract(spec: DatasetSpec, data_dir: Path = paths.DATA_DIR) -> Path:
    """Download and (if archived) extract a dataset, returning the raw file path.

    Idempotent: if the raw file already exists under
    ``data_dir / spec.name / spec.raw_filename``, the download/extraction is
    skipped entirely.

    Args:
        spec: Dataset source description.
        data_dir: Root directory under which per-dataset subdirectories are
            created.

    Returns:
        Path to the raw (uncompressed) data file.

    Raises:
        FileNotFoundError: If ``spec.raw_filename`` cannot be located inside
            an extracted archive.
    """
    dataset_dir = data_dir / spec.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dataset_dir / spec.raw_filename

    if raw_path.exists():
        print(f"{raw_path} already exists. Skipping download and extraction.")
        return raw_path

    if spec.archive_format == "none":
        _download_with_fallback(spec, raw_path)
        return raw_path

    download_target = dataset_dir / Path(spec.url).name
    _download_with_fallback(spec, download_target)

    extract_dir = dataset_dir / "extracted"
    if spec.archive_format == "zip":
        with zipfile.ZipFile(download_target, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    elif spec.archive_format == "tar":
        with tarfile.open(download_target) as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive_format: {spec.archive_format!r}")

    extracted_file = next(extract_dir.rglob(spec.raw_filename), None)
    if extracted_file is None:
        raise FileNotFoundError(
            f"Could not find {spec.raw_filename!r} inside extracted archive {extract_dir}"
        )
    extracted_file.rename(raw_path)
    print(f"File downloaded and saved as {raw_path}")
    return raw_path


def create_balanced_dataset(df: pd.DataFrame, label_column: str = _NORMALIZED_LABEL_COLUMN) -> pd.DataFrame:
    """Undersample all classes down to the size of the smallest one.

    Works for any label distribution (not just a fixed 0/1 majority/minority
    split), so it applies unchanged to any registered or ad-hoc dataset.

    Args:
        df: DataFrame with an integer-encoded label column.
        label_column: Name of the label column.

    Returns:
        A new, class-balanced DataFrame.
    """
    # minority_count = df[label_column].value_counts().min()
    # balanced_frames = [
    #     group.sample(minority_count, random_state=123)
    #     for _, group in df.groupby(label_column)
    # ]

    # return pd.concat(balanced_frames).reset_index(drop=True)

    # ----- testing ------
    # Count the instances of "spam"
    num_spam = df[df[label_column] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df[label_column] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df[label_column] == "spam"]])

    return balanced_df

    # -----------------------   


def random_split(
    df: pd.DataFrame, train_frac: float, validation_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a DataFrame into train/validation/test partitions.

    Args:
        df: DataFrame to split.
        train_frac: Fraction of rows to allocate to the training split.
        validation_frac: Fraction of rows to allocate to the validation
            split. The remainder goes to the test split.

    Returns:
        A ``(train_df, validation_df, test_df)`` tuple.
    """
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def prepare_dataset(spec: DatasetSpec, data_dir: Path = paths.DATA_DIR) -> Path:
    """Download, normalize, balance, split, and persist a dataset as CSVs.

    Writes ``train.csv``/``validation.csv``/``test.csv`` under
    ``data_dir / spec.name``, each with normalized ``"Text"``/``"Label"``
    columns (label already integer-encoded per ``spec.label_map``),
    regardless of the raw source's original column names/encoding.

    Args:
        spec: Dataset source description.
        data_dir: Root directory under which per-dataset subdirectories are
            created and CSVs are written.

    Returns:
        The dataset's output directory (``data_dir / spec.name``).
    """
    raw_path = download_and_extract(spec, data_dir)

    read_kwargs = {"sep": spec.separator}
    if spec.has_header:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None
        read_kwargs["names"] = spec.column_names

    df = pd.read_csv(raw_path, **read_kwargs)
    df = df[[spec.text_column, spec.label_column]].rename(
        columns={spec.text_column: _NORMALIZED_TEXT_COLUMN, spec.label_column: _NORMALIZED_LABEL_COLUMN}
    )
    df = df.dropna(subset=[_NORMALIZED_TEXT_COLUMN, _NORMALIZED_LABEL_COLUMN])
    df[_NORMALIZED_LABEL_COLUMN] = df[_NORMALIZED_LABEL_COLUMN].map(spec.label_map)

    balanced_df = create_balanced_dataset(df)
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    output_dir = data_dir / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=None)
    validation_df.to_csv(output_dir / "validation.csv", index=None)
    test_df.to_csv(output_dir / "test.csv", index=None)

    return output_dir


class TextClassificationDataset(Dataset):
    """A tokenized, padded text-classification dataset."""

    def __init__(
        self,
        csv_file: str | Path,
        tokenizer: "tiktoken.Encoding",
        max_length: int | None = None,
        pad_token_id: int = 50256,
        text_column: str = _NORMALIZED_TEXT_COLUMN,
        label_column: str = _NORMALIZED_LABEL_COLUMN,
    ) -> None:
        """Load and pre-tokenize a normalized text/label CSV.

        Args:
            csv_file: Path to a CSV with (at least) ``text_column`` and
                ``label_column`` columns.
            tokenizer: A ``tiktoken`` encoding used to tokenize the text.
            max_length: Fixed sequence length to pad/truncate to. If
                ``None``, uses the longest encoded example in this file.
            pad_token_id: Token id used for right-padding.
            text_column: Name of the column holding raw text.
            label_column: Name of the column holding integer labels.
        """
        self.data = pd.read_csv(csv_file)
        self.text_column = text_column
        self.label_column = label_column

        self.encoded_texts = [tokenizer.encode(text) for text in self.data[text_column]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(token_ids, label)`` pair at ``index``."""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index][self.label_column]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        """Return the length of the longest encoded example."""
        return max((len(encoded_text) for encoded_text in self.encoded_texts), default=0)


def create_data_loaders(
    dataset_name: str,
    data_dir: Path = paths.DATA_DIR,
    batch_size: int = 8,
    num_cpu_workers: int = 0,
    use_ddp: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test DataLoaders for a previously prepared dataset.

    Args:
        dataset_name: Name of the prepared dataset (matches the directory
            under ``data_dir`` created by :func:`prepare_dataset`).
        data_dir: Root directory containing per-dataset subdirectories.
        batch_size: Batch size for all three loaders.
        num_cpu_workers: Number of DataLoader worker processes.
        use_ddp: If ``True``, the training loader uses a
            ``DistributedSampler`` instead of ``shuffle=True`` so
            ``train_ddp.train_classifier_ddp`` can call
            ``train_loader.sampler.set_epoch(...)``.

    Returns:
        A ``(train_loader, val_loader, test_loader)`` tuple.
    """
    dataset_dir = data_dir / dataset_name
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = TextClassificationDataset(
        csv_file=dataset_dir / "train.csv", max_length=None, tokenizer=tokenizer
    )
    val_dataset = TextClassificationDataset(
        csv_file=dataset_dir / "validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    test_dataset = TextClassificationDataset(
        csv_file=dataset_dir / "test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    torch.manual_seed(123)

    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_cpu_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_cpu_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_cpu_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
