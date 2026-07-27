"""Registry of known text-classification dataset sources.

A :class:`DatasetSpec` fully describes how to fetch and normalize one
dataset: where to download it from, how it is archived, which raw columns
hold the text/label, and how label strings map to integer class ids. This
lets :mod:`gpt2_classifier.data` stay source-agnostic -- adding a new dataset
is a matter of adding one more entry to :data:`DATASET_REGISTRY`, not writing
new pipeline code.

Ad-hoc, unregistered datasets (e.g. a user's own CSV, or an arbitrary URL) are
supported too: :mod:`gpt2_classifier.cli` builds a one-off ``DatasetSpec`` at
runtime from CLI flags (``--data-url``/``--data-path`` plus
``--text-column``/``--label-column``/``--label-map``/etc.) instead of looking
one up here.
"""

from dataclasses import dataclass
from typing import Literal

ArchiveFormat = Literal["zip", "tar", "none"]


@dataclass(frozen=True)
class DatasetSpec:
    """Describes one downloadable text-classification dataset source.

    Attributes:
        name: Short identifier, used to select the spec via ``--dataset`` and
            to namespace prepared CSVs under ``data/<name>/``.
        url: Primary download URL.
        raw_filename: Name of the file to read once downloaded/extracted
            (relative to the extraction directory for archives, or the
            downloaded file itself when ``archive_format == "none"``).
        text_column: Column name holding the raw text in the source file.
        label_column: Column name holding the raw label in the source file.
        label_map: Mapping from raw label strings to integer class ids.
        archive_format: How the downloaded file is packaged.
        separator: Field separator used when reading the raw file with
            pandas (e.g. ``","`` for CSV, ``"\\t"`` for TSV).
        has_header: Whether the raw file has a header row. When ``False``,
            ``column_names`` must be given (in on-disk column order) since
            pandas has nothing to infer column names from.
        column_names: Column names to assign when ``has_header`` is
            ``False``, in the order they appear in the file.
        backup_url: Optional fallback URL used if ``url`` fails to download.
    """

    name: str
    url: str
    raw_filename: str
    text_column: str
    label_column: str
    label_map: dict[str, int]
    archive_format: ArchiveFormat = "none"
    separator: str = ","
    has_header: bool = True
    column_names: list[str] | None = None
    backup_url: str | None = None

    def __post_init__(self) -> None:
        if not self.has_header and not self.column_names:
            raise ValueError(
                f"DatasetSpec {self.name!r} has has_header=False but no "
                "column_names given."
            )


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "sms-spam": DatasetSpec(
        name="sms-spam",
        url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        backup_url=(
            "https://f001.backblazeb2.com/file/LLMs-from-scratch/"
            "sms%2Bspam%2Bcollection.zip"
        ),
        raw_filename="SMSSpamCollection",
        text_column="Text",
        label_column="Label",
        label_map={"ham": 0, "spam": 1},
        archive_format="zip",
        separator="\t",
        has_header=False,
        column_names=["Label", "Text"],
    ),
    "email-spam": DatasetSpec(
        name="email-spam",
        url=(
            "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/"
            "master/enron_spam_data.zip"
        ),
        raw_filename="enron_spam_data.csv",
        text_column="Message",
        label_column="Spam/Ham",
        label_map={"ham": 0, "spam": 1},
        archive_format="zip",
        separator=",",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Look up a registered dataset spec by name.

    Args:
        name: Key into :data:`DATASET_REGISTRY`, e.g. ``"sms-spam"``.

    Returns:
        The matching :class:`DatasetSpec`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Available options: "
            f"{sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]
