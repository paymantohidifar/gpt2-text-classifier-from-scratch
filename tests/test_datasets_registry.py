import pytest

from gpt2_classifier.datasets_registry import DATASET_REGISTRY, DatasetSpec, get_dataset_spec


def test_registry_contains_sms_and_email_specs():
    assert "sms-spam" in DATASET_REGISTRY
    assert "email-spam" in DATASET_REGISTRY


@pytest.mark.parametrize("name", list(DATASET_REGISTRY))
def test_registered_specs_have_required_fields(name):
    spec = DATASET_REGISTRY[name]
    assert isinstance(spec, DatasetSpec)
    assert spec.name == name
    assert spec.url.startswith("http")
    assert spec.raw_filename
    assert spec.text_column
    assert spec.label_column
    assert spec.label_map
    assert spec.archive_format in ("zip", "tar", "none")


def test_get_dataset_spec_returns_matching_spec():
    spec = get_dataset_spec("sms-spam")
    assert spec is DATASET_REGISTRY["sms-spam"]


def test_get_dataset_spec_unknown_name_raises():
    with pytest.raises(KeyError):
        get_dataset_spec("not-a-real-dataset")
