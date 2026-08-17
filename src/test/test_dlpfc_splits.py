import json

import pandas as pd
import pytest

from src.splits.dlpfc import (
    DEFAULT_DONOR_SECTIONS,
    GLOBAL_LABEL_ORDER,
    dataset_fingerprint,
    generate_leave_one_donor_out,
    generate_leave_one_section_out,
    generate_pooled_random_spot,
    generate_within_donor_random_spot,
    generate_within_donor_section_holdout,
    validate_split_manifest,
    write_split_artifact,
)


def synthetic_dataset(spots_per_label: int = 3) -> pd.DataFrame:
    rows = []
    global_row = 0
    for donor, sections in DEFAULT_DONOR_SECTIONS.items():
        for section in sections:
            section_row = 0
            for label in GLOBAL_LABEL_ORDER:
                for replicate in range(spots_per_label):
                    rows.append({
                        "global_row": global_row,
                        "section_row": section_row,
                        "section_id": section,
                        "donor_id": donor,
                        "barcode": f"BC_{label}_{replicate}",
                        "ground_truth": label,
                        "ground_truth_valid": True,
                    })
                    global_row += 1
                    section_row += 1
    return pd.DataFrame(rows)


def role_sections(split, fold_id: str, role: str) -> set[str]:
    table = split.table
    return set(table.loc[(table["fold_id"] == fold_id) & (table["role"] == role), "section_id"])


def test_lodo_uses_exact_test_donors_and_validation_sections() -> None:
    dataset = synthetic_dataset()
    split = generate_leave_one_donor_out(dataset)
    assert len(split.table) == 3 * len(dataset)
    assert role_sections(split, "lodo_d1", "test") == set(DEFAULT_DONOR_SECTIONS[1])
    assert role_sections(split, "lodo_d1", "validation") == {"151669", "151673"}
    assert role_sections(split, "lodo_d2", "validation") == {"151508", "151674"}
    assert role_sections(split, "lodo_d3", "validation") == {"151509", "151671"}


def test_loso_uses_same_ordinal_other_donor_validation() -> None:
    dataset = synthetic_dataset()
    split = generate_leave_one_section_out(dataset)
    assert split.table["fold_id"].nunique() == 12
    assert role_sections(split, "loso_151508", "test") == {"151508"}
    assert role_sections(split, "loso_151508", "validation") == {"151670", "151674"}


def test_within_donor_section_scope_and_cyclic_validation() -> None:
    dataset = synthetic_dataset()
    split = generate_within_donor_section_holdout(dataset)
    fold = split.table.loc[split.table["fold_id"] == "within_d1_151510"]
    assert set(fold["section_id"]) == set(DEFAULT_DONOR_SECTIONS[1])
    assert role_sections(split, "within_d1_151510", "test") == {"151510"}
    assert role_sections(split, "within_d1_151510", "validation") == {"151507"}


def test_random_splits_are_order_independent_stratified_and_seeded() -> None:
    dataset = synthetic_dataset(spots_per_label=20)
    first = generate_pooled_random_spot(dataset, seeds=(7,))
    shuffled = generate_pooled_random_spot(dataset.sample(frac=1, random_state=9), seeds=(7,))
    columns = ["section_id", "barcode", "role", "assignment_sha256"]
    left = first.table.sort_values(columns[:2])[columns].reset_index(drop=True)
    right = shuffled.table.sort_values(columns[:2])[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    counts = first.table.groupby(["section_id", "ground_truth", "role"]).size()
    assert set(counts.unique()) == {3, 14}
    other_seed = generate_pooled_random_spot(dataset, seeds=(8,))
    assert first.table["role"].tolist() != other_seed.table["role"].tolist()


def test_within_donor_random_has_complete_donor_scoped_roles() -> None:
    dataset = synthetic_dataset(spots_per_label=20)
    split = generate_within_donor_random_spot(dataset, seeds=(0, 1))
    assert split.table["fold_id"].nunique() == 6
    fold = split.table.loc[split.table["fold_id"] == "within_d2_seed0"]
    assert set(fold["donor_id"]) == {2}
    assert set(fold["role"]) == {"train", "validation", "test"}


def test_validator_rejects_duplicate_identity_and_fingerprint_mismatch() -> None:
    dataset = synthetic_dataset()
    split = generate_leave_one_donor_out(dataset)
    duplicate = pd.concat([split.table, split.table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="overlapping or duplicate"):
        validate_split_manifest(dataset, duplicate, split.metadata)
    changed = dataset.copy()
    changed.loc[0, "ground_truth"] = "Layer_2"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_split_manifest(changed, split.table, split.metadata)


def test_artifacts_are_parquet_json_and_never_overwritten(tmp_path) -> None:
    dataset = synthetic_dataset(spots_per_label=20)
    split = generate_pooled_random_spot(dataset)
    parquet_path, metadata_path = write_split_artifact(split, tmp_path, "immutable_v1", "audit.parquet")
    restored = pd.read_parquet(parquet_path)
    assert len(restored) == len(dataset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_fingerprint_sha256"] == dataset_fingerprint(dataset)
    assert metadata["immutable"] is True
    assert len(metadata["class_support"]) == 3
    with pytest.raises(FileExistsError):
        write_split_artifact(split, tmp_path, "immutable_v1", "audit.parquet")
