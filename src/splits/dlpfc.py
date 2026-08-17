"""Immutable, compound-identity split manifests for the twelve DLPFC sections."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

GLOBAL_LABEL_ORDER = (
    "Layer_1",
    "Layer_2",
    "Layer_3",
    "Layer_4",
    "Layer_5",
    "Layer_6",
    "WM",
)
ROLES = ("train", "validation", "test")
IDENTITY_COLUMNS = ("section_id", "barcode")
SOURCE_COLUMNS = (
    "global_row",
    "section_row",
    "section_id",
    "donor_id",
    "barcode",
    "ground_truth",
)
DEFAULT_DONOR_SECTIONS = {
    1: ("151507", "151508", "151509", "151510"),
    2: ("151669", "151670", "151671", "151672"),
    3: ("151673", "151674", "151675", "151676"),
}
DEFAULT_LODO_VALIDATION = {
    1: ("151669", "151673"),
    2: ("151508", "151674"),
    3: ("151509", "151671"),
}
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SplitManifest:
    """A validated split table and its serializable provenance metadata."""

    table: pd.DataFrame
    metadata: dict[str, Any]


def load_dataset_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate the audited spot-level dataset manifest."""
    path = Path(path)
    dataset = pd.read_parquet(path)
    required = set(SOURCE_COLUMNS) | {"ground_truth_valid"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {missing}")
    if dataset[list(SOURCE_COLUMNS)].isna().any().any():
        raise ValueError("Dataset manifest contains null required values")
    if not dataset["ground_truth_valid"].astype(bool).all():
        raise ValueError("Dataset manifest contains invalid ground-truth rows")
    if dataset.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Dataset manifest has duplicate (section_id, barcode) identities")
    observed = set(dataset["ground_truth"].astype(str))
    unexpected = sorted(observed - set(GLOBAL_LABEL_ORDER))
    if unexpected:
        raise ValueError(f"Dataset manifest contains labels outside the global order: {unexpected}")
    result = dataset.copy()
    result["section_id"] = result["section_id"].astype(str)
    result["barcode"] = result["barcode"].astype(str)
    result["ground_truth"] = result["ground_truth"].astype(str)
    result["donor_id"] = result["donor_id"].astype(int)
    return result


def dataset_fingerprint(dataset: pd.DataFrame) -> str:
    """Hash canonical biological identity and annotation fields, independent of row order."""
    columns = ("section_id", "barcode", "donor_id", "ground_truth")
    canonical = dataset.loc[:, columns].copy()
    canonical = canonical.sort_values(list(IDENTITY_COLUMNS), kind="stable")
    digest = hashlib.sha256()
    for row in canonical.itertuples(index=False, name=None):
        digest.update(json.dumps(row, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_assignment(family: str, fold_id: str, seed: int, section: str, barcode: str) -> str:
    value = f"dlpfc-split-v{SCHEMA_VERSION}\0{family}\0{fold_id}\0{seed}\0{section}\0{barcode}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_sections(donor_sections: Mapping[int, Sequence[str]]) -> dict[int, tuple[str, ...]]:
    result = {int(donor): tuple(str(section) for section in sections) for donor, sections in donor_sections.items()}
    all_sections = [section for sections in result.values() for section in sections]
    if not result or any(not sections for sections in result.values()):
        raise ValueError("Every donor must have at least one section")
    if len(all_sections) != len(set(all_sections)):
        raise ValueError("A section may belong to only one donor")
    return result


def _validate_dataset_layout(dataset: pd.DataFrame, donor_sections: Mapping[int, Sequence[str]]) -> None:
    expected = {
        (int(donor), str(section))
        for donor, sections in donor_sections.items()
        for section in sections
    }
    observed = set(zip(dataset["donor_id"].astype(int), dataset["section_id"].astype(str)))
    if observed != expected:
        raise ValueError(f"Dataset donor/section layout differs from configuration: expected={sorted(expected)}, observed={sorted(observed)}")


def _base_rows(dataset: pd.DataFrame, family: str, fold_id: str, seed: int, eligible_sections: Iterable[str]) -> pd.DataFrame:
    eligible = {str(section) for section in eligible_sections}
    rows = dataset.loc[dataset["section_id"].isin(eligible), list(SOURCE_COLUMNS)].copy()
    rows.insert(0, "family", family)
    rows.insert(1, "fold_id", fold_id)
    rows.insert(2, "seed", int(seed))
    rows["assignment_sha256"] = [
        _sha256_assignment(family, fold_id, seed, section, barcode)
        for section, barcode in rows.loc[:, IDENTITY_COLUMNS].itertuples(index=False, name=None)
    ]
    return rows


def _apportioned_counts(size: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    raw = [size * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _assign_random_roles(rows: pd.DataFrame, ratios: Sequence[float], strata: Sequence[str]) -> None:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-12:
        raise ValueError("ratios must be three non-negative values summing to one")
    rows["role"] = ""
    grouped = rows.groupby(list(strata), sort=True, observed=True).groups
    for indices in grouped.values():
        ordered = rows.loc[indices].sort_values("assignment_sha256", kind="stable").index
        train_count, validation_count, _ = _apportioned_counts(len(ordered), ratios)
        rows.loc[ordered[:train_count], "role"] = "train"
        rows.loc[ordered[train_count : train_count + validation_count], "role"] = "validation"
        rows.loc[ordered[train_count + validation_count :], "role"] = "test"


def _metadata(
    dataset: pd.DataFrame,
    family: str,
    folds: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "dataset_fingerprint_sha256": dataset_fingerprint(dataset),
        "dataset_row_count": int(len(dataset)),
        "identity_columns": list(IDENTITY_COLUMNS),
        "global_label_order": list(GLOBAL_LABEL_ORDER),
        "roles": list(ROLES),
        "assignment_algorithm": "SHA256 canonical identity ranking within configured strata",
        "config": dict(config),
        "folds": folds,
    }


def _finish(dataset: pd.DataFrame, tables: list[pd.DataFrame], metadata: dict[str, Any]) -> SplitManifest:
    table = pd.concat(tables, ignore_index=True)
    table = table.sort_values(["fold_id", "global_row"], kind="stable").reset_index(drop=True)
    metadata["manifest_row_count"] = int(len(table))
    validate_split_manifest(dataset, table, metadata)
    metadata["class_support"] = _class_support(table)
    return SplitManifest(table=table, metadata=metadata)


def generate_leave_one_donor_out(
    dataset: pd.DataFrame,
    donor_sections: Mapping[int, Sequence[str]] = DEFAULT_DONOR_SECTIONS,
    validation_sections: Mapping[int, Sequence[str]] = DEFAULT_LODO_VALIDATION,
) -> SplitManifest:
    """Generate leave-one-donor-out folds with explicit whole-section validation."""
    sections = _normalise_sections(donor_sections)
    _validate_dataset_layout(dataset, sections)
    tables, folds = [], []
    all_sections = [section for group in sections.values() for section in group]
    for donor in sorted(sections):
        fold_id = f"lodo_d{donor}"
        test = set(sections[donor])
        validation = {str(value) for value in validation_sections[donor]}
        if test & validation or not validation <= set(all_sections):
            raise ValueError(f"Invalid validation sections for {fold_id}")
        rows = _base_rows(dataset, "leave_one_donor_out", fold_id, 0, all_sections)
        rows["role"] = "train"
        rows.loc[rows["section_id"].isin(validation), "role"] = "validation"
        rows.loc[rows["section_id"].isin(test), "role"] = "test"
        tables.append(rows)
        folds.append(_fold_definition(fold_id, all_sections, test, validation))
    metadata = _metadata(dataset, "leave_one_donor_out", folds, {"validation_sections": {str(k): list(v) for k, v in validation_sections.items()}})
    return _finish(dataset, tables, metadata)


def generate_leave_one_section_out(
    dataset: pd.DataFrame,
    donor_sections: Mapping[int, Sequence[str]] = DEFAULT_DONOR_SECTIONS,
) -> SplitManifest:
    """Generate section holdouts, validating on matching ordinal sections of other donors."""
    sections = _normalise_sections(donor_sections)
    _validate_dataset_layout(dataset, sections)
    lengths = {len(value) for value in sections.values()}
    if len(lengths) != 1:
        raise ValueError("Same-ordinal validation requires equal section counts per donor")
    all_sections = [section for group in sections.values() for section in group]
    tables, folds = [], []
    for donor in sorted(sections):
        for ordinal, test_section in enumerate(sections[donor]):
            fold_id = f"loso_{test_section}"
            validation = {group[ordinal] for other, group in sections.items() if other != donor}
            rows = _base_rows(dataset, "leave_one_section_out", fold_id, 0, all_sections)
            rows["role"] = "train"
            rows.loc[rows["section_id"].isin(validation), "role"] = "validation"
            rows.loc[rows["section_id"] == test_section, "role"] = "test"
            tables.append(rows)
            folds.append(_fold_definition(fold_id, all_sections, {test_section}, validation))
    return _finish(dataset, tables, _metadata(dataset, "leave_one_section_out", folds, {"validation": "same ordinal sections of other donors"}))


def generate_within_donor_section_holdout(
    dataset: pd.DataFrame,
    donor_sections: Mapping[int, Sequence[str]] = DEFAULT_DONOR_SECTIONS,
) -> SplitManifest:
    """Generate donor-scoped section tests with the next cyclic section as validation."""
    sections = _normalise_sections(donor_sections)
    _validate_dataset_layout(dataset, sections)
    tables, folds = [], []
    for donor in sorted(sections):
        donor_group = sections[donor]
        if len(donor_group) < 3:
            raise ValueError("Within-donor section holdout requires at least three sections")
        for ordinal, test_section in enumerate(donor_group):
            fold_id = f"within_d{donor}_{test_section}"
            validation = {donor_group[(ordinal + 1) % len(donor_group)]}
            rows = _base_rows(dataset, "within_donor_section_holdout", fold_id, 0, donor_group)
            rows["role"] = "train"
            rows.loc[rows["section_id"].isin(validation), "role"] = "validation"
            rows.loc[rows["section_id"] == test_section, "role"] = "test"
            tables.append(rows)
            folds.append(_fold_definition(fold_id, donor_group, {test_section}, validation))
    return _finish(dataset, tables, _metadata(dataset, "within_donor_section_holdout", folds, {"scope": "one donor per fold", "validation": "next cyclic section"}))


def generate_pooled_random_spot(
    dataset: pd.DataFrame,
    seeds: Sequence[int] = (0,),
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> SplitManifest:
    """Generate pooled random-spot folds stratified by section and global label."""
    all_sections = sorted(dataset["section_id"].astype(str).unique())
    tables, folds = [], []
    for seed in seeds:
        fold_id = f"pooled_seed{int(seed)}"
        rows = _base_rows(dataset, "pooled_random_spot", fold_id, int(seed), all_sections)
        _assign_random_roles(rows, ratios, ("section_id", "ground_truth"))
        tables.append(rows)
        folds.append(_fold_definition(fold_id, all_sections, role_source="sha256_stratified_random"))
    config = {"seeds": [int(seed) for seed in seeds], "ratios": list(ratios), "strata": ["section_id", "ground_truth"]}
    return _finish(dataset, tables, _metadata(dataset, "pooled_random_spot", folds, config))


def generate_within_donor_random_spot(
    dataset: pd.DataFrame,
    seeds: Sequence[int] = (0,),
    ratios: Sequence[float] = DEFAULT_RATIOS,
    donor_sections: Mapping[int, Sequence[str]] = DEFAULT_DONOR_SECTIONS,
) -> SplitManifest:
    """Generate donor-scoped random-spot folds stratified by section and label."""
    sections = _normalise_sections(donor_sections)
    _validate_dataset_layout(dataset, sections)
    tables, folds = [], []
    for seed in seeds:
        for donor in sorted(sections):
            fold_id = f"within_d{donor}_seed{int(seed)}"
            rows = _base_rows(dataset, "within_donor_random_spot", fold_id, int(seed), sections[donor])
            _assign_random_roles(rows, ratios, ("section_id", "ground_truth"))
            tables.append(rows)
            folds.append(_fold_definition(fold_id, sections[donor], role_source="sha256_stratified_random"))
    config = {"seeds": [int(seed) for seed in seeds], "ratios": list(ratios), "strata": ["section_id", "ground_truth"], "scope": "one donor per fold"}
    return _finish(dataset, tables, _metadata(dataset, "within_donor_random_spot", folds, config))


def _fold_definition(
    fold_id: str,
    eligible: Iterable[str],
    test: Iterable[str] = (),
    validation: Iterable[str] = (),
    role_source: str = "section_membership",
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "eligible_sections": sorted(str(value) for value in eligible),
        "test_sections": sorted(str(value) for value in test),
        "validation_sections": sorted(str(value) for value in validation),
        "role_source": role_source,
    }


def _class_support(table: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for fold_id in table["fold_id"].drop_duplicates():
        fold = table.loc[table["fold_id"] == fold_id]
        for role in ROLES:
            counts = fold.loc[fold["role"] == role, "ground_truth"].value_counts()
            label_counts = {label: int(counts.get(label, 0)) for label in GLOBAL_LABEL_ORDER}
            records.append({
                "fold_id": fold_id,
                "role": role,
                "row_count": int(sum(label_counts.values())),
                "label_counts": label_counts,
                "missing_labels": [label for label, count in label_counts.items() if count == 0],
            })
    return records


def validate_split_manifest(dataset: pd.DataFrame, table: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    """Validate fingerprint, complete scoped assignment, identities, roles, and hashes."""
    required = set(SOURCE_COLUMNS) | {"family", "fold_id", "seed", "role", "assignment_sha256"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    if metadata.get("dataset_fingerprint_sha256") != dataset_fingerprint(dataset):
        raise ValueError("Dataset fingerprint mismatch")
    if set(table["family"]) != {metadata.get("family")}:
        raise ValueError("Split table family does not match metadata")
    if table["role"].isna().any() or not set(table["role"]) <= set(ROLES):
        raise ValueError("Every split row must have exactly one valid role")
    if table.duplicated(["fold_id", *IDENTITY_COLUMNS]).any():
        raise ValueError("A fold contains overlapping or duplicate compound identities")
    if int(metadata.get("manifest_row_count", -1)) != len(table):
        raise ValueError("Split manifest row count does not match metadata")
    configured_folds = [fold["fold_id"] for fold in metadata["folds"]]
    if len(configured_folds) != len(set(configured_folds)) or set(table["fold_id"]) != set(configured_folds):
        raise ValueError("Split table folds do not match unique metadata fold definitions")
    dataset_index = dataset.set_index(list(IDENTITY_COLUMNS), drop=False)
    for fold in metadata["folds"]:
        fold_id = fold["fold_id"]
        actual = table.loc[table["fold_id"] == fold_id]
        expected = dataset.loc[dataset["section_id"].isin(fold["eligible_sections"])]
        actual_ids = set(map(tuple, actual.loc[:, IDENTITY_COLUMNS].itertuples(index=False, name=None)))
        expected_ids = set(map(tuple, expected.loc[:, IDENTITY_COLUMNS].itertuples(index=False, name=None)))
        if actual_ids != expected_ids:
            raise ValueError(f"Fold {fold_id} does not completely assign its eligible spots")
        if set(actual["role"]) != set(ROLES):
            raise ValueError(f"Fold {fold_id} must contain train, validation, and test roles")
        indexed = actual.set_index(list(IDENTITY_COLUMNS))
        for column in ("donor_id", "ground_truth", "global_row", "section_row"):
            expected_values = dataset_index.loc[indexed.index, column].astype(str).to_numpy()
            actual_values = indexed[column].astype(str).to_numpy()
            if not (expected_values == actual_values).all():
                raise ValueError(f"Fold {fold_id} has altered source column {column}")
        expected_hashes = [
            _sha256_assignment(str(row.family), str(row.fold_id), int(row.seed), str(row.section_id), str(row.barcode))
            for row in actual.itertuples(index=False)
        ]
        if expected_hashes != actual["assignment_sha256"].tolist():
            raise ValueError(f"Fold {fold_id} has invalid assignment SHA256 values")
        if fold["role_source"] == "section_membership":
            test = set(fold["test_sections"])
            validation = set(fold["validation_sections"])
            expected_roles = actual["section_id"].map(lambda value: "test" if value in test else "validation" if value in validation else "train")
            if not (expected_roles.to_numpy() == actual["role"].to_numpy()).all():
                raise ValueError(f"Fold {fold_id} roles do not match section definitions")
        elif fold["role_source"] == "sha256_stratified_random":
            expected = actual.copy()
            _assign_random_roles(expected, metadata["config"]["ratios"], metadata["config"]["strata"])
            if not (expected["role"].to_numpy() == actual["role"].to_numpy()).all():
                raise ValueError(f"Fold {fold_id} roles do not match SHA256-ranked assignment")


def write_split_artifact(
    split: SplitManifest,
    output_root: str | Path,
    artifact_id: str,
    source_manifest: str | Path,
) -> tuple[Path, Path]:
    """Write Parquet plus JSON metadata into a newly-created immutable directory."""
    if not artifact_id or Path(artifact_id).name != artifact_id:
        raise ValueError("artifact_id must be one non-empty path component")
    artifact_dir = Path(output_root) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    parquet_path = artifact_dir / "split_manifest.parquet"
    metadata_path = artifact_dir / "metadata.json"
    split.table.to_parquet(parquet_path, index=False)
    parquet_digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    metadata = dict(split.metadata)
    metadata.update({
        "artifact_id": artifact_id,
        "immutable": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_manifest": str(Path(source_manifest).as_posix()),
        "split_manifest_file": parquet_path.name,
        "split_manifest_sha256": parquet_digest,
    })
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return parquet_path, metadata_path


def generate_all_artifacts(
    source_manifest: str | Path,
    output_root: str | Path,
    random_seeds: Sequence[int] = (0,),
    id_prefix: str = "20260815_dlpfc",
) -> list[tuple[Path, Path]]:
    """Generate the five canonical split families without overwriting any artifact."""
    dataset = load_dataset_manifest(source_manifest)
    seed_values = tuple(int(seed) for seed in random_seeds)
    seed_tag = f"seed{seed_values[0]}" if len(seed_values) == 1 else "seeds" + "-".join(map(str, seed_values))
    families = (
        ("lodo_v1", generate_leave_one_donor_out(dataset)),
        ("loso_v1", generate_leave_one_section_out(dataset)),
        ("within_donor_section_v1", generate_within_donor_section_holdout(dataset)),
        (f"pooled_random_spot_{seed_tag}_v1", generate_pooled_random_spot(dataset, seeds=seed_values)),
        (f"within_donor_random_spot_{seed_tag}_v1", generate_within_donor_random_spot(dataset, seeds=seed_values)),
    )
    return [
        write_split_artifact(split, output_root, f"{id_prefix}_{suffix}", source_manifest)
        for suffix, split in families
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("outputs/evaluation/20260813_checkpoint_audit_v1/data/dataset_manifest.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--id-prefix", default="20260815_dlpfc")
    args = parser.parse_args()
    for parquet_path, metadata_path in generate_all_artifacts(args.source_manifest, args.output_root, args.random_seeds, args.id_prefix):
        print(f"wrote {parquet_path} and {metadata_path}")


if __name__ == "__main__":
    main()
