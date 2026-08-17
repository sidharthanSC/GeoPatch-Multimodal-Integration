"""Resumable orchestration for independent STAIG feature conditions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.prior_models.staig.config import StaigConfig
from src.prior_models.staig.data import load_feature_bundle
from src.prior_models.staig.train import run_dlpfc


@dataclass(frozen=True)
class StaigFeatureCondition:
    run_id: str
    node_npz: Path
    node_key: str
    image_npz: Path | None = None
    image_key: str | None = None


def is_complete_run(path: Path) -> bool:
    return (
        (path / "summary.json").exists()
        and len(list((path / "checkpoints").glob("*.pt"))) == 12
        and len(list((path / "embeddings").glob("*.npz"))) == 12
    )


def run_staig_matrix(
    checkpoint_path: Path,
    output_root: Path,
    conditions: tuple[StaigFeatureCondition, ...],
    config: StaigConfig,
    device: str | None = None,
) -> dict[str, str]:
    """Run missing conditions and skip only fully verified immutable runs."""
    statuses = {}
    for condition in conditions:
        output_dir = output_root / condition.run_id
        if output_dir.exists():
            if not is_complete_run(output_dir):
                raise FileExistsError(f"Incomplete existing STAIG run requires a new ID: {output_dir}")
            statuses[condition.run_id] = "existing_verified"
            continue
        node = load_feature_bundle(condition.node_npz, condition.node_key)
        image = (
            load_feature_bundle(condition.image_npz, condition.image_key)
            if condition.image_npz is not None and condition.image_key is not None
            else None
        )
        run_dlpfc(
            checkpoint_path,
            output_dir,
            config,
            device=device,
            node_features=node,
            image_features=image,
        )
        if not is_complete_run(output_dir):
            raise RuntimeError(f"STAIG run did not produce complete artifacts: {output_dir}")
        statuses[condition.run_id] = "completed"
    return statuses


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/prior_models"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    conditions = tuple(
        StaigFeatureCondition(
            run_id=record["run_id"],
            node_npz=Path(record["node_npz"]),
            node_key=record["node_key"],
            image_npz=Path(record["image_npz"]) if record.get("image_npz") else None,
            image_key=record.get("image_key"),
        )
        for record in records
    )
    statuses = run_staig_matrix(
        args.checkpoint_path,
        args.output_root,
        conditions,
        StaigConfig(epochs=args.epochs, seed=args.seed),
        args.device,
    )
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
