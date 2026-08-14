"""Common frozen logistic probes over saved multimodal representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.multimodal.representations import existing_embedding_representations


def run_logistic_probes(
    manifest_path: Path,
    embeddings_npz: Path,
    output_dir: Path,
    gene_npz: Path | None = None,
    cross_modal_npz: Path | None = None,
    seeds: tuple[int, ...] = tuple(range(5)),
    test_fraction: float = 0.3,
) -> dict:
    metrics_path = output_dir / "metrics.parquet"
    predictions_path = output_dir / "predictions.parquet"
    summary_path = output_dir / "summary.json"
    if any(path.exists() for path in (metrics_path, predictions_path, summary_path)):
        raise FileExistsError(f"Refusing to overwrite artifacts under {output_dir}")

    manifest = pd.read_parquet(manifest_path)
    valid = manifest["ground_truth_valid"].to_numpy()
    labels = manifest.loc[valid, "ground_truth"].to_numpy()
    with np.load(embeddings_npz) as data:
        source = {key: data[key] for key in data.files}
    if gene_npz is not None:
        with np.load(gene_npz, allow_pickle=True) as data:
            source["gene_emb"] = data["embeddings"]
    if cross_modal_npz is not None:
        with np.load(cross_modal_npz, allow_pickle=True) as data:
            source["gene_emb_cm_img"] = data["gene_projected"]
            source["img_emb_cm"] = data["image_projected"]
    representations = {
        name: values[valid]
        for name, values in existing_embedding_representations(source).items()
    }

    metric_records = []
    prediction_records = []
    indices = np.arange(len(labels))
    for seed in seeds:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=labels,
        )
        for name, values in representations.items():
            scaler = StandardScaler().fit(values[train_idx])
            train_values = scaler.transform(values[train_idx])
            test_values = scaler.transform(values[test_idx])
            model = LogisticRegression(
                C=1.0,
                max_iter=300,
                tol=1e-3,
                solver="saga",
                random_state=seed,
            ).fit(train_values, labels[train_idx])
            predicted = model.predict(test_values)
            metric_records.append(
                {
                    "representation": name,
                    "seed": seed,
                    "split": "random_spot_transductive",
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "accuracy": accuracy_score(labels[test_idx], predicted),
                    "balanced_accuracy": balanced_accuracy_score(labels[test_idx], predicted),
                    "macro_f1": f1_score(labels[test_idx], predicted, average="macro"),
                }
            )
            prediction_records.extend(
                {
                    "representation": name,
                    "seed": seed,
                    "global_row": int(manifest.loc[valid].iloc[row]["global_row"]),
                    "section_id": str(manifest.loc[valid].iloc[row]["section_id"]),
                    "barcode": str(manifest.loc[valid].iloc[row]["barcode"]),
                    "true_label": true,
                    "predicted_label": pred,
                }
                for row, true, pred in zip(test_idx, labels[test_idx], predicted)
            )

    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    summary_json = {
        representation: {
            metric: {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std()),
            }
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        for representation, group in metrics.groupby("representation")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    with summary_path.open("w") as file:
        json.dump(summary_json, file, indent=2)
    return summary_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--gene-npz", type=Path, default=None)
    parser.add_argument("--cross-modal-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_logistic_probes(
        args.manifest_path,
        args.embeddings_npz,
        args.output_dir,
        args.gene_npz,
        args.cross_modal_npz,
        tuple(args.seeds),
        args.test_fraction,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
