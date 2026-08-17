import pandas as pd

from src.train.aggregate_supervised_results import _aggregate


def test_aggregate_uses_equal_fold_rows():
    frame = pd.DataFrame(
        {
            "regime": ["r", "r"],
            "method": ["m", "m"],
            "family": ["f", "f"],
            "fold_id": ["a", "b"],
            "accuracy": [0.2, 0.8],
            "balanced_accuracy": [0.3, 0.7],
            "macro_f1": [0.4, 0.6],
        }
    )
    result = _aggregate(frame, ["regime", "method", "family"]).iloc[0]
    assert result["n_folds"] == 2
    assert result["accuracy_mean"] == 0.5
    assert result["balanced_accuracy_median"] == 0.5
