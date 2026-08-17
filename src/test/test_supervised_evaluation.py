import numpy as np

from src.train.supervised_evaluation import classification_metrics, log_euclidean_mdm


def test_classification_metrics_preserve_absent_layer_as_none():
    result = classification_metrics(
        np.asarray(["Layer_3", "Layer_4", "Layer_3"]),
        np.asarray(["Layer_3", "Layer_3", "Layer_3"]),
    )
    assert result["per_layer_recall"]["Layer_1"] is None
    assert result["per_layer_recall"]["Layer_3"] == 1.0
    assert np.asarray(result["confusion_matrix"]).shape == (7, 7)


def test_log_euclidean_mdm_classifies_separated_toy_pairs():
    image = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [1.05, 0.0, 0.0],
            [0.0, 1.05, 0.0],
        ]
    )
    gene = np.zeros_like(image)
    labels = np.asarray(["Layer_1", "Layer_1", "Layer_2", "Layer_2", "Layer_1", "Layer_2"])
    predicted = log_euclidean_mdm(
        image, gene, labels, np.asarray([0, 1, 2, 3]), np.asarray([4, 5]), 1e-3, chunk_size=2
    )
    assert predicted.tolist() == labels[[4, 5]].tolist()
