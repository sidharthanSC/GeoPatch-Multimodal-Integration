"""Load DlpfcDataset from checkpoints/dlpfc.pkl and print one sample dict.

Usage:
    python -m src.test.print_sample
"""

from pathlib import Path

from src.datasets.dlpfc import DlpfcDataset

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "dlpfc.pkl"


def main() -> None:
    dataset = DlpfcDataset.from_checkpoint(CHECKPOINT_PATH)
    #print(dataset)

    sample = dataset[0]
    print("\nSample 0:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            #print(f"  {key!r}: {type(value).__name__}, shape={tuple(value.shape)}, dtype={value.dtype}")
            print(f"{key}: {tuple(value.shape)}, {value.dtype}")

            #print(f"    {value}")
            print("\n")
        else:
            print(f"  {key!r}: {value!r}")


if __name__ == "__main__":
    main()
