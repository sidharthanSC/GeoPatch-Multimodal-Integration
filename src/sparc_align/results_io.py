"""Result persistence for SPARC runs.

Every experiment writes to ``outputs/sparc_align/`` rather than to stdout alone.
Rows are flushed as they are produced, so a crashed or interrupted run still leaves
the completed rows on disk -- the failure mode that previously left an empty run
directory which then blocked relaunch.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

RESULTS_ROOT = Path("outputs/sparc_align")


def results_dir(name: str, root: Path | None = None) -> Path:
    """Create and return ``outputs/sparc_align/<name>/``."""
    directory = (root or RESULTS_ROOT) / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class IncrementalCsv:
    """Append-as-you-go CSV writer tolerant of rows gaining new keys.

    Ablation arms produce different columns (only the stage-2 arms emit ``attn_*``),
    so the header is recomputed and the file rewritten whenever an unseen key
    appears. Files stay small enough that rewriting is cheaper than reconciling
    schemas by hand after a crash.
    """

    def __init__(self, path: Path, resume: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.fieldnames: list[str] = []
        if resume and path.exists():
            # Without this, a restarted run begins with an empty row list and the
            # first flush overwrites everything the previous process completed.
            with path.open(encoding="utf-8") as handle:
                self.rows = [dict(r) for r in csv.DictReader(handle)]
            if self.rows:
                self.fieldnames = sorted({k for row in self.rows for k in row})

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        new_keys = [k for k in row if k not in self.fieldnames]
        if new_keys:
            self.fieldnames = sorted({*self.fieldnames, *new_keys})
        self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)


def save_rows(name: str, rows: Iterable[dict[str, Any]], filename: str = "results.csv") -> Path:
    """Write rows to ``outputs/sparc_align/<name>/<filename>`` and return the path."""
    rows = list(rows)
    directory = results_dir(name)
    path = directory / filename
    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return path


def save_json(name: str, payload: Any, filename: str = "summary.json") -> Path:
    """Write a JSON payload under ``outputs/sparc_align/<name>/``."""
    path = results_dir(name) / filename
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
