from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import unique_normalized_names


def load_csv_files(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """Load every CSV in the data folder and normalize column names."""
    data_path = Path(data_dir)
    datasets: dict[str, pd.DataFrame] = {}
    if not data_path.exists():
        return datasets

    for csv_path in sorted(data_path.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path, low_memory=False)
            frame.columns = unique_normalized_names(frame.columns)
            datasets[csv_path.stem] = frame
        except Exception as exc:
            datasets[csv_path.stem] = pd.DataFrame({"load_error": [str(exc)]})
    return datasets
