from __future__ import annotations

from pathlib import Path
import json
import xarray as xr


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def open_trajectory_dataset(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)