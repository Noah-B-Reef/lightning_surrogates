"""Data pipeline for the GOW17 PINN.

Follows models/mlp/src/data.py, with two changes:

1. Samples are (state at t) -> (state at t + k*dt) pairs for k = 1..MAX_HORIZON
   instead of only adjacent snapshots, and the elapsed time Delta-t (in years)
   is appended as the last input feature. This makes time a real model input.
2. Abundance columns are re-indexed to gow17_rates.SPECIES order so the
   model's outputs line up with the chemical RHS used in the physics loss.

Input layout:  [physical parameters at t (raw), log10 abundances at t, dt_years]
Target layout: log10 abundances at t + dt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

import settings as config
from gow17_rates import SPECIES

# Same physical inputs as the MLP: dustTemp duplicates gasTemp and zeta is
# constant, so neither is a feature. zeta is still read separately because the
# physics loss needs it (see GravCollapseDataModule.zeta_column_value).
PHYS_COLS = ("Density", "gasTemp", "Av", "radfield")
PHYS_LOG_COLS = ("Density", "Av", "radfield")
PHYS_LOG_FLOOR_DEFAULT = 1e-30
PHYS_LOG_FLOORS = {"radfield": 1e-4}
EXCLUDED_COLS = ("Tracer", "Time", "dstep", "BULK", "SURFACE", "dustTemp", "zeta")

ABUND_LOG_FLOOR = 1e-30


def phys_log_floors(phys_cols):
    """Per-column raw-space floor array aligned to ``phys_cols``."""
    return np.array(
        [PHYS_LOG_FLOORS.get(col, PHYS_LOG_FLOOR_DEFAULT) for col in phys_cols],
        dtype=np.float64,
    )


def load_split_dataframe(split_dir, split_name):
    """Load one split as a long-format DataFrame (.npy + columns.json or .csv)."""
    split_dir = Path(split_dir).expanduser().resolve()
    csv_path = split_dir / f"{split_name}.csv"
    npy_path = split_dir / f"{split_name}.npy"
    if npy_path.is_file():
        columns_path = split_dir / "columns.json"
        if not columns_path.is_file():
            raise FileNotFoundError(f"Missing columns.json next to {npy_path}")
        columns = json.loads(columns_path.read_text())
        return pd.DataFrame(np.load(npy_path), columns=columns)
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Split '{split_name}' not found in {split_dir} (.npy or .csv)"
    )


class GOW17PairDataset(Dataset):
    """Multi-horizon transition samples from a sampled split directory."""

    def __init__(self, split_dir, split_name, max_horizon=config.MAX_HORIZON):
        df = load_split_dataframe(split_dir, split_name)
        df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

        all_cols = df.columns.tolist()
        missing = [s for s in SPECIES if s not in all_cols]
        if missing:
            raise ValueError(
                f"Split {split_dir}/{split_name} lacks GOW17 species {missing}"
            )
        phys_cols = [col for col in PHYS_COLS if col in all_cols]
        abundance_cols = list(SPECIES)  # fixed order shared with gow17_rates
        self._phys_cols = phys_cols
        self._abundance_cols = abundance_cols
        self._feature_names = phys_cols + abundance_cols + ["dt_years"]
        self._target_names = abundance_cols
        self.max_horizon = max(1, int(max_horizon))

        self._time = df["Time"].to_numpy(dtype=np.float64)
        self._phys = df[phys_cols].to_numpy(dtype=np.float32)
        abund = df[abundance_cols].to_numpy(dtype=np.float32)
        self._abund = np.log10(np.maximum(abund, ABUND_LOG_FLOOR)).astype(
            np.float32, copy=False
        )
        self.zeta_column_value = (
            float(df["zeta"].iloc[0]) if "zeta" in all_cols else 1.0
        )

        # Sample index: (start row, horizon k) for every start row that has a
        # k-th successor within the same tracer.
        starts, horizons = [], []
        for _, row_indices in df.groupby("Tracer", sort=False).indices.items():
            num_steps = len(row_indices)
            first_row = int(row_indices[0])
            for k in range(1, self.max_horizon + 1):
                if num_steps <= k:
                    break
                s = first_row + np.arange(num_steps - k, dtype=np.int64)
                starts.append(s)
                horizons.append(np.full(num_steps - k, k, dtype=np.int64))
        self._sample_starts = (
            np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
        )
        self._sample_horizons = (
            np.concatenate(horizons) if horizons else np.empty(0, dtype=np.int64)
        )

    def __len__(self):
        return len(self._sample_starts)

    def __getitem__(self, idx):
        start = self._sample_starts[idx]
        end = start + self._sample_horizons[idx]
        dt_years = np.float32(self._time[end] - self._time[start])
        inputs = np.concatenate(
            [self._phys[start], self._abund[start], [dt_years]]
        ).astype(np.float32, copy=False)
        target = self._abund[end]
        return torch.from_numpy(inputs), torch.from_numpy(target)

    def feature_names(self):
        return self._feature_names

    @property
    def target_names(self):
        return self._target_names

    @property
    def num_features(self):
        return len(self._feature_names)

    @property
    def num_targets(self):
        return len(self._target_names)

    @property
    def num_phys(self):
        return len(self._phys_cols)

    def phys_stats(self):
        """Per-column normalization stats for the physical parameters.

        Same scheme as the MLP: multi-decade columns are log10-transformed
        (after a raw-space floor) before standardization; stats are applied
        inside the model's forward pass.
        """
        n = len(self._phys_cols)
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty.copy(), empty.copy()
        mask = np.array(
            [col in PHYS_LOG_COLS for col in self._phys_cols], dtype=bool
        )
        floor = phys_log_floors(self._phys_cols)
        transformed = self._phys.astype(np.float64, copy=True)
        if mask.any():
            transformed[:, mask] = np.log10(
                np.maximum(transformed[:, mask], floor[mask])
            )
        mean = transformed.mean(axis=0)
        std = transformed.std(axis=0)
        std[std < 1e-8] = 1.0
        return (
            mask.astype(np.float32),
            mean.astype(np.float32),
            std.astype(np.float32),
            floor.astype(np.float32),
        )


class GOW17DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=str(config.DEFAULT_SPLIT_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        max_horizon=config.MAX_HORIZON,
        pin_memory=False,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_horizon = int(max_horizon)
        self.pin_memory = bool(pin_memory)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = GOW17PairDataset(self.data_dir, "train", self.max_horizon)
            self.val_ds = GOW17PairDataset(self.data_dir, "val", self.max_horizon)
        if stage in (None, "test", "predict"):
            self.test_ds = GOW17PairDataset(self.data_dir, "test", self.max_horizon)
        schema_ds = getattr(self, "train_ds", None) or getattr(self, "test_ds", None)
        self._num_features = schema_ds.num_features
        self._num_targets = schema_ds.num_targets
        self._feature_names = schema_ds.feature_names()
        self._target_names = schema_ds.target_names
        stats_ds = getattr(self, "train_ds", None) or schema_ds
        self._num_phys = stats_ds.num_phys
        self._phys_stats = stats_ds.phys_stats()
        self._zeta_column_value = stats_ds.zeta_column_value

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_targets(self):
        return self._num_targets

    @property
    def num_phys(self):
        return self._num_phys

    def phys_norm_config(self):
        """Normalization + physics fields to merge into the model config."""
        log_mask, mean, std, floor = self._phys_stats
        return {
            "num_phys": int(self._num_phys),
            "phys_cols": list(self._feature_names[: self._num_phys]),
            "phys_log_mask": log_mask.tolist(),
            "phys_mean": mean.tolist(),
            "phys_std": std.tolist(),
            "phys_log_floor": floor.tolist(),
            "zeta_column_value": float(self._zeta_column_value),
        }

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def target_names(self):
        return self._target_names
