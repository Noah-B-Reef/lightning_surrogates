import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

import settings as config


# dustTemp and zeta were dropped from the inputs: in this dataset dustTemp is
# an exact duplicate of gasTemp (no information) and zeta is constant (a dead
# input that only ever standardizes to zero). Av is multi-decade and strongly
# right-skewed, so it is log10-transformed alongside Density and radfield.
PHYS_COLS = ("Density", "gasTemp", "Av", "radfield")

# Physical parameters that span many orders of magnitude and are strictly
# positive. These are log10-transformed before standardization; the rest are
# standardized directly.
PHYS_LOG_COLS = ("Density", "Av", "radfield")

# Per-column raw-space floor applied before log10. radfield contains exact
# zeros and near-denormal values (~1e-45) that would map to extreme negative
# outliers (~-16 sigma) after the log; flooring at 1e-4 (below the 0.1th
# percentile of real radfield values, ~3.9e-4) collapses that junk without
# clipping any legitimate data. Columns not listed fall back to a tiny epsilon
# that has no real effect on their well-behaved ranges.
PHYS_LOG_FLOOR_DEFAULT = 1e-30
PHYS_LOG_FLOORS = {"radfield": 1e-4}
# dustTemp and zeta are dropped entirely: they are neither model inputs nor
# prediction targets. Without this they would fall through into the
# abundance/species columns and be predicted as spurious targets.
EXCLUDED_COLS = ("Tracer", "Time", "dstep", "BULK", "SURFACE", "dustTemp", "zeta")


def phys_log_floors(phys_cols):
    """Per-column raw-space floor array aligned to ``phys_cols``."""
    return np.array(
        [PHYS_LOG_FLOORS.get(col, PHYS_LOG_FLOOR_DEFAULT) for col in phys_cols],
        dtype=np.float64,
    )


def load_split_dataframe(split_dir, split_name):
    """Load one split as a long-format DataFrame.

    Supports both storage formats produced by the samplers:
        {split_dir}/{split_name}.csv
        {split_dir}/{split_name}.npy  (+ columns.json)
    """
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


class GravCollapseDataset(Dataset):
    """
    Rollout windows of ``rollout_steps`` consecutive transitions from a
    sampled split directory. Each sample is

        phys_seq: [rollout_steps, num_phys]   physical drivers at t .. t+k-1
        abund0:   [num_species]               log10 abundances at t
        targets:  [rollout_steps, num_species] log10 abundances at t+1 .. t+k

    The model consumes the true drivers at every step but feeds its own
    abundance predictions forward, so windows must be contiguous within a
    single tracer. ``rollout_steps=1`` reproduces the old one-step samples.
    """

    def __init__(self, split_dir, split_name, rollout_steps=1):
        self._rollout_steps = max(1, int(rollout_steps))
        df = load_split_dataframe(split_dir, split_name)
        df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

        all_cols = df.columns.tolist()
        phys_cols = [col for col in PHYS_COLS if col in all_cols]
        abundance_cols = [
            col for col in all_cols if col not in EXCLUDED_COLS and col not in phys_cols
        ]
        self._phys_cols = phys_cols
        self._abundance_cols = abundance_cols
        self._feature_names = phys_cols + abundance_cols
        self._target_names = abundance_cols

        self._phys = df[phys_cols].to_numpy(dtype=np.float32)
        abund = df[abundance_cols].to_numpy(dtype=np.float32)
        self._abund = np.log10(np.maximum(abund, config.ABUND_FLOOR)).astype(
            np.float32, copy=False
        )

        # Sample index: every row with rollout_steps successor rows of the
        # same tracer (windows never cross a tracer boundary).
        window = self._rollout_steps
        starts = []
        for _, row_indices in df.groupby("Tracer", sort=False).indices.items():
            num_steps = len(row_indices)
            if num_steps <= window:
                continue
            first_row = int(row_indices[0])
            starts.append(first_row + np.arange(num_steps - window, dtype=np.int64))
        self._sample_starts = (
            np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
        )
        if len(self._sample_starts) == 0:
            raise ValueError(
                f"Split '{split_name}' has no tracer with more than "
                f"{window} steps; cannot build rollout windows."
            )

    def __len__(self):
        return len(self._sample_starts)

    def __getitem__(self, idx):
        start = self._sample_starts[idx]
        end = start + self._rollout_steps
        phys_seq = self._phys[start:end]
        abund0 = self._abund[start]
        targets = self._abund[start + 1 : end + 1]
        return (
            torch.from_numpy(phys_seq),
            torch.from_numpy(abund0),
            torch.from_numpy(targets),
        )

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

        Returns ``(log_mask, mean, std, floor)`` as float32 arrays of length
        ``num_phys``. Multi-decade columns (see ``PHYS_LOG_COLS``) are
        log10-transformed before the mean/std are computed, after applying the
        per-column raw floor (see ``PHYS_LOG_FLOORS``). The same mask, floor,
        mean, and std are applied inside the model's forward pass.
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
        std[std < 1e-8] = 1.0  # guard against constant columns
        return (
            mask.astype(np.float32),
            mean.astype(np.float32),
            std.astype(np.float32),
            floor.astype(np.float32),
        )


class GravCollapseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=str(config.DEFAULT_SPLIT_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        rollout_steps=config.ROLLOUT_STEPS,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.rollout_steps = max(1, int(rollout_steps))

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = GravCollapseDataset(
                self.data_dir, "train", rollout_steps=self.rollout_steps
            )
            self.val_ds = GravCollapseDataset(
                self.data_dir, "val", rollout_steps=self.rollout_steps
            )
        if stage in (None, "test", "predict"):
            self.test_ds = GravCollapseDataset(
                self.data_dir, "test", rollout_steps=self.rollout_steps
            )
        schema_ds = getattr(self, "train_ds", None) or getattr(self, "test_ds", None)
        self._num_features = schema_ds.num_features
        self._num_targets = schema_ds.num_targets
        self._feature_names = schema_ds.feature_names()
        self._target_names = schema_ds.target_names
        # Normalization stats are always derived from the training split when
        # it is available, so val/test never leak into the transform.
        stats_ds = getattr(self, "train_ds", None) or schema_ds
        self._num_phys = stats_ds.num_phys
        self._phys_stats = stats_ds.phys_stats()

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
        """Normalization fields to merge into the model config (JSON-safe)."""
        log_mask, mean, std, floor = self._phys_stats
        return {
            "num_phys": int(self._num_phys),
            "phys_log_mask": log_mask.tolist(),
            "phys_mean": mean.tolist(),
            "phys_std": std.tolist(),
            "phys_log_floor": floor.tolist(),
        }

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def target_names(self):
        return self._target_names
