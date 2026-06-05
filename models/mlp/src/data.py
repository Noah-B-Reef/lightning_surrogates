import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

import settings as config


PHYS_COLS = ("Density", "gasTemp", "dustTemp", "Av", "radfield", "zeta")

# Physical parameters that span many orders of magnitude and are strictly
# positive. These are log10-transformed before standardization; the rest are
# standardized directly.
PHYS_LOG_COLS = ("Density", "radfield", "zeta")


class GravCollapseDataset(Dataset):
    """
    Build one-step transition samples from a sampled split CSV.

    Input layout:
        [physical parameters at t, log10 abundances at t]

    Target layout:
        log10 abundances at t + 1
    """

    def __init__(self, csv_path, max_rollout_steps=1):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        self.max_rollout_steps = max(1, int(max_rollout_steps))

        header = pd.read_csv(csv_path, nrows=0)
        all_cols = header.columns.tolist()
        phys_cols = [col for col in PHYS_COLS if col in all_cols]
        abundance_cols = [
            col
            for col in all_cols
            if col not in ("Tracer", "Time", "dstep", "BULK", "SURFACE")
            and col not in phys_cols
        ]
        usecols = ["Tracer", "Time"] + phys_cols + abundance_cols
        dtype = {col: np.float32 for col in phys_cols + abundance_cols}
        df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype)
        df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

        self._phys_cols = phys_cols
        self._abundance_cols = abundance_cols
        self._feature_names = phys_cols + abundance_cols
        self._target_names = abundance_cols
        self._tracer_by_row = df["Tracer"].to_numpy()
        self._time_by_row = df["Time"].to_numpy()
        self._phys = df[phys_cols].to_numpy(dtype=np.float32, copy=True)
        abund = df[abundance_cols].to_numpy(dtype=np.float32, copy=True)
        self._abund = np.log10(np.maximum(abund, 1e-30)).astype(np.float32, copy=False)

        starts = []
        rollout_lengths = []
        sample_tracers = []
        for tracer_id, row_indices in df.groupby("Tracer", sort=False).indices.items():
            num_steps = len(row_indices)
            if num_steps <= 1:
                continue
            first_row = int(row_indices[0])
            offsets = np.arange(num_steps - 1, dtype=np.int64)
            valid_starts = first_row + offsets
            valid_lengths = np.minimum(self.max_rollout_steps, num_steps - 1 - offsets)
            starts.append(valid_starts)
            rollout_lengths.append(valid_lengths.astype(np.int64, copy=False))
            sample_tracers.append(np.full(len(valid_starts), tracer_id))

        self._sample_starts = (
            np.concatenate(starts).astype(np.int64, copy=False)
            if starts
            else np.empty(0, dtype=np.int64)
        )
        self._sample_tracers = (
            np.concatenate(sample_tracers)
            if sample_tracers
            else np.empty(0, dtype=self._tracer_by_row.dtype)
        )
        self._rollout_lengths = (
            np.concatenate(rollout_lengths).astype(np.int64, copy=False)
            if rollout_lengths
            else np.empty(0, dtype=np.int64)
        )

    def __len__(self):
        return len(self._sample_starts)

    def __getitem__(self, idx):
        start = self._sample_starts[idx]
        initial = np.concatenate([self._phys[start], self._abund[start]]).astype(
            np.float32, copy=False
        )
        horizon = self.max_rollout_steps
        valid_steps = int(self._rollout_lengths[idx])
        phys_seq = np.zeros((horizon, len(self._phys_cols)), dtype=np.float32)
        target_seq = np.zeros((horizon, len(self._target_names)), dtype=np.float32)
        mask = np.zeros(horizon, dtype=np.float32)
        for step in range(valid_steps):
            phys_seq[step] = self._phys[start + step]
            target_seq[step] = self._abund[start + step + 1]
            mask[step] = 1.0
        return (
            torch.from_numpy(initial),
            torch.from_numpy(phys_seq),
            torch.from_numpy(target_seq),
            torch.from_numpy(mask),
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
    def num_phys(self):
        return len(self._phys_cols)

    def phys_stats(self):
        """Per-column normalization stats for the physical parameters.

        Returns ``(log_mask, mean, std)`` as float32 arrays of length
        ``num_phys``. Multi-decade columns (see ``PHYS_LOG_COLS``) are
        log10-transformed before the mean/std are computed, matching the
        transform applied inside the model's forward pass.
        """
        n = len(self._phys_cols)
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty.copy()
        mask = np.array(
            [col in PHYS_LOG_COLS for col in self._phys_cols], dtype=bool
        )
        transformed = self._phys.astype(np.float64, copy=True)
        if mask.any():
            transformed[:, mask] = np.log10(
                np.maximum(transformed[:, mask], 1e-30)
            )
        mean = transformed.mean(axis=0)
        std = transformed.std(axis=0)
        std[std < 1e-8] = 1.0  # guard against constant columns
        return (
            mask.astype(np.float32),
            mean.astype(np.float32),
            std.astype(np.float32),
        )

    @property
    def num_targets(self):
        return len(self._target_names)

    def tracer_ids(self):
        return self._sample_tracers.copy()


class GravCollapseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=str(config.DEFAULT_SPLIT_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        max_rollout_steps=config.ROLLOUT_STEPS,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_rollout_steps = max(1, int(max_rollout_steps))

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = GravCollapseDataset(
                os.path.join(self.data_dir, "train.csv"),
                max_rollout_steps=self.max_rollout_steps,
            )
            self.val_ds = GravCollapseDataset(
                os.path.join(self.data_dir, "val.csv"),
                max_rollout_steps=self.max_rollout_steps,
            )
        if stage in (None, "test", "predict"):
            self.test_ds = GravCollapseDataset(
                os.path.join(self.data_dir, "test.csv"),
                max_rollout_steps=self.max_rollout_steps,
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

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

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
        log_mask, mean, std = self._phys_stats
        return {
            "num_phys": int(self._num_phys),
            "phys_log_mask": log_mask.tolist(),
            "phys_mean": mean.tolist(),
            "phys_std": std.tolist(),
            "rollout_steps": int(self.max_rollout_steps),
        }

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def target_names(self):
        return self._target_names
