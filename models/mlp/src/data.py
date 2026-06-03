import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

import settings as config


PHYS_COLS = ("Density", "gasTemp", "dustTemp", "Av", "radfield", "zeta")


class GravCollapseDataset(Dataset):
    """
    Build one-step transition samples from a sampled split CSV.

    Input layout:
        [physical parameters at t, log10 abundances at t]

    Target layout:
        log10 abundances at t + 1
    """

    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

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
        sample_tracers = []
        for tracer_id, row_indices in df.groupby("Tracer", sort=False).indices.items():
            num_steps = len(row_indices)
            if num_steps <= 1:
                continue
            first_row = int(row_indices[0])
            valid_starts = first_row + np.arange(num_steps - 1, dtype=np.int64)
            starts.append(valid_starts)
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

    def __len__(self):
        return len(self._sample_starts)

    def __getitem__(self, idx):
        start = self._sample_starts[idx]
        initial = np.concatenate([self._phys[start], self._abund[start]]).astype(
            np.float32, copy=False
        )
        target = self._abund[start + 1]
        return torch.from_numpy(initial), torch.from_numpy(target)

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

    def tracer_ids(self):
        return self._sample_tracers.copy()


class GravCollapseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=str(config.DEFAULT_SPLIT_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = GravCollapseDataset(os.path.join(self.data_dir, "train.csv"))
            self.val_ds = GravCollapseDataset(os.path.join(self.data_dir, "val.csv"))
        if stage in (None, "test", "predict"):
            self.test_ds = GravCollapseDataset(os.path.join(self.data_dir, "test.csv"))
        schema_ds = getattr(self, "train_ds", None) or getattr(self, "test_ds", None)
        self._num_features = schema_ds.num_features
        self._num_targets = schema_ds.num_targets
        self._feature_names = schema_ds.feature_names()
        self._target_names = schema_ds.target_names

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
    def feature_names(self):
        return self._feature_names

    @property
    def target_names(self):
        return self._target_names
