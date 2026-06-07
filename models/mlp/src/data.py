import json
import os
import shutil
import time
from pathlib import Path

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
EXCLUDED_COLS = ("Tracer", "Time", "dstep", "BULK", "SURFACE")
CACHE_VERSION = 1


def _select_sample_indices(sample_tracers, stride=1, fraction=1.0):
    stride = max(1, int(stride))
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample fraction must be in (0, 1]")

    indices = np.arange(len(sample_tracers), dtype=np.int64)
    if stride > 1 and len(indices):
        keep = np.zeros(len(indices), dtype=bool)
        boundaries = np.flatnonzero(
            np.r_[True, sample_tracers[1:] != sample_tracers[:-1], True]
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            keep[start:end:stride] = True
        indices = indices[keep]
    if fraction < 1.0 and len(indices):
        count = max(1, int(round(len(indices) * fraction)))
        positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
        indices = indices[positions]
    return indices


class CompactRolloutCollator:
    """Vectorize rollout construction from row-level memory-mapped arrays."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch_indices):
        ds = self.dataset
        indices = np.asarray(batch_indices, dtype=np.int64)
        starts = np.asarray(ds._sample_starts[indices])
        lengths = np.asarray(ds._rollout_lengths[indices])
        steps = np.arange(ds.max_rollout_steps, dtype=np.int64)
        valid = lengths[:, None] > steps[None, :]
        rows = starts[:, None] + steps[None, :]
        safe_rows = np.where(valid, rows, starts[:, None])

        initial = np.concatenate(
            [np.asarray(ds._phys[starts]), np.asarray(ds._abund[starts])], axis=1
        ).astype(np.float32, copy=False)
        phys_seq = np.asarray(ds._phys[safe_rows], dtype=np.float32)
        target_seq = np.asarray(ds._abund[safe_rows + 1], dtype=np.float32)
        mask = valid.astype(np.float32)
        phys_seq[~valid] = 0.0
        target_seq[~valid] = 0.0
        return tuple(
            torch.from_numpy(array)
            for array in (initial, phys_seq, target_seq, mask)
        )


def _cache_dir_for(csv_path, max_rollout_steps, compact=False):
    csv_path = Path(csv_path).expanduser().resolve()
    layout = "_compact" if compact else ""
    return (
        csv_path.parent
        / ".preprocessed"
        / f"{csv_path.stem}_rollout{int(max_rollout_steps)}{layout}_v{CACHE_VERSION}"
    )


def _read_cache_metadata(cache_dir):
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None


def _cache_is_valid(cache_dir, csv_path, max_rollout_steps, compact=False):
    metadata = _read_cache_metadata(cache_dir)
    if metadata is None:
        return False
    csv_path = Path(csv_path).expanduser().resolve()
    expected_files = [
        "phys.npy",
        "abund.npy",
        "sample_starts.npy",
        "rollout_lengths.npy",
        "sample_tracers.npy",
    ]
    if not compact:
        expected_files.extend(("initial.npy", "phys_seq.npy", "target_seq.npy", "mask.npy"))
    if not all((cache_dir / name).is_file() for name in expected_files):
        return False
    stat = csv_path.stat()
    return (
        metadata.get("version") == CACHE_VERSION
        and metadata.get("source_path") == str(csv_path)
        and metadata.get("source_size") == stat.st_size
        and metadata.get("source_mtime_ns") == stat.st_mtime_ns
        and int(metadata.get("max_rollout_steps", 0)) == int(max_rollout_steps)
        and metadata.get("layout", "full") == ("compact" if compact else "full")
    )


def _write_preprocessed_cache(csv_path, cache_dir, max_rollout_steps, compact=False):
    csv_path = Path(csv_path).expanduser().resolve()
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    header = pd.read_csv(csv_path, nrows=0)
    all_cols = header.columns.tolist()
    phys_cols = [col for col in PHYS_COLS if col in all_cols]
    abundance_cols = [
        col for col in all_cols if col not in EXCLUDED_COLS and col not in phys_cols
    ]
    usecols = ["Tracer", "Time"] + phys_cols + abundance_cols
    dtype = {col: np.float32 for col in phys_cols + abundance_cols}
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype)
    df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

    phys = df[phys_cols].to_numpy(dtype=np.float32, copy=True)
    abund = df[abundance_cols].to_numpy(dtype=np.float32, copy=True)
    abund = np.log10(np.maximum(abund, 1e-30)).astype(np.float32, copy=False)

    starts = []
    rollout_lengths = []
    sample_tracers = []
    for tracer_id, row_indices in df.groupby("Tracer", sort=False).indices.items():
        num_steps = len(row_indices)
        if num_steps <= 1:
            continue
        first_row = int(row_indices[0])
        offsets = np.arange(num_steps - 1, dtype=np.int64)
        starts.append(first_row + offsets)
        rollout_lengths.append(
            np.minimum(max_rollout_steps, num_steps - 1 - offsets).astype(
                np.int64, copy=False
            )
        )
        sample_tracers.append(np.full(len(offsets), tracer_id))

    sample_starts = (
        np.concatenate(starts).astype(np.int64, copy=False)
        if starts
        else np.empty(0, dtype=np.int64)
    )
    rollout_lengths = (
        np.concatenate(rollout_lengths).astype(np.int64, copy=False)
        if rollout_lengths
        else np.empty(0, dtype=np.int64)
    )
    sample_tracers = (
        np.concatenate(sample_tracers)
        if sample_tracers
        else np.empty(0, dtype=df["Tracer"].to_numpy().dtype)
    )

    np.save(tmp_dir / "phys.npy", phys)
    np.save(tmp_dir / "abund.npy", abund)
    np.save(tmp_dir / "sample_starts.npy", sample_starts)
    np.save(tmp_dir / "rollout_lengths.npy", rollout_lengths)
    np.save(tmp_dir / "sample_tracers.npy", sample_tracers)

    num_samples = len(sample_starts)
    horizon = int(max_rollout_steps)
    num_phys = len(phys_cols)
    num_targets = len(abundance_cols)
    if not compact:
        initial = np.lib.format.open_memmap(
            tmp_dir / "initial.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, num_phys + num_targets),
        )
        phys_seq = np.lib.format.open_memmap(
            tmp_dir / "phys_seq.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, horizon, num_phys),
        )
        target_seq = np.lib.format.open_memmap(
            tmp_dir / "target_seq.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, horizon, num_targets),
        )
        mask = np.lib.format.open_memmap(
            tmp_dir / "mask.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, horizon),
        )

        initial[:, :num_phys] = phys[sample_starts]
        initial[:, num_phys:] = abund[sample_starts]
        phys_seq[:] = 0.0
        target_seq[:] = 0.0
        mask[:] = 0.0
        for step in range(horizon):
            valid = rollout_lengths > step
            if not valid.any():
                break
            rows = sample_starts[valid] + step
            phys_seq[valid, step, :] = phys[rows]
            target_seq[valid, step, :] = abund[rows + 1]
            mask[valid, step] = 1.0

        for arr in (initial, phys_seq, target_seq, mask):
            arr.flush()
        del initial, phys_seq, target_seq, mask

    stat = csv_path.stat()
    metadata = {
        "version": CACHE_VERSION,
        "source_path": str(csv_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "max_rollout_steps": int(max_rollout_steps),
        "phys_cols": phys_cols,
        "abundance_cols": abundance_cols,
        "num_rows": int(len(df)),
        "num_samples": int(num_samples),
        "layout": "compact" if compact else "full",
    }
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    return metadata


def _migrate_full_cache_to_compact(csv_path, cache_dir, max_rollout_steps):
    full_dir = _cache_dir_for(csv_path, max_rollout_steps, compact=False)
    if not _cache_is_valid(full_dir, csv_path, max_rollout_steps, compact=False):
        return None
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "phys.npy",
        "abund.npy",
        "sample_starts.npy",
        "rollout_lengths.npy",
        "sample_tracers.npy",
    ):
        os.link(full_dir / name, tmp_dir / name)
    metadata = _read_cache_metadata(full_dir)
    metadata["layout"] = "compact"
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    tmp_dir.rename(cache_dir)
    return metadata


def _ensure_preprocessed_cache(
    csv_path, cache_dir, max_rollout_steps, compact=False
):
    if _cache_is_valid(cache_dir, csv_path, max_rollout_steps, compact=compact):
        return _read_cache_metadata(cache_dir)

    lock_dir = cache_dir.with_name(f"{cache_dir.name}.lock")
    start = time.time()
    owns_lock = False
    while not owns_lock:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            owns_lock = True
        except FileExistsError:
            if _cache_is_valid(
                cache_dir, csv_path, max_rollout_steps, compact=compact
            ):
                return _read_cache_metadata(cache_dir)
            if time.time() - start > 7200:
                raise TimeoutError(f"Timed out waiting for dataset cache lock: {lock_dir}")
            time.sleep(5)

    try:
        if not _cache_is_valid(
            cache_dir, csv_path, max_rollout_steps, compact=compact
        ):
            if compact:
                migrated = _migrate_full_cache_to_compact(
                    csv_path, cache_dir, max_rollout_steps
                )
                if migrated is not None:
                    return migrated
            print(
                f"Building preprocessed dataset cache: {cache_dir}",
                flush=True,
            )
            return _write_preprocessed_cache(
                csv_path,
                cache_dir,
                max_rollout_steps,
                compact=compact,
            )
        return _read_cache_metadata(cache_dir)
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


class GravCollapseDataset(Dataset):
    """
    Build one-step transition samples from a sampled split CSV.

    Input layout:
        [physical parameters at t, log10 abundances at t]

    Target layout:
        log10 abundances at t + 1
    """

    def __init__(
        self,
        csv_path,
        max_rollout_steps=1,
        use_preprocessed=True,
        sample_stride=1,
        sample_fraction=1.0,
        compact_batches=False,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        self.max_rollout_steps = max(1, int(max_rollout_steps))
        self.sample_stride = max(1, int(sample_stride))
        self.sample_fraction = float(sample_fraction)
        self.compact_batches = bool(compact_batches)
        self._cache_dir = _cache_dir_for(
            csv_path,
            self.max_rollout_steps,
            compact=self.compact_batches,
        )
        if use_preprocessed:
            try:
                self._load_or_create_preprocessed(csv_path)
                self._configure_sample_selection()
                return
            except (OSError, TimeoutError) as exc:
                print(
                    f"Preprocessed cache unavailable for {csv_path}: {exc}; "
                    "falling back to in-memory CSV dataset.",
                    flush=True,
                )

        self._load_csv_in_memory(csv_path)
        self._configure_sample_selection()

    def _configure_sample_selection(self):
        self._selected_indices = _select_sample_indices(
            self._sample_tracers,
            stride=self.sample_stride,
            fraction=self.sample_fraction,
        )

    def _load_or_create_preprocessed(self, csv_path):
        cache_dir = self._cache_dir
        metadata = _ensure_preprocessed_cache(
            csv_path,
            cache_dir,
            self.max_rollout_steps,
            compact=self.compact_batches,
        )

        self._phys_cols = metadata["phys_cols"]
        self._abundance_cols = metadata["abundance_cols"]
        self._feature_names = self._phys_cols + self._abundance_cols
        self._target_names = self._abundance_cols
        self._phys = np.load(cache_dir / "phys.npy", mmap_mode="r")
        self._abund = np.load(cache_dir / "abund.npy", mmap_mode="r")
        if not self.compact_batches:
            self._initial = np.load(cache_dir / "initial.npy", mmap_mode="c")
            self._phys_seq = np.load(cache_dir / "phys_seq.npy", mmap_mode="c")
            self._target_seq = np.load(cache_dir / "target_seq.npy", mmap_mode="c")
            self._mask = np.load(cache_dir / "mask.npy", mmap_mode="c")
        self._sample_starts = np.load(cache_dir / "sample_starts.npy", mmap_mode="r")
        self._rollout_lengths = np.load(
            cache_dir / "rollout_lengths.npy", mmap_mode="r"
        )
        self._sample_tracers = np.load(cache_dir / "sample_tracers.npy", mmap_mode="r")
        self._precomputed_samples = True

    def _load_csv_in_memory(self, csv_path):
        self._precomputed_samples = False

        header = pd.read_csv(csv_path, nrows=0)
        all_cols = header.columns.tolist()
        phys_cols = [col for col in PHYS_COLS if col in all_cols]
        abundance_cols = [
            col
            for col in all_cols
            if col not in EXCLUDED_COLS and col not in phys_cols
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
        return len(self._selected_indices)

    def __getitem__(self, idx):
        idx = int(self._selected_indices[idx])
        if self.compact_batches:
            return idx
        if self._precomputed_samples:
            return (
                torch.from_numpy(np.asarray(self._initial[idx])),
                torch.from_numpy(np.asarray(self._phys_seq[idx])),
                torch.from_numpy(np.asarray(self._target_seq[idx])),
                torch.from_numpy(np.asarray(self._mask[idx])),
            )

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
        return np.asarray(self._sample_tracers[self._selected_indices]).copy()


class GravCollapseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=str(config.DEFAULT_SPLIT_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        max_rollout_steps=config.ROLLOUT_STEPS,
        val_rollout_steps=None,
        train_sample_stride=1,
        val_fraction=1.0,
        compact_batches=False,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_rollout_steps = max(1, int(max_rollout_steps))
        self.val_rollout_steps = max(
            1,
            int(
                self.max_rollout_steps
                if val_rollout_steps is None
                else val_rollout_steps
            ),
        )
        self.train_sample_stride = max(1, int(train_sample_stride))
        self.val_fraction = float(val_fraction)
        self.compact_batches = bool(compact_batches)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = GravCollapseDataset(
                os.path.join(self.data_dir, "train.csv"),
                max_rollout_steps=self.max_rollout_steps,
                sample_stride=self.train_sample_stride,
                compact_batches=self.compact_batches,
            )
            self.val_ds = GravCollapseDataset(
                os.path.join(self.data_dir, "val.csv"),
                max_rollout_steps=self.val_rollout_steps,
                sample_fraction=self.val_fraction,
                compact_batches=self.compact_batches,
            )
        if stage in (None, "test", "predict"):
            self.test_ds = GravCollapseDataset(
                os.path.join(self.data_dir, "test.csv"),
                max_rollout_steps=self.max_rollout_steps,
                compact_batches=self.compact_batches,
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
            collate_fn=(
                CompactRolloutCollator(self.train_ds)
                if self.compact_batches
                else None
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=(
                CompactRolloutCollator(self.val_ds)
                if self.compact_batches
                else None
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=(
                CompactRolloutCollator(self.test_ds)
                if self.compact_batches
                else None
            ),
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
