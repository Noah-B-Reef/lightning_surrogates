import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

PHYS_COLS = ("Density", "gasTemp", "dustTemp", "Av", "radfield", "zeta")


def dataset_flatten(dataset):
    """
    Flatten (Tracer, Time, features) DataFrame into a vector bundle.

    Only abundance columns are log10-transformed (physical params stay linear) so the
    bundle round-trips cleanly back to the scale GravCollapseDataset expects.

    Returns dict with keys: vectors, tracer_ids, feature_cols, log_cols,
    time_grid, T.
    """
    dataset = dataset.drop(columns=["dstep"], errors="ignore")
    dataset = dataset.sort_values(by=["Tracer", "Time"]).reset_index(drop=True)
    EPS = 1e-30

    feature_cols = [
        c for c in dataset.columns if c not in ("Tracer", "Time", "BULK", "SURFACE")
    ]
    abundance_cols = [c for c in feature_cols if c not in PHYS_COLS]

    # log10 only abundances (GravCollapseDataset re-logs abundances on load)
    dataset[abundance_cols] = np.log10(np.maximum(dataset[abundance_cols], EPS))

    features = dataset[feature_cols].values
    tracer_ids = dataset["Tracer"].values
    times = dataset["Time"].values

    uniq_ids, start_idx, counts = np.unique(
        tracer_ids, return_index=True, return_counts=True
    )
    T = counts[0]
    assert np.all(counts == T), "Tracers have varying timestep counts; cannot reshape."

    n_features = features.shape[1]
    vectors = features.reshape(len(uniq_ids), T * n_features)
    time_grid = times[start_idx[0] : start_idx[0] + T]

    return {
        "vectors": vectors,
        "tracer_ids": uniq_ids,
        "feature_cols": np.array(feature_cols),
        "log_cols": np.array(abundance_cols),
        "time_grid": time_grid,
        "T": np.int64(T),
    }


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Passing a string path to np.savez appends ".npz" when the suffix is not
    # already .npz. Use a file handle so callers get the exact path requested.
    with path.open("wb") as f:
        np.savez(f, **bundle)


def load_bundle(path):
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".npz").exists():
        path = path.with_suffix(path.suffix + ".npz")

    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def reformat(bundle, tracer_subset=None):
    """
    Inverse of dataset_flatten. Reconstructs a long-format DataFrame
    (Tracer, Time, feature_cols...) on the original (un-logged) scale.
    """
    V = bundle["vectors"]
    ids = bundle["tracer_ids"]
    cols = list(bundle["feature_cols"])
    time_grid = bundle["time_grid"]
    T = int(bundle["T"])

    if tracer_subset is not None:
        if len(tracer_subset) == 0:
            return pd.DataFrame(columns=["Tracer", "Time"] + cols)
        mask = np.isin(ids, tracer_subset)
        V = V[mask]
        ids = ids[mask]

    n_feat = len(cols)
    df = pd.DataFrame(V.reshape(-1, n_feat), columns=cols)
    df.insert(0, "Time", np.tile(time_grid, len(ids)))
    df.insert(0, "Tracer", np.repeat(ids, T))

    log_cols = [c for c in bundle["log_cols"].tolist() if c in df.columns]
    if log_cols:
        df[log_cols] = np.power(10.0, df[log_cols].values)
    return df


def dataframe_splits(
    bundle,
    train_tracers,
    val_tracers,
    test_tracers,
    save_dir=None,
    sampling_procedure=None,
    storage_format="csv",
):
    train_df = reformat(bundle, tracer_subset=train_tracers)
    val_df = reformat(bundle, tracer_subset=val_tracers)
    test_df = reformat(bundle, tracer_subset=test_tracers)

    output_dir = None
    if save_dir is not None:
        output_dir = datasets_save(
            train_df, test_df, val_df, save_dir, sampling_procedure, storage_format
        )

    return train_df, val_df, test_df, output_dir


def _tracer_vectors_from_df(df, feature_cols=None, max_tracers=None, random_state=42):
    """
    Convert a long-format split DataFrame into one flattened trajectory vector
    per tracer.
    """
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c not in ("Tracer", "Time", "dstep", "BULK", "SURFACE")
        ]

    if df.empty:
        return np.empty((0, len(feature_cols))), feature_cols

    sorted_df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)
    tracer_ids, counts = np.unique(sorted_df["Tracer"].values, return_counts=True)
    if not np.all(counts == counts[0]):
        raise ValueError("All tracers must have the same number of timesteps.")

    if max_tracers is not None and len(tracer_ids) > max_tracers:
        rng = np.random.default_rng(random_state)
        tracer_ids = rng.choice(tracer_ids, size=max_tracers, replace=False)
        sorted_df = sorted_df[sorted_df["Tracer"].isin(tracer_ids)]
        sorted_df = sorted_df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

    values = sorted_df[feature_cols].to_numpy()
    vectors = values.reshape(sorted_df["Tracer"].nunique(), -1)
    return vectors, feature_cols


def plt_similarity(
    train_df,
    test_df,
    val_df,
    *,
    max_tracers=None,
    random_state=42,
    cmap="viridis",
    save_path=None,
):
    """
    Plot cosine similarity matrices for train, validation, and test splits.

    The input DataFrames should be in long format with columns:
    Tracer, Time, feature columns...

    Returns:
        fig, axes, matrices
    """
    train_vectors, feature_cols = _tracer_vectors_from_df(
        train_df, max_tracers=max_tracers, random_state=random_state
    )
    val_vectors, _ = _tracer_vectors_from_df(
        val_df,
        feature_cols=feature_cols,
        max_tracers=max_tracers,
        random_state=random_state,
    )
    test_vectors, _ = _tracer_vectors_from_df(
        test_df,
        feature_cols=feature_cols,
        max_tracers=max_tracers,
        random_state=random_state,
    )

    splits = {
        "Train": train_vectors,
        "Validation": val_vectors,
        "Test": test_vectors,
    }
    matrices = {
        name: cosine_similarity(vectors) if len(vectors) else np.empty((0, 0))
        for name, vectors in splits.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Cosine Similarity by Dataset Split", fontsize=16, fontweight="bold")

    for ax, (name, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0)
        ax.set_title(f"{name} Cosine Similarity")
        ax.set_xlabel("Tracer Index")
        ax.set_ylabel("Tracer Index")
        fig.colorbar(image, ax=ax, label="Similarity")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, matrices


def datasets_save(train_df, test_df, val_df, path, sampling_procedure, storage_format="csv"):
    """
    Save train/test/validation splits under {path}/{sampling_procedure}/{storage_format}/.

    storage_format:
        "csv": one CSV per split with a header row.
        "npy": one float64 .npy per split (row-major, long format) plus a
               columns.json listing the column order shared by all splits.

    Example:
        datasets_save(train, test, val, "sampled_dataset", "density", "npy")

    writes:
        sampled_dataset/density/npy/train.npy
        sampled_dataset/density/npy/val.npy
        sampled_dataset/density/npy/test.npy
        sampled_dataset/density/npy/columns.json

    Returns:
        Path to the {sampling_procedure}/{storage_format} directory.
    """
    storage_format = str(storage_format).lower()
    if storage_format not in ("csv", "npy"):
        raise ValueError(f"storage_format must be 'csv' or 'npy', got {storage_format!r}")

    output_dir = Path(path) / sampling_procedure / storage_format
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    if storage_format == "csv":
        for name, df in splits.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)
    else:
        columns = list(train_df.columns)
        for name, df in splits.items():
            if list(df.columns) != columns:
                raise ValueError(f"Split '{name}' columns do not match train columns.")
            np.save(output_dir / f"{name}.npy", df.to_numpy(dtype=np.float64))
        (output_dir / "columns.json").write_text(json.dumps(columns, indent=2))

    return output_dir
