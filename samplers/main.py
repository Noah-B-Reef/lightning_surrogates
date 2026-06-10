import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from samplers import (
    QR_sampler,
    density_sampler,
    random_sample,
    similarity_constrained_split,
    svd_fps,
)
from path_utils import DEFAULT_DATA_DIR
from util import load_bundle, plt_similarity

DEFAULT_N_SAMPLES = 6000
DEFAULT_MAX_SIMILARITY_TRACERS = 500
SAMPLERS = {
    "random": random_sample,
    "density": density_sampler,
    "qr_pivot": QR_sampler,
    "svd_fps": svd_fps,
    "similarity_constrained": similarity_constrained_split,
}
DEFAULT_SAMPLERS = ["random", "density", "qr_pivot", "svd_fps"]

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
parser.add_argument(
    "--max-similarity-tracers",
    type=int,
    default=DEFAULT_MAX_SIMILARITY_TRACERS,
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path(os.environ.get("SAMPLERS_DATA_DIR", DEFAULT_DATA_DIR)),
    help=(
        "Directory containing flattened_dataset.h5 and receiving sampled split "
        "outputs. Defaults to ../sampled_datasets relative to this script, or "
        "SAMPLERS_DATA_DIR when set."
    ),
)
parser.add_argument(
    "--bundle-path",
    type=Path,
    default=None,
    help="Optional explicit path to the flattened bundle. Defaults to DATA_DIR/flattened_dataset.h5.",
)
parser.add_argument(
    "--samplers",
    nargs="+",
    default=DEFAULT_SAMPLERS,
    choices=list(SAMPLERS),
    help="Sampler names to run. Defaults to random density qr_pivot svd_fps.",
)
parser.add_argument(
    "--storage-format",
    choices=("csv", "npy"),
    default="csv",
    help="Split storage format. Outputs go to DATA_DIR/{sampler}/{format}/.",
)
args = parser.parse_args()

data_dir = args.data_dir.expanduser().resolve()
bundle_path = (
    args.bundle_path.expanduser().resolve()
    if args.bundle_path is not None
    else data_dir / "flattened_dataset.h5"
)

# Load the flattened dataset
print(f"Loading flattened dataset from {bundle_path}...")
dataset_loaded = load_bundle(bundle_path)
print(f"Loaded {len(dataset_loaded['vectors'])} vectors")

for sampler_name in args.samplers:
    sampler = SAMPLERS[sampler_name]
    print(f"Sampling...({sampler_name.upper()})...")
    sampled = sampler(
        dataset_loaded,
        n_samples=args.n_samples,
        save_dir=data_dir,
        storage_format=args.storage_format,
    )

    train_df, val_df, test_df = sampled[:3]
    print(f"Sampled ({sampler_name.upper()})")
    print(f"Train Shape {train_df.shape}")
    print(f"Test Shape {test_df.shape}")
    print(f"Val Shape {val_df.shape}")

    output_dir = data_dir / sampler_name / args.storage_format
    similarity_plot_path = output_dir / "cosine_similarity.png"
    similarity_matrices_path = output_dir / "cosine_similarity_matrices.npz"

    fig, axes, matrices = plt_similarity(
        train_df,
        test_df,
        val_df,
        max_tracers=args.max_similarity_tracers,
        save_path=similarity_plot_path,
    )
    plt.close(fig)

    np.savez(
        similarity_matrices_path,
        train=matrices["Train"],
        validation=matrices["Validation"],
        test=matrices["Test"],
    )
    print(f"Saved cosine similarity plot to {similarity_plot_path}")
    print(f"Saved cosine similarity matrices to {similarity_matrices_path}")
