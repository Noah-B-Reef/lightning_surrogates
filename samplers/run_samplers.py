import argparse
import subprocess
import sys
import time
from pathlib import Path

from path_utils import default_data_dir, default_raw_h5


def run_command(command, step_name):
    print(f"\n--- Starting Step: {step_name} ---")
    print(f"Command: {' '.join(command)}")
    start_time = time.time()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error in step '{step_name}': command exited with non-zero status.")
        sys.exit(exc.returncode)
    elapsed = time.time() - start_time
    print(f"--- Finished Step: {step_name} successfully in {elapsed:.2f}s ---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten the raw dataset if needed, then generate sampler split datasets."
    )
    parser.add_argument(
        "--raw-h5",
        type=Path,
        default=None,
        help="Path to raw chemistry H5 file. Defaults to SAMPLERS_RAW_H5.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing flattened_dataset.h5 and sampler split folders.",
    )
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument(
        "--storage-format",
        choices=("csv", "npy"),
        default="csv",
        help="Split storage format; outputs go to DATA_DIR/{sampler}/{format}/.",
    )
    parser.add_argument("--max-similarity-tracers", type=int, default=500)
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["random", "density", "qr_pivot", "svd_fps"],
        choices=["random", "density", "qr_pivot", "svd_fps", "similarity_constrained"],
        help="Sampler names to run.",
    )
    parser.add_argument("--force-flatten", action="store_true")
    parser.add_argument("--force-sample", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    raw_h5 = args.raw_h5 if args.raw_h5 is not None else default_raw_h5()
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = data_dir / "flattened_dataset.h5"

    if args.force_flatten or not bundle_path.is_file():
        run_command(
            [
                sys.executable,
                str(script_dir / "flatten_dataset.py"),
                "--input-h5",
                str(raw_h5),
                "--output-path",
                str(bundle_path),
            ],
            "Flattening Raw HDF5 Dataset",
        )
    else:
        print(f"Flattened bundle already exists at: {bundle_path}. Skipping flatten step.")

    needs_sampling = args.force_sample
    if not needs_sampling:
        for sampler in args.samplers:
            split_file = f"train.{args.storage_format}"
            if not (data_dir / sampler / args.storage_format / split_file).is_file():
                needs_sampling = True
                break

    if needs_sampling:
        run_command(
            [
                sys.executable,
                str(script_dir / "main.py"),
                "--data-dir",
                str(data_dir),
                "--bundle-path",
                str(bundle_path),
                "--n-samples",
                str(args.n_samples),
                "--max-similarity-tracers",
                str(args.max_similarity_tracers),
                "--samplers",
                *args.samplers,
                "--storage-format",
                args.storage_format,
            ],
            "Generating Sampler Split Datasets",
        )
    else:
        print("Sampler split datasets already exist. Skipping sampling step.")

    print(f"Sampler datasets are available in: {data_dir}")


if __name__ == "__main__":
    main()
