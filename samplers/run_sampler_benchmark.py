import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from path_utils import default_data_dir, default_results_dir


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


def get_best_sampler(benchmark_json):
    with Path(benchmark_json).open() as f:
        benchmark = json.load(f)

    if benchmark.get("best_sampler"):
        return benchmark["best_sampler"]

    ranking = benchmark.get("ranking", [])
    if ranking:
        return ranking[0]["sampler"]

    raise RuntimeError(f"No successful sampler ranking found in {benchmark_json}")


def export_best_sampler(data_dir, best_sampler, best_sampler_dir):
    source_name = "qr_pivot" if best_sampler == "qr" else best_sampler
    # Splits live in {sampler}/{storage_format}/ (csv preferred, then npy).
    source_dir = None
    required_files = None
    for storage_format in ("csv", "npy"):
        candidate = data_dir / source_name / storage_format
        names = [f"{split}.{storage_format}" for split in ("train", "val", "test")]
        if storage_format == "npy":
            names.append("columns.json")
        if all((candidate / name).is_file() for name in names):
            source_dir = candidate
            required_files = names
            break
    if source_dir is None:
        raise FileNotFoundError(
            f"Best sampler dataset is incomplete under {data_dir / source_name}; "
            "expected complete csv/ or npy/ split directory"
        )

    if best_sampler_dir.exists():
        shutil.rmtree(best_sampler_dir)
    best_sampler_dir.mkdir(parents=True)

    for name in required_files:
        shutil.copy2(source_dir / name, best_sampler_dir / name)

    qr_dir = source_dir / "qr_samples"
    if qr_dir.is_dir():
        shutil.copytree(qr_dir, best_sampler_dir / "qr_samples")

    (best_sampler_dir / "best_sampler.txt").write_text(f"{best_sampler}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run sampler similarity benchmarking and export the best-ranked "
            "split dataset to a best_sampler directory."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing flattened_dataset.h5 and sampler split folders.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--best-sampler-dir",
        type=Path,
        default=None,
        help="Directory receiving train/val/test.csv for the best sampler.",
    )
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["random", "density", "qr_pivot", "svd_fps"],
        choices=["random", "density", "qr_pivot", "qr", "svd_fps", "similarity_constrained"],
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.95, 0.99, 0.999, 0.9999],
    )
    parser.add_argument("--primary-threshold", type=float, default=0.9999)
    parser.add_argument("--max-reference", type=int, default=500)
    parser.add_argument("--max-candidate", type=int, default=1000)
    parser.add_argument("--max-pairwise", type=int, default=500)
    parser.add_argument("--force-benchmark", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    results_dir = args.results_dir if args.results_dir is not None else default_results_dir()
    best_sampler_dir = (
        args.best_sampler_dir if args.best_sampler_dir is not None else data_dir / "best_sampler"
    )
    bundle_path = data_dir / "flattened_dataset.h5"
    benchmark_json = results_dir / "sampler_benchmark" / "sampler_benchmark_results.json"

    if args.force_benchmark or not benchmark_json.is_file():
        run_command(
            [
                sys.executable,
                str(script_dir / "benchmark_samplers.py"),
                "--bundle-path",
                str(bundle_path),
                "--results-dir",
                str(results_dir),
                "--n-samples",
                str(args.n_samples),
                "--random-state",
                str(args.random_state),
                "--samplers",
                *args.samplers,
                "--thresholds",
                *[str(threshold) for threshold in args.thresholds],
                "--primary-threshold",
                str(args.primary_threshold),
                "--max-reference",
                str(args.max_reference),
                "--max-candidate",
                str(args.max_candidate),
                "--max-pairwise",
                str(args.max_pairwise),
            ],
            "Benchmarking Split Similarity and Memorization Risk",
        )
    else:
        print(f"Similarity benchmark results already exist at: {benchmark_json}.")

    best_sampler = get_best_sampler(benchmark_json)
    export_best_sampler(data_dir, best_sampler, best_sampler_dir)

    print("\n=========================================================================")
    print("                  SAMPLER BENCHMARK COMPLETE")
    print("=========================================================================")
    print(f"Best Sampler Selected: {best_sampler}")
    print(f"Benchmark results: {benchmark_json}")
    print(f"Best sampler dataset: {best_sampler_dir}")
    print(f"Best sampler marker: {best_sampler_dir / 'best_sampler.txt'}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
