"""Run the sampler-agnostic MLP benchmark on a chosen split directory.

This script intentionally does not run samplers. Generate candidate split
folders elsewhere, choose one, then pass its path here. The split directory must
contain train.csv, val.csv, and test.csv. With no path it defaults to the
best-sampler split exported by the samplers benchmark.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

import config


def run_step(step_num, name, command, skip=False):
    if skip:
        print(f"--- Skipping Step {step_num}: {name} ---")
        return
    print(f"\n=== Step {step_num}: {name} ===", flush=True)
    start = time.time()
    subprocess.run(command, check=True)
    elapsed = int(time.time() - start)
    print(f"=== Step {step_num} complete ({elapsed}s) ===", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna, final training, and rollout testing for an MLP split dataset."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Split directory with train/val/test.csv (default: best-sampler split).",
    )
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--optuna-results-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default="mlp_grav_collapse.ckpt")
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    split_dir = config.resolve_split_dir(args.dataset_path)
    results_dir = args.results_dir.expanduser().resolve()
    optuna_dir = (
        args.optuna_results_dir.expanduser().resolve()
        if args.optuna_results_dir is not None
        else results_dir / "optimization"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    run_step(1, "Optimize MLP hyperparameters", [
        python, str(SCRIPT_DIR / "src" / "optimize.py"), str(split_dir),
        "--results-dir", str(optuna_dir),
        "--num-trials", str(args.num_trials),
        "--tune-epochs", str(args.tune_epochs),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--num-workers", str(args.num_workers),
    ], skip=args.skip_optimize)

    run_step(2, "Train final MLP", [
        python, str(SCRIPT_DIR / "src" / "train.py"), str(split_dir),
        "--results-dir", str(results_dir),
        "--config-file", str(optuna_dir / "best_params.json"),
        "--checkpoint", args.checkpoint,
        "--epochs", str(args.train_epochs),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--num-workers", str(args.num_workers),
    ], skip=args.skip_train)

    run_step(3, "Autoregressive rollout test", [
        python, str(SCRIPT_DIR / "src" / "test.py"), str(split_dir),
        "--model-checkpoint", str(results_dir / args.checkpoint),
        "--output-dir", str(results_dir / "test_results"),
    ], skip=args.skip_test)

    print("\n=== MLP benchmark complete ===")


if __name__ == "__main__":
    main()
