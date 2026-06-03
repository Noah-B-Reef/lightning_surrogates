import argparse
import json
import os
import time
from pathlib import Path

import optuna

import config
from optimize import TRAINING_PARAM_KEYS, objective, save_best_params


def parse_rank_context():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    node_id = os.environ.get("SLURM_NODEID", "0")
    return rank, world_size, node_id


def is_sqlite_storage(storage):
    return storage.startswith("sqlite:")


def assigned_trials(total_trials, world_size, rank):
    base = total_trials // world_size
    remainder = total_trials % world_size
    return base + (1 if rank < remainder else 0)


def create_or_load_study(args):
    return optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )


def finished_trial_count(study):
    return sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())


def wait_for_trials(study, target_trials, timeout_seconds, poll_seconds):
    deadline = time.time() + timeout_seconds
    while True:
        count = finished_trial_count(study)
        if count >= target_trials:
            return count
        if time.time() >= deadline:
            raise TimeoutError(
                f"Only {count}/{target_trials} trials finished before finalize timeout."
            )
        print(f"[rank 0] waiting for workers: finished_trials={count}/{target_trials}", flush=True)
        time.sleep(poll_seconds)


def filtered_best_params(study):
    raw_best_params = study.best_trial.user_attrs.get("params_for_training", dict(study.best_params))
    return {key: raw_best_params[key] for key in TRAINING_PARAM_KEYS}


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna MLP optimization with multiple Slurm worker ranks."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Split directory with train/val/test.csv (default: best-sampler split).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Alias for dataset_path.")
    parser.add_argument("--results-dir", type=Path, default=config.DEFAULT_RESULTS_DIR / "optimization")
    parser.add_argument("--num-trials", type=int, default=config.OPTUNA_N_TRIALS)
    parser.add_argument("--trials-per-worker", type=int, default=None)
    parser.add_argument("--tune-epochs", type=int, default=config.OPTUNA_TUNE_EPOCHS)
    parser.add_argument("--study-name", type=str, default=f"{config.OPTUNA_STUDY_NAME}_parallel")
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--accelerator", type=str, default=config.ACCELERATOR)
    parser.add_argument("--devices", default=config.NUM_DEVICES)
    parser.add_argument("--precision", default=config.PRECISION)
    parser.add_argument("--patience", type=int, default=config.OPTUNA_PRUNER_PATIENCE)
    parser.add_argument("--min-relative-improvement", type=float, default=0.02)
    parser.add_argument("--finalize-timeout", type=int, default=3600)
    parser.add_argument("--finalize-poll-seconds", type=int, default=30)
    parser.add_argument("--allow-sqlite", action="store_true")
    args = parser.parse_args()

    rank, world_size, node_id = parse_rank_context()
    if world_size > 1 and is_sqlite_storage(args.storage) and not args.allow_sqlite:
        raise ValueError(
            "Parallel Optuna across Slurm ranks requires server-backed storage "
            "such as PostgreSQL or MySQL. SQLite is not safe for multi-node workers."
        )

    dataset_path = args.data_dir or args.dataset_path
    split_dir = config.resolve_split_dir(dataset_path)
    args.results_dir = Path(args.results_dir).expanduser().resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    study = create_or_load_study(args)
    initial_finished = finished_trial_count(study)
    if args.trials_per_worker is None:
        worker_trials = assigned_trials(args.num_trials, world_size, rank)
        target_new_trials = args.num_trials
    else:
        worker_trials = args.trials_per_worker
        target_new_trials = args.trials_per_worker * world_size
    target_finished_trials = initial_finished + target_new_trials
    print(
        f"[rank {rank}/{world_size} node {node_id}] study={args.study_name} "
        f"worker_trials={worker_trials} target_new_trials={target_new_trials} "
        f"initial_finished_trials={initial_finished} "
        f"storage={args.storage}",
        flush=True,
    )

    if worker_trials > 0:
        study.optimize(lambda trial: objective(trial, args, split_dir), n_trials=worker_trials)
    else:
        print(f"[rank {rank}] no assigned trials; exiting worker loop.", flush=True)

    if rank != 0:
        return

    finished = wait_for_trials(
        study,
        target_trials=target_finished_trials,
        timeout_seconds=args.finalize_timeout,
        poll_seconds=args.finalize_poll_seconds,
    )
    best_params = filtered_best_params(study)
    json_path, txt_path = save_best_params(best_params, args.results_dir, study.best_value)
    summary = {
        "split_dir": str(split_dir),
        "study_name": args.study_name,
        "storage": args.storage,
        "world_size": world_size,
        "target_new_trials": target_new_trials,
        "finished_trials": finished,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": best_params,
    }
    (args.results_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[rank 0] Best value: {study.best_value:.6g}", flush=True)
    print(f"[rank 0] Wrote {json_path}", flush=True)
    print(f"[rank 0] Wrote {txt_path}", flush=True)


if __name__ == "__main__":
    main()
