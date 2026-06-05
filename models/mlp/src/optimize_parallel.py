import argparse
import json
import os
import random
import time
from pathlib import Path

import optuna

import settings as config
from optimize import (
    TRAINING_PARAM_KEYS,
    build_optuna_storage,
    create_study_with_retry,
    objective,
    prepare_storage_for_resume,
    save_best_params,
    is_retryable_storage_error,
    is_sqlite_storage,
    sqlite_path_from_storage,
)


# Stamped on the study by rank 0 after a fresh reset so other ranks can tell the
# newly created study apart from the stale pre-reset one (see wait_for_fresh_study).
FRESH_STUDY_MARKER = "fresh_session_marker"


def parse_rank_context():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    node_id = os.environ.get("SLURM_NODEID", "0")
    return rank, world_size, node_id


def reset_storage(args, storage):
    sqlite_path = sqlite_path_from_storage(storage)
    if sqlite_path is not None:
        for path in (
            sqlite_path,
            Path(f"{sqlite_path}-journal"),
            Path(f"{sqlite_path}-wal"),
            Path(f"{sqlite_path}-shm"),
        ):
            if path.exists():
                path.unlink()
                print(f"Removed {path}", flush=True)
    else:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
            print(f"Deleted study '{args.study_name}' from storage.", flush=True)
        except KeyError:
            pass  # Study does not exist


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


def optimize_worker_trials_with_retry(study, args, split_dir, worker_trials, storage, rank):
    completed = 0
    consecutive_failures = 0
    max_consecutive_failures = 8

    while completed < worker_trials:
        try:
            study.optimize(lambda trial: objective(trial, args, split_dir), n_trials=1)
            completed += 1
            consecutive_failures = 0
        except Exception as exc:
            if not is_sqlite_storage(storage) or not is_retryable_storage_error(exc):
                raise
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                raise
            sleep_time = min(300.0, 5.0 * consecutive_failures) + random.uniform(0.0, 3.0)
            print(
                f"[rank {rank}] SQLite storage contention after "
                f"{completed}/{worker_trials} assigned trials; retrying in {sleep_time:.1f}s.",
                flush=True,
            )
            time.sleep(sleep_time)
            study = create_study_with_retry(args, storage)

    return study


def run_until_target(study, args, split_dir, target_finished_trials, storage, rank):
    """Run trials until the SHARED study reaches target_finished_trials.

    The stop condition is checked against the shared study before each trial, so
    workers never pre-divide a remaining budget from a per-rank snapshot of the
    finished count. That snapshot races: ranks start at different times and would
    disagree on how many trials remain, making the study over- or under-shoot the
    target. Here every rank simply keeps contributing trials until the global
    finished count reaches the target, so the total is correct regardless of
    start order. Overshoot is bounded by the number of trials in flight across
    all ranks (<= world_size).
    """
    while finished_trial_count(study) < target_finished_trials:
        study = optimize_worker_trials_with_retry(study, args, split_dir, 1, storage, rank)
    return study


def wait_for_fresh_study(args, storage, expected_marker, timeout_seconds, poll_seconds, rank):
    """Wait until rank 0 has reset and recreated the study for this run.

    In fresh mode, non-zero ranks must not create the study themselves: doing so
    races rank 0's reset (a worker could create/load the study before rank 0
    deletes it, or attach to the stale pre-reset study). Instead they only *load*
    the study and wait until it carries the marker rank 0 stamps on the freshly
    created study, so they can never attach to the old study or create a competing
    one.
    """
    wrapped_storage = build_optuna_storage(storage)
    deadline = time.time() + timeout_seconds
    while True:
        try:
            study = optuna.load_study(study_name=args.study_name, storage=wrapped_storage)
        except KeyError:
            study = None  # Rank 0 has not created the fresh study yet.
        if study is not None and study.user_attrs.get(FRESH_STUDY_MARKER) == expected_marker:
            return study
        if time.time() >= deadline:
            raise TimeoutError(
                f"[rank {rank}] timed out waiting for rank 0 to initialize a fresh study."
            )
        print(
            f"[rank {rank}] waiting for rank 0 to initialize a fresh study.",
            flush=True,
        )
        time.sleep(poll_seconds)


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
    parser.add_argument("--results-dir", type=Path, default=config.DEFAULT_OPTUNA_PARALLEL_RESULTS_DIR)
    parser.add_argument("--num-trials", type=int, default=config.OPTUNA_PARALLEL_N_TRIALS)
    parser.add_argument("--trials-per-worker", type=int, default=None)
    parser.add_argument("--tune-epochs", type=int, default=config.OPTUNA_PARALLEL_TUNE_EPOCHS)
    parser.add_argument("--study-name", type=str, default=config.OPTUNA_PARALLEL_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--rollout-steps", type=int, default=config.ROLLOUT_STEPS)
    parser.add_argument("--accelerator", type=str, default=config.ACCELERATOR)
    parser.add_argument("--devices", default=config.NUM_DEVICES)
    parser.add_argument("--precision", default=config.PRECISION)
    parser.add_argument("--patience", type=int, default=config.OPTUNA_PRUNER_PATIENCE)
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=config.OPTUNA_MIN_RELATIVE_IMPROVEMENT,
    )
    parser.add_argument("--finalize-timeout", type=int, default=3600)
    parser.add_argument("--finalize-poll-seconds", type=int, default=30)
    parser.add_argument("--allow-sqlite", action="store_true")
    parser.add_argument(
        "--journal-mode",
        choices=("resume", "fresh"),
        default=config.OPTUNA_JOURNAL_MODE,
        help=(
            "resume: reuse the existing Optuna journal and run only enough trials "
            "to reach --num-trials finished trials. fresh: remove/reset the "
            "Optuna storage and start a new study."
        ),
    )
    args = parser.parse_args()

    configured_storage = config.OPTUNA_PARALLEL_STORAGE or None
    args.results_dir = Path(args.results_dir).expanduser().resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or configured_storage or f"sqlite:///{args.results_dir / 'optuna.sqlite3'}"

    rank, world_size, node_id = parse_rank_context()
    if world_size > 1 and is_sqlite_storage(storage) and not args.allow_sqlite:
        raise ValueError(
            "Parallel Optuna across Slurm ranks requires server-backed storage "
            "such as PostgreSQL or MySQL. SQLite is not safe for multi-node workers. "
            "Pass --allow-sqlite only if you intentionally accept that risk."
        )

    dataset_path = args.data_dir or args.dataset_path
    split_dir = config.resolve_split_dir(dataset_path)

    # Shared across all ranks of the same Slurm job, so every rank agrees on the
    # marker identifying this run's fresh study. Falls back to a constant when no
    # job id is present (single-node/local), which is still safe because the old
    # study is deleted before the marker is stamped.
    fresh_marker = os.environ.get("SLURM_JOB_ID") or os.environ.get("PMI_JOBID") or "fresh"
    if args.journal_mode == "fresh":
        if rank == 0:
            reset_storage(args, storage)
            prepare_storage_for_resume(storage)
            study = create_study_with_retry(args, storage)
            study.set_user_attr(FRESH_STUDY_MARKER, fresh_marker)
        else:
            study = wait_for_fresh_study(
                args,
                storage,
                fresh_marker,
                timeout_seconds=args.finalize_timeout,
                poll_seconds=args.finalize_poll_seconds,
                rank=rank,
            )
    else:
        if rank == 0:
            prepare_storage_for_resume(storage)
        study = create_study_with_retry(args, storage)

    initial_finished = 0 if args.journal_mode == "fresh" else finished_trial_count(study)
    if args.trials_per_worker is None:
        # Target-total mode (matches serial optimize.py): --num-trials is the
        # desired number of finished trials in the SHARED study. Every rank keeps
        # running new trials until that global count is reached, instead of
        # pre-dividing a remaining budget from a per-rank snapshot of the finished
        # count (which races because ranks start at different times).
        target_finished_trials = max(args.num_trials, initial_finished)
        print(
            f"[rank {rank}/{world_size} node {node_id}] study={args.study_name} "
            f"mode=target-total initial_finished_trials={initial_finished} "
            f"target_finished_trials={target_finished_trials} storage={storage}",
            flush=True,
        )
        study = run_until_target(study, args, split_dir, target_finished_trials, storage, rank)
    else:
        # Explicit additive mode: each rank runs exactly trials_per_worker new
        # trials. The per-rank counts are independent, so there is no shared-count
        # race; the total added is trials_per_worker * world_size.
        worker_trials = args.trials_per_worker
        target_finished_trials = initial_finished + worker_trials * world_size
        print(
            f"[rank {rank}/{world_size} node {node_id}] study={args.study_name} "
            f"mode=additive worker_trials={worker_trials} "
            f"initial_finished_trials={initial_finished} "
            f"target_finished_trials={target_finished_trials} storage={storage}",
            flush=True,
        )
        if worker_trials > 0:
            study = optimize_worker_trials_with_retry(
                study,
                args,
                split_dir,
                worker_trials,
                storage,
                rank,
            )
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
        "storage": storage,
        "world_size": world_size,
        "target_trials": target_finished_trials,
        "initial_finished_trials": initial_finished,
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
