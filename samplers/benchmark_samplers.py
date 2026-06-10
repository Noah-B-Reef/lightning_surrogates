import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity

import samplers
from path_utils import DEFAULT_BUNDLE_PATH, default_results_dir, resolve_path
from util import load_bundle


SAMPLER_REGISTRY = {
    "random": samplers.random_sample,
    "density": samplers.density_sampler,
    "qr_pivot": samplers.QR_sampler,
    "qr": samplers.QR_sampler,
    "svd_fps": samplers.svd_fps,
    "similarity_constrained": samplers.similarity_constrained_split,
}

DEFAULT_SAMPLERS = ["random", "density", "qr_pivot", "svd_fps"]


def _sample_indices(indices, max_n, seed):
    indices = np.asarray(indices)
    if max_n is None or len(indices) <= max_n:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_n, replace=False)


def _id_to_indices(all_ids, tracer_ids):
    tracer_ids = np.asarray(tracer_ids)
    positions = np.searchsorted(all_ids, tracer_ids)
    if not np.all(all_ids[positions] == tracer_ids):
        missing = tracer_ids[all_ids[positions] != tracer_ids][:5]
        raise ValueError(f"Sampler returned tracer IDs not in bundle: {missing}")
    return positions


def _centered_arrays(vectors, left_idx, right_idx, center_idx=None):
    if center_idx is None:
        center_idx = right_idx
    mean = np.mean(vectors[center_idx], axis=0)
    return vectors[left_idx] - mean, vectors[right_idx] - mean


def nearest_similarity_stats(
    vectors,
    reference_idx,
    candidate_idx,
    thresholds,
    *,
    centered=True,
    center_idx=None,
    max_reference=500,
    max_candidate=1000,
    seed=42,
):
    ref = _sample_indices(reference_idx, max_reference, seed)
    cand = _sample_indices(candidate_idx, max_candidate, seed + 1)
    if len(ref) == 0 or len(cand) == 0:
        return {"reference_n": int(len(ref)), "candidate_n": int(len(cand))}

    if centered:
        left, right = _centered_arrays(vectors, ref, cand, center_idx=center_idx)
    else:
        left, right = vectors[ref], vectors[cand]

    sims = cosine_similarity(left, right)
    nearest = np.max(sims, axis=1)
    distances = 1.0 - nearest

    stats = {
        "reference_n": int(len(ref)),
        "candidate_n": int(len(cand)),
        "mean_nearest_similarity": float(np.mean(nearest)),
        "median_nearest_similarity": float(np.median(nearest)),
        "p05_nearest_similarity": float(np.quantile(nearest, 0.05)),
        "p95_nearest_similarity": float(np.quantile(nearest, 0.95)),
        "max_nearest_similarity": float(np.max(nearest)),
        "mean_nearest_distance": float(np.mean(distances)),
        "p95_nearest_distance": float(np.quantile(distances, 0.95)),
    }
    stats["thresholds"] = {
        str(threshold): {
            "fraction_at_or_above": float(np.mean(nearest >= threshold)),
            "count_at_or_above": int(np.sum(nearest >= threshold)),
        }
        for threshold in thresholds
    }
    return stats


def pairwise_similarity_stats(
    vectors,
    indices,
    thresholds,
    *,
    centered=True,
    max_n=500,
    seed=42,
):
    sample = _sample_indices(indices, max_n, seed)
    if len(sample) < 2:
        return {"n": int(len(sample))}

    arr = vectors[sample]
    if centered:
        arr = arr - np.mean(vectors[indices], axis=0)
    sims = cosine_similarity(arr)
    offdiag_mask = ~np.eye(len(sample), dtype=bool)
    offdiag = sims[offdiag_mask]
    no_diag = sims.copy()
    np.fill_diagonal(no_diag, -np.inf)
    nearest = np.max(no_diag, axis=1)

    return {
        "n": int(len(sample)),
        "mean_offdiag_similarity": float(np.mean(offdiag)),
        "p95_offdiag_similarity": float(np.quantile(offdiag, 0.95)),
        "max_offdiag_similarity": float(np.max(offdiag)),
        "mean_nearest_similarity": float(np.mean(nearest)),
        "p95_nearest_similarity": float(np.quantile(nearest, 0.95)),
        "thresholds": {
            str(threshold): {
                "fraction_pairs_at_or_above": float(np.mean(offdiag >= threshold)),
                "fraction_nearest_at_or_above": float(np.mean(nearest >= threshold)),
            }
            for threshold in thresholds
        },
    }


def final_density_ks(bundle, split_idx):
    feature_cols = list(bundle["feature_cols"])
    density_idx = feature_cols.index("Density")
    t_steps = int(bundle["T"])
    final_density_idx = (t_steps - 1) * len(feature_cols) + density_idx
    final_density = bundle["vectors"][:, final_density_idx]
    return float(ks_2samp(final_density[split_idx], final_density).statistic)


def capture_sampler_split(bundle, sampler_name, sampler_fn, n_samples, random_state):
    captured = {}
    original_dataframe_splits = samplers.dataframe_splits

    def capture(bundle_arg, train_tracers, val_tracers, test_tracers, save_dir=None, sampling_procedure=None, storage_format="csv"):
        captured["train_tracers"] = np.asarray(train_tracers)
        captured["val_tracers"] = np.asarray(val_tracers)
        captured["test_tracers"] = np.asarray(test_tracers)
        return None, None, None, None

    samplers.dataframe_splits = capture
    try:
        kwargs = {"save_qr": False} if sampler_name in {"qr", "qr_pivot"} else {}
        if sampler_name == "qr":
            kwargs["sampling_procedure"] = "qr"
        sampler_fn(
            bundle,
            n_samples=n_samples,
            random_state=random_state,
            save_dir=None,
            **kwargs,
        )
    finally:
        samplers.dataframe_splits = original_dataframe_splits

    if not captured:
        raise RuntimeError(f"{sampler_name} did not produce a split")
    return captured


def score_results(results, primary_threshold):
    ok_results = [r for r in results if r["status"] == "ok"]
    score_rows = []
    for result in ok_results:
        threshold_key = str(primary_threshold)
        val_dup = result["cross_split_similarity"]["val_to_train"]["thresholds"][
            threshold_key
        ]["fraction_at_or_above"]
        test_dup = result["cross_split_similarity"]["test_to_train"]["thresholds"][
            threshold_key
        ]["fraction_at_or_above"]
        score_rows.append(
            {
                "sampler": result["sampler"],
                "near_duplicate_rate": (val_dup + test_dup) / 2.0,
                "coverage_distance": result["train_coverage_vs_full"][
                    "mean_nearest_distance"
                ],
                "density_ks_train": result["final_density_ks_vs_full"]["train"],
                "train_nearest_similarity": result["within_split_similarity"]["train"][
                    "mean_nearest_similarity"
                ],
                "seconds": result["seconds"],
            }
        )

    rank_specs = [
        ("near_duplicate_rate", 3.0),
        ("coverage_distance", 1.5),
        ("density_ks_train", 1.5),
        ("train_nearest_similarity", 1.0),
        ("seconds", 0.25),
    ]
    for metric, _weight in rank_specs:
        ordered = sorted((row[metric], row["sampler"]) for row in score_rows)
        ranks = {sampler: rank + 1 for rank, (_value, sampler) in enumerate(ordered)}
        for row in score_rows:
            row[f"{metric}_rank"] = ranks[row["sampler"]]

    for row in score_rows:
        row["composite_score"] = sum(
            row[f"{metric}_rank"] * weight for metric, weight in rank_specs
        )

    return sorted(score_rows, key=lambda row: row["composite_score"])


def write_ranking_csv(path, ranking):
    fieldnames = [
        "sampler",
        "composite_score",
        "near_duplicate_rate",
        "coverage_distance",
        "density_ks_train",
        "train_nearest_similarity",
        "seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow({key: row[key] for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sampler representativeness and memorization risk."
    )
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=DEFAULT_SAMPLERS,
        choices=list(SAMPLER_REGISTRY),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.95, 0.99, 0.999, 0.9999],
        help="Cosine similarity thresholds for near-duplicate accounting.",
    )
    parser.add_argument("--primary-threshold", type=float, default=0.9999)
    parser.add_argument("--max-reference", type=int, default=500)
    parser.add_argument("--max-candidate", type=int, default=1000)
    parser.add_argument("--max-pairwise", type=int, default=500)
    args = parser.parse_args()

    bundle_path = resolve_path(args.bundle_path)
    results_dir = resolve_path(args.results_dir) / "sampler_benchmark"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading flattened bundle: {bundle_path}")
    bundle = load_bundle(bundle_path)
    vectors = bundle["vectors"].astype(np.float32, copy=False)
    tracer_ids = bundle["tracer_ids"]
    all_indices = np.arange(vectors.shape[0])

    results = []
    for sampler_name in args.samplers:
        print(f"\nBenchmarking {sampler_name}...")
        start = time.perf_counter()
        try:
            split = capture_sampler_split(
                bundle,
                sampler_name,
                SAMPLER_REGISTRY[sampler_name],
                args.n_samples,
                args.random_state,
            )
            train_idx = _id_to_indices(tracer_ids, split["train_tracers"])
            val_idx = _id_to_indices(tracer_ids, split["val_tracers"])
            test_idx = _id_to_indices(tracer_ids, split["test_tracers"])
            elapsed = time.perf_counter() - start

            combined = np.concatenate([train_idx, val_idx, test_idx])
            overlap_count = int(len(combined) - len(np.unique(combined)))
            result = {
                "sampler": sampler_name,
                "status": "ok",
                "seconds": elapsed,
                "sizes": {
                    "train": int(len(train_idx)),
                    "val": int(len(val_idx)),
                    "test": int(len(test_idx)),
                },
                "overlap_count": overlap_count,
                "train_coverage_vs_full": nearest_similarity_stats(
                    vectors,
                    all_indices,
                    train_idx,
                    args.thresholds,
                    centered=True,
                    center_idx=train_idx,
                    max_reference=args.max_reference,
                    max_candidate=args.max_candidate,
                    seed=args.random_state,
                ),
                "cross_split_similarity": {
                    "val_to_train": nearest_similarity_stats(
                        vectors,
                        val_idx,
                        train_idx,
                        args.thresholds,
                        centered=True,
                        center_idx=train_idx,
                        max_reference=args.max_reference,
                        max_candidate=args.max_candidate,
                        seed=args.random_state + 10,
                    ),
                    "test_to_train": nearest_similarity_stats(
                        vectors,
                        test_idx,
                        train_idx,
                        args.thresholds,
                        centered=True,
                        center_idx=train_idx,
                        max_reference=args.max_reference,
                        max_candidate=args.max_candidate,
                        seed=args.random_state + 20,
                    ),
                    "test_to_val": nearest_similarity_stats(
                        vectors,
                        test_idx,
                        val_idx,
                        args.thresholds,
                        centered=True,
                        center_idx=val_idx,
                        max_reference=args.max_reference,
                        max_candidate=args.max_candidate,
                        seed=args.random_state + 30,
                    ),
                },
                "within_split_similarity": {
                    "train": pairwise_similarity_stats(
                        vectors,
                        train_idx,
                        args.thresholds,
                        centered=True,
                        max_n=args.max_pairwise,
                        seed=args.random_state + 40,
                    ),
                    "val": pairwise_similarity_stats(
                        vectors,
                        val_idx,
                        args.thresholds,
                        centered=True,
                        max_n=args.max_pairwise,
                        seed=args.random_state + 50,
                    ),
                    "test": pairwise_similarity_stats(
                        vectors,
                        test_idx,
                        args.thresholds,
                        centered=True,
                        max_n=args.max_pairwise,
                        seed=args.random_state + 60,
                    ),
                },
                "final_density_ks_vs_full": {
                    "train": final_density_ks(bundle, train_idx),
                    "val": final_density_ks(bundle, val_idx),
                    "test": final_density_ks(bundle, test_idx),
                },
            }
            print(
                f"  ok in {elapsed:.2f}s; sizes={result['sizes']}; "
                f"overlap={overlap_count}"
            )
        except Exception as exc:
            result = {
                "sampler": sampler_name,
                "status": "error",
                "seconds": time.perf_counter() - start,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  error: {result['error']}")
        results.append(result)

    ranking = score_results(results, args.primary_threshold)
    output = {
        "config": {
            "bundle_path": str(bundle_path),
            "n_samples": args.n_samples,
            "random_state": args.random_state,
            "thresholds": args.thresholds,
            "primary_threshold": args.primary_threshold,
            "max_reference": args.max_reference,
            "max_candidate": args.max_candidate,
            "max_pairwise": args.max_pairwise,
            "score_definition": (
                "Lower is better. Composite rank weights: near_duplicate_rate=3, "
                "coverage_distance=1.5, density_ks_train=1.5, "
                "train_nearest_similarity=1, seconds=0.25."
            ),
        },
        "results": results,
        "ranking": ranking,
        "best_sampler": ranking[0]["sampler"] if ranking else None,
    }

    json_path = results_dir / "sampler_benchmark_results.json"
    ranking_path = results_dir / "sampler_ranking.csv"
    json_path.write_text(json.dumps(output, indent=2))
    write_ranking_csv(ranking_path, ranking)

    print("\nRanking:")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank}. {row['sampler']} score={row['composite_score']:.2f} "
            f"near_dup={row['near_duplicate_rate']:.4f} "
            f"coverage_dist={row['coverage_distance']:.4e} "
            f"ks={row['density_ks_train']:.4f}"
        )
    print(f"\nWrote {json_path}")
    print(f"Wrote {ranking_path}")


if __name__ == "__main__":
    main()
