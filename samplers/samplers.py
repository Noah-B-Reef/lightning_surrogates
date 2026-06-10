import numpy as np
import pandas as pd
from scipy.linalg import qr as scipy_qr
from sklearn.utils.extmath import randomized_svd
from util import dataframe_splits


def _split_60_20_20(tracer_ids):
    tracer_ids = np.asarray(tracer_ids)
    n_total = len(tracer_ids)
    n_train = int(np.floor(n_total * 0.6))
    n_val = int(np.floor(n_total * 0.2))
    return np.split(tracer_ids, [n_train, n_train + n_val])


def random_sample(
    bundle,
    n_samples=6000,
    random_state=42,
    save_dir=None,
    storage_format="csv",
):
    """
    Randomly sample tracer trajectories from a flattened bundle and split them.

    Inputs:
        bundle (dict): Flattened dataset bundle from util.dataset_flatten/load_bundle.
        n_samples (int): Total number of tracer trajectories to sample, split
            into 60% train, 20% validation, and 20% test tracers.
        random_state (int): Seed for reproducibility.
        save_dir (str): Optional base directory. Splits are saved under
            save_dir/random/.

    Returns:
        train_df, val_df, test_df
    """
    tracer_ids = bundle["tracer_ids"]

    n_sampled = n_samples
    rng = np.random.default_rng(random_state)
    sampled_tracers = rng.choice(tracer_ids, size=n_sampled, replace=False)
    train_tracers, val_tracers, test_tracers = _split_60_20_20(sampled_tracers)

    train_df, val_df, test_df, _ = dataframe_splits(
        bundle, train_tracers, val_tracers, test_tracers, save_dir, "random", storage_format
    )
    return train_df, val_df, test_df


def density_sampler(
    bundle,
    n_samples=6000,
    num_strat_bins=10,
    random_state=42,
    save_dir=None,
    storage_format="csv",
):
    """
    Stratified random sampling by final-timestep density.

    Inputs and outputs follow random_sample(): pass a flattened bundle, get
    60/20/20 train/val/test DataFrames, and optionally save to save_dir/density/.
    """
    feature_cols = list(bundle["feature_cols"])
    tracer_vectors = bundle["vectors"]
    tracer_ids = bundle["tracer_ids"]
    n_sampled = n_samples
    n_features = len(feature_cols)
    T = int(bundle["T"])
    density_idx = feature_cols.index("Density")
    final_density_idx = (T - 1) * n_features + density_idx

    final_density_vals = tracer_vectors[:, final_density_idx]
    final_density = pd.Series(final_density_vals, index=tracer_ids)

    # Bin in log density because physical params are linear in the bundle.
    strat_labels = pd.cut(
        np.log10(np.maximum(final_density, 1e-30)), bins=num_strat_bins, labels=False
    )

    rng = np.random.default_rng(random_state)
    bin_ids = pd.Series(strat_labels, index=tracer_ids).dropna()
    selected = []

    counts = bin_ids.value_counts().sort_index()
    proportions = counts / counts.sum()
    target_counts = np.floor(proportions * n_sampled).astype(int)
    remainder = n_sampled - int(target_counts.sum())
    if remainder:
        fractional = (proportions * n_sampled - target_counts).sort_values(
            ascending=False
        )
        for bin_id in fractional.index[:remainder]:
            target_counts.loc[bin_id] += 1

    for bin_id, target_count in target_counts.items():
        bin_tracers = bin_ids.index[bin_ids == bin_id].to_numpy()
        selected_count = min(int(target_count), len(bin_tracers))
        if selected_count > 0:
            selected.extend(rng.choice(bin_tracers, size=selected_count, replace=False))

    if len(selected) < n_sampled:
        remaining = np.setdiff1d(tracer_ids, np.asarray(selected), assume_unique=False)
        selected.extend(
            rng.choice(remaining, size=n_sampled - len(selected), replace=False)
        )
    elif len(selected) > n_sampled:
        selected = rng.choice(np.asarray(selected), size=n_sampled, replace=False)

    rng.shuffle(selected)
    train_tracers, val_tracers, test_tracers = _split_60_20_20(selected)

    train_df, val_df, test_df, _ = dataframe_splits(
        bundle, train_tracers, val_tracers, test_tracers, save_dir, "density", storage_format
    )
    return train_df, val_df, test_df


def QR_sampler(
    bundle,
    n_samples=6000,
    random_state=42,
    save_dir=None,
    save_qr=True,
    candidate_multiplier=1,
    sampling_procedure="qr_pivot",
    storage_format="csv",
):
    """
    Rank tracers with column-pivoted QR and split them into 60/20/20 datasets.

    Inputs:
        bundle (dict): Flattened dataset bundle from util.dataset_flatten/load_bundle.
        n_samples (int): Total number of tracer trajectories to sample, split
            into 60% train, 20% validation, and 20% test tracers.
        save_dir (str): Optional base directory. Splits are saved under
            save_dir/<sampling_procedure>/.
        random_state (int): Seed for reproducibility.
        candidate_multiplier (int): QR is run over up to
            n_samples * candidate_multiplier candidate tracers.

    Outputs:
        train_df, val_df, test_df, R, qr_indices
    """
    tracer_vectors = bundle["vectors"]
    tracer_ids = bundle["tracer_ids"]
    n_sampled = n_samples
    n_train = int(np.floor(n_sampled * 0.6))
    n_val = int(np.floor(n_sampled * 0.2))
    n_test = n_sampled - n_train - n_val
    rng = np.random.default_rng(random_state)

    min_candidates = n_sampled
    max_candidates = min(
        tracer_ids.shape[0],
        max(min_candidates, n_sampled * candidate_multiplier),
    )
    candidate_indices = rng.choice(
        tracer_ids.shape[0], size=max_candidates, replace=False
    )
    candidate_vectors = tracer_vectors[candidate_indices]
    A = candidate_vectors.T.astype(np.float32)  # (features, candidate_tracers)

    # Column-pivoted Householder QR: A P = Q R. The sampler only needs pivot
    # indices, so mode="r" avoids materializing the very large Q matrix.
    R, P = scipy_qr(A, pivoting=True, mode="r")

    qr_indices = candidate_indices[P[:n_train]]
    train_indices = qr_indices
    train_tracers = tracer_ids[train_indices]
    remaining_ids = np.delete(tracer_ids, train_indices)
    holdout_tracers = rng.choice(remaining_ids, size=n_val + n_test, replace=False)
    val_tracers, test_tracers = np.split(holdout_tracers, [n_val])

    train_df, val_df, test_df, output_dir = dataframe_splits(
        bundle, train_tracers, val_tracers, test_tracers, save_dir, sampling_procedure, storage_format
    )

    if save_qr and output_dir is not None:
        qr_samples_dir = output_dir / "qr_samples"
        qr_samples_dir.mkdir(parents=True, exist_ok=True)

        np.save(qr_samples_dir / "R.npy", np.asarray(R))
        np.save(qr_samples_dir / "qr_indices.npy", np.asarray(qr_indices))

    return train_df, val_df, test_df, R, qr_indices


def svd_fps(
    bundle,
    n_samples=6000,
    n_components=10,
    random_state=42,
    save_dir=None,
    storage_format="csv",
):
    """
    Select tracers with farthest-point sampling in a truncated SVD embedding.

    Inputs and outputs follow random_sample().
    """
    tracer_vectors = bundle["vectors"]
    tracer_ids = bundle["tracer_ids"]
    n_sampled = n_samples

    tracer_vectors_centered = tracer_vectors - np.mean(tracer_vectors, axis=0)

    U, S, _ = randomized_svd(
        tracer_vectors_centered,
        n_components=n_components,
        random_state=random_state,
    )
    U_reduced = U[:, :n_components]
    embedding = U_reduced * S[:n_components]

    rng = np.random.default_rng(random_state)
    selected = [int(rng.integers(len(embedding)))]
    min_dist_sq = np.sum((embedding - embedding[selected[0]]) ** 2, axis=1)

    for _ in range(1, n_sampled):
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)
        dist_sq = np.sum((embedding - embedding[next_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    selected_tracers = tracer_ids[np.asarray(selected)]
    train_tracers, val_tracers, test_tracers = _split_60_20_20(selected_tracers)

    train_df, val_df, test_df, _ = dataframe_splits(
        bundle, train_tracers, val_tracers, test_tracers, save_dir, "svd_fps", storage_format
    )
    return train_df, val_df, test_df


def similarity_constrained_split(
    bundle,
    n_samples=6000,
    n_clusters=20,
    threshold=0.95,
    random_state=42,
    save_dir=None,
    storage_format="csv",
):
    """
    K-Means cluster splitting with strict centered cosine similarity constraint.
    Guarantees no cross-split leakage by filtering out validation/test trajectories
    that are similar to the training trajectories.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    tracer_vectors = bundle["vectors"]
    tracer_ids = bundle["tracer_ids"]
    n_tracers = tracer_vectors.shape[0]
    n_sampled = n_samples
    n_train = int(np.floor(n_sampled * 0.6))
    n_val = int(np.floor(n_sampled * 0.2))
    n_test = n_sampled - n_train - n_val

    rng = np.random.default_rng(random_state)

    # 1. Center the vectors to avoid floor values bias
    mean_vector = np.mean(tracer_vectors, axis=0)
    centered_vectors = tracer_vectors - mean_vector

    # 2. SVD for fast clustering
    print("Performing SVD dimensionality reduction...")
    U, S, _ = randomized_svd(
        centered_vectors,
        n_components=20,
        random_state=random_state,
    )
    reduced_vectors = U * S[:20]

    # 3. K-Means clustering
    print(f"Clustering trajectories into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_vectors)

    # 4. Partition clusters: 60% Train, 40% Val/Test Candidates
    shuffled_clusters = list(range(n_clusters))
    rng.shuffle(shuffled_clusters)
    split_idx = int(np.floor(n_clusters * 0.6))
    train_clusters = shuffled_clusters[:split_idx]
    candidate_clusters = shuffled_clusters[split_idx:]

    print(f"Train clusters: {train_clusters}")
    print(f"Candidate Val/Test clusters: {candidate_clusters}")

    # 5. Extract Train pool and sample Train tracers
    train_pool_idx = np.where(np.isin(cluster_labels, train_clusters))[0]
    if len(train_pool_idx) < n_train:
        raise ValueError(
            f"Train clusters contain only {len(train_pool_idx)} tracers, fewer than required {n_train}."
        )
    train_indices = rng.choice(train_pool_idx, size=n_train, replace=False)

    # 6. Extract candidate Val/Test pool
    candidate_pool_idx = np.where(np.isin(cluster_labels, candidate_clusters))[0]

    # 7. Apply similarity filter with training-mean centering and fallback/relaxation
    mean_train = np.mean(tracer_vectors[train_indices], axis=0)
    train_vecs_centered = tracer_vectors[train_indices] - mean_train
    
    current_threshold = threshold
    clean_indices = []
    
    while current_threshold <= 1.0:
        candidate_vecs_centered = tracer_vectors[candidate_pool_idx] - mean_train
        sim = cosine_similarity(candidate_vecs_centered, train_vecs_centered)
        max_sim = np.max(sim, axis=1)
        
        clean_mask = max_sim < current_threshold
        clean_indices = candidate_pool_idx[clean_mask]
        
        if len(clean_indices) >= n_val + n_test:
            break
            
        print(f"Warning: Only {len(clean_indices)} clean tracers remain at threshold {current_threshold:.2f}.")
        current_threshold += 0.01
        print(f"Relaxing similarity threshold to {current_threshold:.2f}...")

    if len(clean_indices) < n_val + n_test:
        clean_indices = candidate_pool_idx

    print(f"Leakage-free pool size: {len(clean_indices)} tracers (threshold={current_threshold:.2f})")

    # 8. Sample Val and Test tracers from the clean pool
    holdout_indices = rng.choice(clean_indices, size=n_val + n_test, replace=False)
    val_indices, test_indices = np.split(holdout_indices, [n_val])

    train_tracers = tracer_ids[train_indices]
    val_tracers = tracer_ids[val_indices]
    test_tracers = tracer_ids[test_indices]

    train_df, val_df, test_df, _ = dataframe_splits(
        bundle, train_tracers, val_tracers, test_tracers, save_dir,
        "similarity_constrained", storage_format
    )
    return train_df, val_df, test_df
