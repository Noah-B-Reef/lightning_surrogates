import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import settings as config
from data import PHYS_COLS
from model import MLP


DEFAULT_SPECIES = ["H", "H2", "O", "C", "N", "CL", "E_minus", "CO", "MG", "#C", "H2O", "SI"]


def resolve_test_csv(data_dir=None, test_csv=None):
    if test_csv is not None:
        path = Path(test_csv).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Test CSV not found: {path}")
        return path
    # No explicit test CSV: resolve the split directory (defaulting to the
    # best-sampler split) and use its test.csv.
    return config.resolve_split_dir(data_dir, required=("test.csv",)) / "test.csv"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["dstep"], errors="ignore")
    df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)
    phys_cols = [col for col in PHYS_COLS if col in df.columns]
    abundance_cols = [
        col
        for col in df.columns
        if col not in ("Tracer", "Time", "dstep", "BULK", "SURFACE") and col not in phys_cols
    ]
    return df, phys_cols, abundance_cols


def aliases(species):
    values = [species]
    if species == "E_minus":
        values.extend(["E-", "E"] )
    if species.startswith("#"):
        values.append("@" + species[1:])
    return values


def resolve_species(requested, abundance_cols):
    selected = []
    for species in requested:
        for alias in aliases(species):
            if alias in abundance_cols and alias not in selected:
                selected.append(alias)
                break
    return selected


def rollout_tracer(model, tracer_df, phys_cols, abundance_cols, device):
    if len(tracer_df) < 2:
        return None, None
    phys = tracer_df[phys_cols].to_numpy(dtype=np.float32)
    abund = tracer_df[abundance_cols].to_numpy(dtype=np.float32)
    true_log = np.log10(np.maximum(abund, 1e-30)).astype(np.float32)
    current = torch.tensor(true_log[0], dtype=torch.float32, device=device)
    predictions = [true_log[0]]
    model.eval()
    with torch.no_grad():
        for step in range(len(tracer_df) - 1):
            phys_t = torch.tensor(phys[step], dtype=torch.float32, device=device)
            x = torch.cat([phys_t, current]).unsqueeze(0)
            current = model(x).squeeze(0)
            predictions.append(current.detach().cpu().numpy())
    return np.asarray(predictions), true_log


def plot_rollout(tracer, time, true_vals, pred_vals, species, path):
    n_cols = 3
    n_rows = int(np.ceil(len(species) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, name in enumerate(species):
        ax = axes[idx]
        ax.plot(time, true_vals[:, idx], label="True")
        ax.plot(time, pred_vals[:, idx], label="Rollout", linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("log10 abundance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for idx in range(len(species), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Autoregressive Rollout: Tracer {tracer}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    model_checkpoint,
    data_dir=None,
    output_dir=config.TEST_OUTPUT_DIR,
    test_dir=None,
    species=None,
    num_tracers=10,
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_csv = resolve_test_csv(data_dir=data_dir, test_csv=test_dir)
    model = MLP.load_from_checkpoint(str(Path(model_checkpoint).expanduser().resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data, phys_cols, abundance_cols = load_dataset(test_csv)
    selected_species = resolve_species(species or DEFAULT_SPECIES, abundance_cols)
    selected_idx = [abundance_cols.index(name) for name in selected_species]
    predictions = []
    tracer_errors = []
    all_errors = []

    for tracer, tracer_df in data.groupby("Tracer", sort=False):
        tracer_df = tracer_df.reset_index(drop=True)
        pred, true = rollout_tracer(model, tracer_df, phys_cols, abundance_cols, device)
        if pred is None:
            continue
        se = (pred[1:] - true[1:]) ** 2
        all_errors.append(se)
        tracer_errors.append(
            {
                "tracer": tracer,
                "mse": float(np.mean(se)),
                "min_mse": float(np.min(np.mean(se, axis=1))),
                "max_mse": float(np.max(np.mean(se, axis=1))),
            }
        )
        pred_df = tracer_df.copy()
        pred_df.loc[:, abundance_cols] = pred
        predictions.append(pred_df)

    if not all_errors:
        raise RuntimeError("No valid test tracers were available for rollout.")

    all_errors = np.concatenate(all_errors, axis=0)
    species_mse = np.mean(all_errors, axis=0)
    summary = {
        "test_csv": str(test_csv),
        "num_tracers": len(tracer_errors),
        "overall_mse": float(np.mean(all_errors)),
        "min_tracer_mse": float(min(row["mse"] for row in tracer_errors)),
        "avg_tracer_mse": float(np.mean([row["mse"] for row in tracer_errors])),
        "max_tracer_mse": float(max(row["mse"] for row in tracer_errors)),
        "plot_species": selected_species,
    }
    pd.DataFrame(tracer_errors).to_csv(output_dir / "tracer_errors.csv", index=False)
    pd.DataFrame({"species": abundance_cols, "mse": species_mse}).to_csv(
        output_dir / "species_mse.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "test_predictions_log10.csv", index=False
    )
    (output_dir / "error_summary.json").write_text(json.dumps(summary, indent=2))

    if selected_species:
        ranked = sorted(tracer_errors, key=lambda row: row["mse"])
        half = max(1, int(num_tracers) // 2)
        plot_dir = output_dir / "rollouts"
        plot_dir.mkdir(exist_ok=True)
        for row in ranked[:half] + ranked[-half:]:
            tracer = row["tracer"]
            tracer_df = data[data["Tracer"] == tracer].reset_index(drop=True)
            pred, true = rollout_tracer(model, tracer_df, phys_cols, abundance_cols, device)
            plot_rollout(
                tracer,
                tracer_df["Time"].to_numpy(),
                true[:, selected_idx],
                pred[:, selected_idx],
                selected_species,
                plot_dir / f"rollout_tracer_{tracer}.png",
            )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an MLP checkpoint with autoregressive rollout.")
    parser.add_argument("dataset_path", nargs="?", default=None, help="Split directory containing test.csv.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Alias for dataset_path.")
    parser.add_argument("--model-checkpoint", "--model_checkpoint", type=Path, default=config.TEST_MODEL_CHECKPOINT)
    parser.add_argument("--test-dir", "--test_dir", "--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=config.TEST_OUTPUT_DIR)
    parser.add_argument("--species", nargs="+", default=DEFAULT_SPECIES)
    parser.add_argument("--num-tracers", "--num_tracers", type=int, default=10)
    args = parser.parse_args()
    main(
        model_checkpoint=args.model_checkpoint,
        data_dir=args.data_dir or args.dataset_path,
        output_dir=args.output_dir,
        test_dir=args.test_dir,
        species=args.species,
        num_tracers=args.num_tracers,
    )
