import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import PHYS_COLS
from model import MLP


DEFAULT_SPECIES = ["H", "H2", "O", "C", "N", "CL", "E_minus", "CO", "MG", "#C", "H2O", "SI"]


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
        values.extend(["E-", "E"])
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


def main(model_checkpoint, test_dir, output_dir, epoch_checkpoint_dir=None, species=None, num_tracers=10):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model = MLP.load_from_checkpoint(model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data, phys_cols, abundance_cols = load_dataset(test_dir)
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

    all_errors = np.concatenate(all_errors, axis=0)
    species_mse = np.mean(all_errors, axis=0)
    summary = {
        "num_tracers": len(tracer_errors),
        "overall_mse": float(np.mean(all_errors)),
        "min_tracer_mse": float(min(row["mse"] for row in tracer_errors)),
        "avg_tracer_mse": float(np.mean([row["mse"] for row in tracer_errors])),
        "max_tracer_mse": float(max(row["mse"] for row in tracer_errors)),
        "plot_species": selected_species,
    }
    pd.DataFrame(tracer_errors).to_csv(os.path.join(output_dir, "tracer_errors.csv"), index=False)
    pd.DataFrame({"species": abundance_cols, "mse": species_mse}).to_csv(
        os.path.join(output_dir, "species_mse.csv"), index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        os.path.join(output_dir, "test_predictions_log10.csv"), index=False
    )
    with open(os.path.join(output_dir, "error_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if selected_species:
        ranked = sorted(tracer_errors, key=lambda row: row["mse"])
        half = max(1, num_tracers // 2)
        plot_dir = os.path.join(output_dir, "rollouts")
        os.makedirs(plot_dir, exist_ok=True)
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
                os.path.join(plot_dir, f"rollout_tracer_{tracer}.png"),
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="results/mlp_grav_collapse.ckpt")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/test_results")
    parser.add_argument("--epoch_checkpoint_dir", type=str, default=None)
    parser.add_argument("--species", nargs="+", default=DEFAULT_SPECIES)
    parser.add_argument("--num_tracers", type=int, default=10)
    args = parser.parse_args()
    main(
        args.model_checkpoint,
        args.test_dir,
        args.output_dir,
        args.epoch_checkpoint_dir,
        args.species,
        args.num_tracers,
    )
