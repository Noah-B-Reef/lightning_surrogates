"""Sanity-check the torch GOW17 RHS against finite differences from the data.

For random rows of a split, compares (y_{t+1} - y_t) / dt with the RHS
evaluated at the midpoint state, per species, as a relative error. Also
sweeps candidate zeta unit conversions (the dataset's zeta column is a bare
number) to see which one the data was generated with.

Usage: python validate_rhs.py [split_dir] [--split train] [--n 4096]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gow17_rates import GOW17RHS, SPECIES  # noqa: E402
import settings as config  # noqa: E402
from data import load_split_dataframe  # noqa: E402

SECONDS_PER_YEAR = 3.15576e7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", nargs="?", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_dir = config.resolve_split_dir(args.split_dir, required=(args.split,))
    df = load_split_dataframe(split_dir, args.split)
    df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    same_tracer = df["Tracer"].to_numpy()[:-1] == df["Tracer"].to_numpy()[1:]
    starts = np.flatnonzero(same_tracer)
    starts = rng.choice(starts, size=min(args.n, len(starts)), replace=False)

    y0 = torch.tensor(df.loc[starts, list(SPECIES)].to_numpy(), dtype=torch.float64)
    y1 = torch.tensor(df.loc[starts + 1, list(SPECIES)].to_numpy(), dtype=torch.float64)
    dt_s = torch.tensor(
        (df.loc[starts + 1, "Time"].to_numpy() - df.loc[starts, "Time"].to_numpy())
        * SECONDS_PER_YEAR,
        dtype=torch.float64,
    )
    env = {
        k: torch.tensor(df.loc[starts, c].to_numpy(), dtype=torch.float64)
        for k, c in [("nH", "Density"), ("T", "gasTemp"), ("Av", "Av"),
                     ("chi", "radfield"), ("zeta_raw", "zeta")]
    }

    fd = (y1 - y0) / dt_s[:, None]
    ymid = 0.5 * (y0 + y1)
    rhs = GOW17RHS()

    zeta_units = {
        "zeta_col * 1.3e-17 (UCLCHEM unit)": env["zeta_raw"] * 1.3e-17,
        "zeta_col * 1e-16": env["zeta_raw"] * 1e-16,
        "2e-16 (GOW17 default)": torch.full_like(env["zeta_raw"], 2e-16),
        "1.3e-17 (ignore column)": torch.full_like(env["zeta_raw"], 1.3e-17),
    }

    for label, zeta in zeta_units.items():
        pred = rhs(ymid, env["T"], env["nH"], env["chi"], env["Av"], zeta)
        rel = (pred - fd).abs() / (pred.abs() + fd.abs() + 1e-30)
        print(f"\n=== zeta = {label} ===")
        print(f"  median relative residual over all species: {rel.median():.4f}")
        for i, sp in enumerate(SPECIES):
            active = fd[:, i].abs() > 0
            med = rel[active, i].median() if active.any() else float("nan")
            print(f"    {sp:6s} median_rel={med:.4f}  (active rows: {int(active.sum())})")


if __name__ == "__main__":
    main()
