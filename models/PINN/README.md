# PINN — physics-informed surrogate for the GOW17 network

Physics-informed neural network for the GOW17 (Gong, Ostriker & Wolfire 2017,
ApJ 843, 38) astrochemistry network, mirroring the MLP surrogate in
`models/mlp` with two additions: elapsed time Δt is a model input, and the
chemical ODE system enters the loss as a physics residual.

## Model

```
yhat(Δt) = y0 + (Δt / dt_ref) * net([phys_norm, y0, Δt/dt_ref])
```

- `y0` = log10 abundances at t, `phys` = (Density, gasTemp, Av, radfield),
  normalized exactly as in the MLP (log10 + standardization, train-split stats).
- Hard initial condition: `yhat(0) = y0`.
- Training pairs span 1..`PINN_MAX_HORIZON` (default 4) consecutive snapshots,
  so the network sees Δt = 253, 506, 759, 1012 yr instead of a single step.

Loss = L1 data loss
  + `PINN_PHYSICS_WEIGHT` (0.1) × mean |relative ODE residual|
  + `PINN_CONSERVATION_WEIGHT` (0.01) × mean |relative invariant drift|.

The ODE residual compares the autograd time derivative of the prediction
(double-vjp trick, one collocation time per sample, random in (0, Δt] by
default) with the GOW17 RHS at the predicted state:

```
r_i = (dy_i/dt − (P_i − D_i)) / (|dy_i/dt| + P_i + D_i + y_i·rate_floor + eps)
```

P/D are gross production/destruction (s^-1); the symmetric normalization
bounds |r| ≤ 1 per species — necessary because the system is stiff (chemical
timescales span >10 orders of magnitude). Conservation drift is measured for
the 6 linear invariants (charge + elemental H/He/C/O/Si), each normalized by
its gross inventory (net charge is ~0 by neutrality).

## Layout

- [src/gow17_network.py](src/gow17_network.py) — parses
  [network/](network/)`{species,reactions}.dat` (copied from
  `chemistry-benchmark-surrogates/networks/gow17/`); prints the full ODE
  listing ([GOW17_ODES.txt](GOW17_ODES.txt)).
- [src/gow17_rates.py](src/gow17_rates.py) — differentiable torch RHS: all 50
  reactions including the frml-7 "customized" rates ported from
  `kida_gow17.cpp` (grain-assisted recombination, H2O+/CH2+ branchings,
  collisional dissociation, charge exchange). Stoichiometry is built from
  `reactions.dat`, so `R @ nu^T` conserves the invariants to machine precision.
- [src/data.py](src/data.py) — multi-horizon pair dataset, Δt as last feature,
  abundances re-indexed to `gow17_rates.SPECIES` order.
- [src/model.py](src/model.py) — `PINN` LightningModule (losses above).
- [src/train.py](src/train.py) / [src/test.py](src/test.py) — same CLI shape
  as the MLP; `test.py --rollout-stride k` rolls out with k-snapshot jumps to
  exercise the Δt input.
- [src/validate_rhs.py](src/validate_rhs.py) — checks the RHS against finite
  differences of the dataset and sweeps zeta-unit candidates.
- [src/optimize.py](src/optimize.py) — sequential Optuna study mirroring the
  MLP's: tunes architecture, learning rate, batch size, and the
  physics/conservation loss weights. Objective is `val_mse`, a data-space
  metric independent of the loss weights, so trials are comparable.
- [slurm/](slurm/) — pipeline/optimize/train/test jobs; configuration comes
  from the repo-root `config.sh` (defaults to `DATASET_NAME=gow17_R0.05_M6.0`).
  `pipeline.slurm` runs sampling -> optimization -> training -> testing,
  mirroring `models/mlp/slurm/pipeline.slurm`.

Run locally (lightning conda env):

```bash
source config.sh
python models/PINN/src/optimize.py   # optional; writes best_params.json
python models/PINN/src/train.py      # add --config-file <best_params.json>
python models/PINN/src/test.py
```

## RHS validation against the dataset (gow17_R0.05_M6.0)

`validate_rhs.py` and the production/destruction balance at data states give:

- **zeta calibration**: the dataset's `zeta` column (constant 1.2213740458)
  has no documented unit. The H2+ quasi-steady-state balance pins the
  effective primary CR ionization rate to **2.024e-17 s^-1** (spread < 1%),
  so `PINN_ZETA_UNIT` defaults to 1.657e-17.
- **Verified clean** (P/D at data states ≈ 1.000): H+, H2+, HCO+; slow
  species C, CO track the finite-difference dy/dt to ~5–15%.
- **Known mismatches** vs. whatever rate set generated the data
  (M. Bonfand's run of Gong's original code): He+/Si+ recombination ~1.27×,
  H3+ destruction ~2× (O + H3+ channel), OHx equilibrium ~10^4-10^5 off, O+
  production effectively absent below ~100 K in our rates. The bounded
  residual keeps these from destabilizing training — they contribute an O(1)
  floor to the physics loss for those (trace) species. Revisit if the exact
  generator rates become available.
- FUV photo rates are ~0 everywhere in this dataset (Av ≥ 2.4), so the
  missing H2/CO self-shielding treatment is irrelevant here.
- Collisional dissociation (R41–43) only activates above 700 K; dataset max
  is 36 K.

## Conservation laws (free PINN constraints)

- charge: n(e-) = Σ n(ion+)
- H: n(H) + 2n(H2) + n(H+) + 2n(H2+) + 3n(H3+) + n(CHx) + n(OHx) + n(HCO+)
- He: n(He) + n(He+);  Si: n(Si) + n(Si+)
- C: n(C) + n(C+) + n(CHx) + n(CO) + n(HCO+)
- O: n(O) + n(O+) + n(OHx) + n(CO) + n(HCO+)
