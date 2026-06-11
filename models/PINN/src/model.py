"""Physics-informed MLP surrogate for the GOW17 network.

Input:  [physical parameters at t (raw), log10 abundances at t, dt_years]
Output: log10 abundances at t + dt

The network predicts a correction around the initial state with a hard
initial condition,

    yhat(dt) = y0 + (dt / dt_ref) * net(phys, y0, dt),

so yhat(0) = y0 exactly and d(yhat)/d(dt) is meaningful at dt = 0.

Loss = L1(yhat(dt), y(t+dt))                                  (data)
     + w_phys * |relative ODE residual at a collocation time|  (physics)
     + w_cons * |relative drift of charge/element invariants|  (conservation)

The ODE residual compares the autograd time derivative of the prediction
with the GOW17 chemical RHS evaluated at the predicted state:

    r_i = (dy_i/dt - (P_i - D_i))
          / (|dy_i/dt| + P_i + D_i + y_i * rate_floor + eps)

where P/D are the gross production/destruction rates (s^-1). The symmetric
normalization bounds |r| <= 1 for every species regardless of how many
orders of magnitude its chemical timescale spans (the system is stiff), and
the rate_floor mutes species that are chemically frozen on the simulated
timescales. Without the |dy/dt| term the residual of an untrained network
is unbounded and overflows float32 in the backward pass.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim

from gow17_rates import GOW17RHS, SPECIES, conservation_matrix

SECONDS_PER_YEAR = 3.15576e7
LN10 = 2.302585092994046
# log10-abundance clamp before exponentiation, for float32 safety.
LOG_ABUND_MIN, LOG_ABUND_MAX = -37.0, 1.0
EPS = 1e-30


class PINN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.num_phys = int(config["num_phys"])
        self.num_species = int(config["output_size"])
        if self.num_species != len(SPECIES):
            raise ValueError(
                f"output_size={self.num_species} != {len(SPECIES)} GOW17 species"
            )
        num_inputs = int(config["num_inputs"])  # phys + species + dt
        hidden_layers = int(config["num_hidden_layers"])
        hidden_units = int(config["num_neurons_per_hidden_layer"])
        self.learning_rate = float(config.get("learning_rate", 1e-3))

        # Loss weights and physics constants (saved with the checkpoint).
        self.physics_weight = float(config.get("physics_weight", 0.1))
        self.conservation_weight = float(config.get("conservation_weight", 0.01))
        self.dt_ref_years = float(config.get("dt_ref_years", 1000.0))
        self.zeta_phys = float(config.get("zeta_column_value", 1.0)) * float(
            config.get("zeta_unit", 1.657e-17)
        )
        self.residual_rate_floor = float(config.get("residual_rate_floor", 1e-14))
        self.random_collocation = bool(config.get("random_collocation", True))

        # Map physical columns to the chemical environment variables.
        phys_cols = list(config.get("phys_cols", ["Density", "gasTemp", "Av", "radfield"]))
        try:
            self._env_idx = {
                "nH": phys_cols.index("Density"),
                "T": phys_cols.index("gasTemp"),
                "Av": phys_cols.index("Av"),
                "chi": phys_cols.index("radfield"),
            }
        except ValueError as err:
            raise ValueError(f"phys_cols {phys_cols} missing a physics input") from err

        # Physical-parameter normalization, same scheme as the MLP: stats come
        # from the training split (data.GOW17DataModule.phys_norm_config).
        log_mask = config.get("phys_log_mask") or [0.0] * self.num_phys
        phys_mean = config.get("phys_mean") or [0.0] * self.num_phys
        phys_std = config.get("phys_std") or [1.0] * self.num_phys
        phys_log_floor = config.get("phys_log_floor") or [1e-30] * self.num_phys
        self.register_buffer("phys_log_mask", torch.tensor(log_mask, dtype=torch.float32))
        self.register_buffer("phys_mean", torch.tensor(phys_mean, dtype=torch.float32))
        self.register_buffer("phys_std", torch.tensor(phys_std, dtype=torch.float32))
        self.register_buffer(
            "phys_log_floor", torch.tensor(phys_log_floor, dtype=torch.float32)
        )
        self.register_buffer("cons_matrix", conservation_matrix())

        self.rhs = GOW17RHS()

        layers = [nn.Linear(num_inputs, hidden_units), nn.Tanh()]
        for _ in range(max(0, hidden_layers - 1)):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, self.num_species))
        self.network = nn.Sequential(*layers)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def split_inputs(self, x):
        """[phys raw, log10 abundances, dt_years] columns of a batch."""
        n = self.num_phys
        return x[:, :n], x[:, n : n + self.num_species], x[:, -1]

    def _normalize_phys(self, phys):
        mask = self.phys_log_mask.bool()
        if mask.any():
            logged = torch.log10(torch.maximum(phys, self.phys_log_floor))
            phys = torch.where(mask, logged, phys)
        return (phys - self.phys_mean) / self.phys_std

    def predict_log_abund(self, phys, y0, dt_years):
        """yhat(dt) with the hard initial condition yhat(0) = y0."""
        dt_norm = dt_years / self.dt_ref_years
        z = torch.cat(
            [self._normalize_phys(phys), y0, dt_norm.unsqueeze(-1)], dim=1
        )
        return y0 + dt_norm.unsqueeze(-1) * self.network(z)

    def forward(self, x):
        phys, y0, dt_years = self.split_inputs(x)
        return self.predict_log_abund(phys, y0, dt_years)

    # ------------------------------------------------------------------ #
    # losses                                                             #
    # ------------------------------------------------------------------ #
    def _env(self, phys):
        i = self._env_idx
        return (
            phys[:, i["T"]],
            phys[:, i["nH"]],
            phys[:, i["chi"]],
            phys[:, i["Av"]],
            torch.full_like(phys[:, 0], self.zeta_phys),
        )

    def physics_residual_loss(self, phys, y0, dt_years):
        """Mean |relative ODE residual| at one collocation time per sample.

        Needs grad mode even during validation (autograd computes d(yhat)/dt
        via the double-vjp trick, which works for any nn.Module and keeps the
        graph so the loss trains the network).
        """
        if self.random_collocation:
            tau = torch.rand_like(dt_years) * dt_years
        else:
            tau = dt_years
        tau = tau.detach().requires_grad_(True)

        pred = self.predict_log_abund(phys, y0, tau)
        # jvp via two vjps: first u^T J w.r.t. tau, then differentiate by u.
        u = torch.zeros_like(pred, requires_grad=True)
        vjp = torch.autograd.grad(pred, tau, grad_outputs=u, create_graph=True)[0]
        dpred_dtau = torch.autograd.grad(
            vjp, u, grad_outputs=torch.ones_like(tau), create_graph=True
        )[0]

        pred = torch.clamp(pred, LOG_ABUND_MIN, LOG_ABUND_MAX)
        x_pred = torch.pow(10.0, pred)
        # d(log10 y)/d(tau_years) -> dy/dt in s^-1
        lhs = LN10 * x_pred * dpred_dtau / SECONDS_PER_YEAR

        T, nH, chi, Av, zeta = self._env(phys)
        prod, dest = self.rhs.production_destruction(x_pred, T, nH, chi, Av, zeta)
        denom = (
            lhs.abs() + prod + dest + x_pred * self.residual_rate_floor + EPS
        )
        residual = (lhs - (prod - dest)) / denom
        return residual.abs().mean()

    def conservation_loss(self, pred, y0):
        """Relative drift of charge/elemental invariants from t to t+dt.

        Normalized by the gross inventory |Q| @ x0 of each invariant, not by
        the (signed) invariant itself: the net charge is ~0 by neutrality and
        would otherwise blow up the ratio.
        """
        x_pred = torch.pow(10.0, torch.clamp(pred, LOG_ABUND_MIN, LOG_ABUND_MAX))
        x0 = torch.pow(10.0, torch.clamp(y0, LOG_ABUND_MIN, LOG_ABUND_MAX))
        drift = (x_pred - x0) @ self.cons_matrix.T
        scale = x0 @ self.cons_matrix.abs().T + EPS
        return (drift / scale).abs().mean()

    # ------------------------------------------------------------------ #
    # lightning steps                                                    #
    # ------------------------------------------------------------------ #
    def _step(self, batch, stage):
        inputs, targets = batch
        phys, y0, dt_years = self.split_inputs(inputs)
        preds = self.predict_log_abund(phys, y0, dt_years)
        data_loss = F.l1_loss(preds, targets)

        compute_physics = self.physics_weight > 0 and (
            stage == "train" or not self.trainer.sanity_checking
        )
        if compute_physics:
            with torch.enable_grad():
                physics_loss = self.physics_residual_loss(phys, y0, dt_years)
        else:
            physics_loss = torch.zeros((), device=data_loss.device)
        if self.conservation_weight > 0:
            cons_loss = self.conservation_loss(preds, y0)
        else:
            cons_loss = torch.zeros((), device=data_loss.device)

        loss = (
            data_loss
            + self.physics_weight * physics_loss
            + self.conservation_weight * cons_loss
        )

        metric = getattr(self, f"{stage}_mse")
        metric(preds, targets)
        self.log(f"{stage}_loss", data_loss, prog_bar=(stage != "test"),
                 on_step=False, on_epoch=True)
        self.log(f"{stage}_total_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_physics_loss", physics_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_conservation_loss", cons_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_mse", metric, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
