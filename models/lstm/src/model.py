import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim


LOSS_FUNCTIONS = {
    "l1": F.l1_loss,
    "mse": F.mse_loss,
    "smooth_l1": F.smooth_l1_loss,
}

# This surrogate is locked to an LSTM cell. The config key ``rnn_cell_type`` is
# kept for backward compatibility but only accepts "lstm".
RNN_CELLS = {"lstm": nn.LSTM}


class LSTM(pl.LightningModule):
    """LSTM one-step surrogate with carried hidden state.

    Input:  [physical parameters at t, log10 abundances at t]
    Output: log10 abundances at t + 1
    Loss:   selectable via config["loss_function"] (l1 | mse | smooth_l1),
            always computed on the log10 abundances. Default: l1.

    Unlike the stateless MLP, the model maintains a recurrent hidden state that
    is threaded across the trajectory, so each prediction is conditioned on
    history rather than only the current state. Training unrolls the model
    autoregressively for config["rollout_steps"] steps: the true physical
    drivers are fed at every step, but the abundance input from step 2 onward is
    the model's own prediction, with gradients (and the hidden state) flowing
    through the whole chain. Step j (0-based) is weighted by
    config["rollout_decay_base"]**j. The training horizon follows a doubling
    curriculum (1, 2, 4, ... capped at rollout_steps), advancing every
    config["rollout_curriculum_epochs"] epochs; validation and test always use
    the full horizon so their loss keeps one definition across epochs. The
    hidden state starts at zero for each (short, within-tracer) window; at test
    time it is threaded across the full tracer.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        num_inputs = int(config["num_inputs"])
        hidden_dim = int(config["rnn_hidden_dim"])
        num_layers = int(config["rnn_num_layers"])
        output_size = int(config["output_size"])
        # Architecture is locked to LSTM; the only accepted cell type is "lstm".
        self.cell_type = str(config.get("rnn_cell_type", "lstm")).lower()
        dropout = float(config.get("rnn_dropout", 0.0))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.loss_function = str(config.get("loss_function", "l1"))
        # LR schedule: "none" | "cosine" | "plateau". L1/Huber losses have a
        # non-vanishing gradient near the optimum, so a fixed LR makes the
        # weights dither at step size ~lr forever — visible as a noisy
        # validation curve. Decaying the LR lets the optimizer settle.
        self.lr_scheduler = str(config.get("lr_scheduler", "none"))
        self.lr_min = float(config.get("lr_min", 1e-6))
        self.lr_plateau_factor = float(config.get("lr_plateau_factor", 0.5))
        self.lr_plateau_patience = int(config.get("lr_plateau_patience", 3))
        # Trace-species downweighting: loss elements whose target log10
        # abundance is at or below trace_threshold_log10 are scaled by
        # trace_weight (1.0 disables).
        self.trace_threshold_log10 = float(
            config.get("trace_threshold_log10", -float("inf"))
        )
        self.trace_weight = float(config.get("trace_weight", 1.0))
        # Multi-step rollout training (see class docstring).
        self.rollout_steps = max(1, int(config.get("rollout_steps", 1)))
        self.rollout_decay_base = float(config.get("rollout_decay_base", 0.5))
        self.rollout_curriculum_epochs = int(
            config.get("rollout_curriculum_epochs", 0)
        )
        if self.loss_function not in LOSS_FUNCTIONS:
            raise ValueError(
                f"loss_function must be one of {sorted(LOSS_FUNCTIONS)}, "
                f"got {self.loss_function!r}"
            )
        if self.cell_type not in RNN_CELLS:
            raise ValueError(
                f"rnn_cell_type must be one of {sorted(RNN_CELLS)}, "
                f"got {self.cell_type!r}"
            )

        # Physical-parameter normalization. Stats are computed on the training
        # split (see data.GravCollapseDataModule.phys_norm_config) and stored
        # in the saved hyperparameters, so they travel with the checkpoint.
        self.num_phys = int(config.get("num_phys", 0))
        log_mask = config.get("phys_log_mask") or [0.0] * self.num_phys
        phys_mean = config.get("phys_mean") or [0.0] * self.num_phys
        phys_std = config.get("phys_std") or [1.0] * self.num_phys
        phys_log_floor = config.get("phys_log_floor") or [1e-30] * self.num_phys
        self.register_buffer(
            "phys_log_mask", torch.tensor(log_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "phys_mean", torch.tensor(phys_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "phys_std", torch.tensor(phys_std, dtype=torch.float32)
        )
        self.register_buffer(
            "phys_log_floor", torch.tensor(phys_log_floor, dtype=torch.float32)
        )

        # PyTorch applies inter-layer dropout only when num_layers > 1.
        self.rnn = RNN_CELLS[self.cell_type](
            input_size=num_inputs,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Linear(hidden_dim, output_size)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def _normalize_phys(self, x):
        """Apply log10 (multi-decade cols) + standardization to the leading
        physical-parameter columns, leaving the log10 abundances untouched."""
        n = self.num_phys
        if n == 0:
            return x
        phys, rest = x[:, :n], x[:, n:]
        mask = self.phys_log_mask.bool()
        if mask.any():
            logged = torch.log10(torch.maximum(phys, self.phys_log_floor))
            phys = torch.where(mask, logged, phys)
        phys = (phys - self.phys_mean) / self.phys_std
        return torch.cat([phys, rest], dim=1)

    def step(self, x, hidden):
        """Advance one timestep. ``x`` is [batch, num_inputs]; ``hidden`` is the
        recurrent state (None to start from zeros). Returns the predicted log10
        abundances [batch, output_size] and the updated hidden state."""
        x = self._normalize_phys(x)
        out, hidden = self.rnn(x.unsqueeze(1), hidden)  # out: [batch, 1, hidden]
        return self.head(out.squeeze(1)), hidden

    def forward(self, x):
        """Single-step convenience map from a zero hidden state."""
        return self.step(x, None)[0]

    def _weighted_loss(self, preds, targets):
        if self.trace_weight != 1.0:
            elementwise = LOSS_FUNCTIONS[self.loss_function](
                preds, targets, reduction="none"
            )
            weights = torch.where(
                targets <= self.trace_threshold_log10,
                torch.as_tensor(self.trace_weight, device=targets.device),
                torch.ones((), device=targets.device),
            )
            return (elementwise * weights).sum() / weights.sum()
        return LOSS_FUNCTIONS[self.loss_function](preds, targets)

    def _horizon(self, stage):
        """Rollout horizon for this step. Training follows the doubling
        curriculum; val/test always use the full horizon so their loss
        keeps a single definition across epochs."""
        if stage != "train" or self.rollout_curriculum_epochs <= 0:
            return self.rollout_steps
        return min(
            self.rollout_steps,
            2 ** (self.current_epoch // self.rollout_curriculum_epochs),
        )

    def _step(self, batch, stage):
        phys_seq, abund, targets = batch
        horizon = self._horizon(stage)
        metric = getattr(self, f"{stage}_mse")
        hidden = None
        loss = abund.new_zeros(())
        weight_sum = 0.0
        for j in range(horizon):
            preds, hidden = self.step(torch.cat([phys_seq[:, j], abund], dim=1), hidden)
            # The window slice is non-contiguous; torchmetrics needs .view().
            step_targets = targets[:, j].contiguous()
            step_weight = self.rollout_decay_base**j
            loss = loss + step_weight * self._weighted_loss(preds, step_targets)
            weight_sum += step_weight
            metric(preds, step_targets)
            abund = preds
        loss = loss / weight_sum
        if stage == "train":
            self.log("train_rollout_k", float(horizon), on_step=False, on_epoch=True)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage != "test"),
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{stage}_mse", metric, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler == "none":
            return optimizer
        if self.lr_scheduler == "cosine":
            # Anneal over the full run; T_max comes from the trainer.
            t_max = getattr(self.trainer, "max_epochs", None) or 100
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(t_max), eta_min=self.lr_min
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        if self.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_plateau_factor,
                patience=self.lr_plateau_patience,
                min_lr=self.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        raise ValueError(
            f"lr_scheduler must be 'none', 'cosine', or 'plateau', "
            f"got {self.lr_scheduler!r}"
        )
