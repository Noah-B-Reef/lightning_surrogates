import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim


class MLP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        num_inputs = int(config["num_inputs"])
        hidden_layers = int(config["num_hidden_layers"])
        hidden_units = int(config["num_neurons_per_hidden_layer"])
        output_size = int(config["output_size"])
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.rollout_steps = int(config.get("rollout_steps", 1))

        # Physical-parameter normalization. Stats are computed on the training
        # split (see data.GravCollapseDataModule.phys_norm_config) and stored
        # in the saved hyperparameters, so they travel with the checkpoint and
        # are applied identically during autoregressive rollout. Defaults make
        # this a no-op for checkpoints trained before normalization existed.
        self.num_phys = int(config.get("num_phys", 0))
        log_mask = config.get("phys_log_mask") or [0.0] * self.num_phys
        phys_mean = config.get("phys_mean") or [0.0] * self.num_phys
        phys_std = config.get("phys_std") or [1.0] * self.num_phys
        self.register_buffer(
            "phys_log_mask", torch.tensor(log_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "phys_mean", torch.tensor(phys_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "phys_std", torch.tensor(phys_std, dtype=torch.float32)
        )

        layers = [
            nn.Linear(num_inputs, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.ReLU(),
        ]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend(
                [
                    nn.Linear(hidden_units, hidden_units),
                    nn.LayerNorm(hidden_units),
                    nn.ReLU(),
                ]
            )
        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_units, output_size)

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
            logged = torch.log10(torch.clamp(phys, min=1e-30))
            phys = torch.where(mask, logged, phys)
        phys = (phys - self.phys_mean) / self.phys_std
        return torch.cat([phys, rest], dim=1)

    def forward(self, x):
        x = self._normalize_phys(x)
        return self.output_proj(self.network(x))

    def _rollout(self, initial, phys_seq):
        current = initial[:, self.num_phys:]
        preds = []
        steps = phys_seq.shape[1]
        for step in range(steps):
            x = torch.cat([phys_seq[:, step, :], current], dim=1)
            current = self(x)
            preds.append(current)
        return torch.stack(preds, dim=1)

    def _masked_rollout_loss(self, preds, targets, mask):
        per_value = F.smooth_l1_loss(preds, targets, reduction="none")
        per_step = per_value.mean(dim=-1)
        masked = per_step * mask
        return masked.sum() / mask.sum().clamp_min(1.0)

    def _step(self, batch, stage):
        if len(batch) == 2:
            initial, targets = batch
            preds = self(initial)
            loss = F.smooth_l1_loss(preds, targets)
            metric = getattr(self, f"{stage}_mse")
            metric(preds, targets)
        else:
            initial, phys_seq, targets, mask = batch
            preds = self._rollout(initial, phys_seq)
            loss = self._masked_rollout_loss(preds, targets, mask)
            valid = mask.bool()
            metric = getattr(self, f"{stage}_mse")
            metric(preds[valid], targets[valid])
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "test"))
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
