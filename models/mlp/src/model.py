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


class MLP(pl.LightningModule):
    """Simple MLP one-step surrogate.

    Input:  [physical parameters at t, log10 abundances at t]
    Output: log10 abundances at t + 1
    Loss:   selectable via config["loss_function"] (l1 | mse | smooth_l1),
            always computed on the log10 abundances. Default: l1.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        num_inputs = int(config["num_inputs"])
        hidden_layers = int(config["num_hidden_layers"])
        hidden_units = int(config["num_neurons_per_hidden_layer"])
        output_size = int(config["output_size"])
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.loss_function = str(config.get("loss_function", "l1"))
        if self.loss_function not in LOSS_FUNCTIONS:
            raise ValueError(
                f"loss_function must be one of {sorted(LOSS_FUNCTIONS)}, "
                f"got {self.loss_function!r}"
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

        layers = [nn.Linear(num_inputs, hidden_units), nn.ReLU()]
        for _ in range(max(0, hidden_layers - 1)):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, output_size))
        self.network = nn.Sequential(*layers)

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

    def forward(self, x):
        return self.network(self._normalize_phys(x))

    def _step(self, batch, stage):
        inputs, targets = batch
        preds = self(inputs)
        loss = LOSS_FUNCTIONS[self.loss_function](preds, targets)
        metric = getattr(self, f"{stage}_mse")
        metric(preds, targets)
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
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
