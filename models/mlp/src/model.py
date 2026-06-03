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

    def forward(self, x):
        return self.output_proj(self.network(x))

    def _step(self, batch, stage):
        initial, targets = batch
        preds = self(initial)
        loss = F.smooth_l1_loss(preds, targets)
        metric = getattr(self, f"{stage}_mse")
        metric(preds, targets)
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
