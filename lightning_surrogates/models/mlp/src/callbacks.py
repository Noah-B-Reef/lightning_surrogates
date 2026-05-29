import torch
import pytorch_lightning as pl


class RelativeImprovementEarlyStopping(pl.Callback):
    """Stop when a monitored metric fails to improve by a relative threshold."""

    def __init__(
        self,
        monitor="val_loss",
        min_relative_improvement=0.02,
        patience=8,
        mode="min",
        verbose=True,
    ):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.min_relative_improvement = min_relative_improvement
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.wait_count = 0

    def state_dict(self):
        return {"best_score": self.best_score, "wait_count": self.wait_count}

    def load_state_dict(self, state_dict):
        self.best_score = state_dict.get("best_score")
        self.wait_count = state_dict.get("wait_count", 0)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        if isinstance(current, torch.Tensor):
            current = current.detach()
            if not torch.isfinite(current):
                return
            current = current.item()

        if self.best_score is None:
            self.best_score = current
            return
        if self._is_improvement(current):
            self.best_score = current
            self.wait_count = 0
            return

        self.wait_count += 1
        if self.wait_count >= self.patience:
            trainer.should_stop = True
            if self.verbose and trainer.is_global_zero:
                print(
                    f"Stopping early: {self.monitor} did not improve by "
                    f"{self.min_relative_improvement:.1%} within {self.patience} epochs."
                )

    def _is_improvement(self, current):
        if self.mode == "min":
            return current <= self.best_score * (1.0 - self.min_relative_improvement)
        return current >= self.best_score * (1.0 + self.min_relative_improvement)
