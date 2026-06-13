"""Fast unit checks for the vendored xLSTM cells and the LightningModule.

No training run and no dataset: these exercise shapes, state threading, the
numerical stabilizer, and a single backward pass. Run with:

    cd models/xlstm/src && python -m pytest ../tests/test_xlstm.py -q

(``src`` on ``sys.path`` is required because the package uses sibling imports;
the path insert below makes the file runnable from anywhere too.)
"""

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xlstm_cells import xLSTMStack  # noqa: E402
from model import XLSTM  # noqa: E402


# The surrogate is autoregressive: its prediction becomes the next step's
# abundance input, so output_size must equal the abundance-input dimension
# (num_inputs - num_phys). With num_phys = 2 below, NUM_INPUTS - 2 == OUTPUT.
HIDDEN = 16
NUM_BLOCKS = 2
NUM_HEADS = 4
BATCH = 5
OUTPUT = 7
NUM_INPUTS = OUTPUT + 2


@pytest.mark.parametrize("cell_type", ["slstm", "mlstm", "mixed"])
def test_stack_step_shapes_and_state(cell_type):
    stack = xLSTMStack(
        NUM_INPUTS, HIDDEN, NUM_BLOCKS, num_heads=NUM_HEADS, cell_type=cell_type
    )
    x = torch.randn(BATCH, NUM_INPUTS)
    out, state = stack.step(x, None)
    assert out.shape == (BATCH, HIDDEN)
    assert len(state) == NUM_BLOCKS
    # A second step with the carried state must round-trip shapes.
    out2, state2 = stack.step(x, state)
    assert out2.shape == (BATCH, HIDDEN)
    assert len(state2) == NUM_BLOCKS


@pytest.mark.parametrize("cell_type", ["slstm", "mlstm", "mixed"])
def test_state_actually_carries(cell_type):
    """Threading state through two steps must differ from two independent
    zero-state steps -- otherwise the hidden state is inert."""
    torch.manual_seed(0)
    stack = xLSTMStack(
        NUM_INPUTS, HIDDEN, NUM_BLOCKS, num_heads=NUM_HEADS, cell_type=cell_type
    )
    x = torch.randn(BATCH, NUM_INPUTS)
    _, state = stack.step(x, None)
    threaded, _ = stack.step(x, state)
    fresh, _ = stack.step(x, None)
    assert not torch.allclose(threaded, fresh, atol=1e-5)


@pytest.mark.parametrize("cell_type", ["slstm", "mlstm", "mixed"])
def test_stabilizer_keeps_outputs_finite(cell_type):
    """Large-magnitude inputs must not overflow the exponential gates."""
    stack = xLSTMStack(
        NUM_INPUTS, HIDDEN, NUM_BLOCKS, num_heads=NUM_HEADS, cell_type=cell_type
    )
    x = torch.randn(BATCH, NUM_INPUTS) * 50.0
    out, state = stack.step(x, None)
    out2, _ = stack.step(x, state)
    assert torch.isfinite(out).all()
    assert torch.isfinite(out2).all()


def test_invalid_cell_type_rejected():
    with pytest.raises(ValueError):
        xLSTMStack(NUM_INPUTS, HIDDEN, NUM_BLOCKS, cell_type="bogus")


def test_hidden_not_divisible_by_heads_rejected():
    with pytest.raises(ValueError):
        xLSTMStack(NUM_INPUTS, hidden_size=10, num_blocks=1, num_heads=4)


def _tiny_model_config(cell_type="slstm", num_phys=2):
    return {
        "num_inputs": NUM_INPUTS,
        "output_size": OUTPUT,
        "xlstm_num_blocks": NUM_BLOCKS,
        "xlstm_hidden_dim": HIDDEN,
        "xlstm_num_heads": NUM_HEADS,
        "xlstm_cell_type": cell_type,
        "xlstm_dropout": 0.0,
        "learning_rate": 1e-3,
        "loss_function": "smooth_l1",
        "trace_threshold_log10": -20.0,
        "trace_weight": 0.1,
        "rollout_steps": 3,
        "rollout_decay_base": 0.5,
        "rollout_curriculum_epochs": 0,
        "lr_scheduler": "none",
        "num_phys": num_phys,
        "phys_log_mask": [1.0, 0.0],
        "phys_mean": [0.0, 0.0],
        "phys_std": [1.0, 1.0],
        "phys_log_floor": [1e-30, 1e-30],
    }


@pytest.mark.parametrize("cell_type", ["slstm", "mlstm", "mixed"])
def test_model_forward_shape(cell_type):
    model = XLSTM(_tiny_model_config(cell_type))
    x = torch.randn(BATCH, NUM_INPUTS)
    out = model(x)
    assert out.shape == (BATCH, OUTPUT)


@pytest.mark.parametrize("cell_type", ["slstm", "mlstm", "mixed"])
def test_model_step_loss_and_backprop(cell_type):
    """A rollout training step returns a finite scalar loss and produces grads."""
    num_phys = 2
    abund_dim = NUM_INPUTS - num_phys
    model = XLSTM(_tiny_model_config(cell_type, num_phys=num_phys))
    rollout = model.rollout_steps
    phys_seq = torch.randn(BATCH, rollout, num_phys)
    abund0 = torch.randn(BATCH, abund_dim)
    targets = torch.randn(BATCH, rollout, OUTPUT)
    loss = model._step((phys_seq, abund0, targets), "train")
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed to model parameters"
    assert all(torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
