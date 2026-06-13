"""Pure-PyTorch xLSTM cells (Beck et al. 2024, "xLSTM: Extended Long Short-Term
Memory") in their *recurrent* (step-wise) form.

This is a dependency-free reimplementation rather than the official ``xlstm``
package: that package's sLSTM relies on custom CUDA/Triton kernels (no CPU/MPS
backend), and its block stack is built for whole-sequence processing. The
chemical-abundance surrogate instead threads a hidden state one timestep at a
time (the rollout curriculum in ``model.py`` feeds the model its own predictions
and carries the state across the window; ``test.py`` threads it across a full
tracer), so a recurrent step API is what is needed here.

Both novel cells from the paper are provided:

* :class:`sLSTMCell` -- scalar memory with exponential input/forget gating, a
  normalizer state, and recurrent (hidden-to-hidden) memory mixing. The
  exponential gates are evaluated in log space against a running stabilizer
  ``m`` so they never overflow (Eqs. 15-17 of the paper).
* :class:`mLSTMCell` -- matrix memory ``C`` updated with the outer-product
  covariance rule, a vector normalizer ``n``, and the same exponential-gate
  stabilizer. Written in the recurrent step form (not the parallel chunkwise
  kernel) so it composes with the autoregressive rollout.

Both are multi-head: the hidden dimension is split into ``num_heads`` equal
heads that each carry independent memory, matching the paper's head structure.
``xLSTMBlock`` wraps a cell with pre-LayerNorm, a residual connection, and a
position-wise feed-forward sublayer; ``xLSTMStack`` projects the input, stacks
blocks (optionally alternating cell kinds), and exposes ``step(x, state)``.
"""

import torch
from torch import nn


def _zeros(batch, *shape, device, dtype):
    return torch.zeros(batch, *shape, device=device, dtype=dtype)


class sLSTMCell(nn.Module):
    """Scalar-memory xLSTM cell with exponential gating and memory mixing.

    State is the tuple ``(c, n, h, m)`` with shapes ``[batch, hidden]`` except
    the stabilizer ``m`` which is ``[batch, num_heads]`` (one running maximum
    per head). ``forward(x, state)`` advances a single timestep and returns
    ``(h, new_state)`` where ``h`` is ``[batch, hidden]``.
    """

    def __init__(self, input_size, hidden_size, num_heads=1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        # Input projections for the four gates (cell z, input i, forget f,
        # output o) and the recurrent hidden-to-hidden memory mixing.
        self.weight_ih = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.weight_hh = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_state(self, batch, device, dtype):
        c = _zeros(batch, self.hidden_size, device=device, dtype=dtype)
        n = _zeros(batch, self.hidden_size, device=device, dtype=dtype)
        h = _zeros(batch, self.hidden_size, device=device, dtype=dtype)
        # Stabilizer starts at -inf so the first step's max() is driven purely
        # by that step's input gate pre-activation.
        m = torch.full(
            (batch, self.num_heads), float("-inf"), device=device, dtype=dtype
        )
        return (c, n, h, m)

    def _per_head(self, gate):
        """Reshape a ``[batch, hidden]`` gate to ``[batch, num_heads, head_dim]``."""
        return gate.view(-1, self.num_heads, self.head_dim)

    def forward(self, x, state):
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        c_prev, n_prev, h_prev, m_prev = state

        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        z_pre, i_pre, f_pre, o_pre = gates.chunk(4, dim=-1)

        z = torch.tanh(z_pre)
        o = torch.sigmoid(o_pre)

        # Exponential input/forget gates, stabilized per head in log space.
        # i_pre is the log input gate; the log forget gate is logsigmoid(f_pre)
        # (the paper allows exp or sigmoid for f -- the stabilized exp form is
        # used here). m_t = max(log_f + m_{t-1}, log_i) keeps exp() arguments
        # <= 0 so they cannot overflow.
        log_f = nn.functional.logsigmoid(f_pre)

        i_pre_h = self._per_head(i_pre)
        log_f_h = self._per_head(log_f)
        # Reduce the gate pre-activations to one scalar per head for the
        # stabilizer (max over the head's channels). The stabilizer tracks the
        # running max of the carried log-forget plus the current log-input, per
        # head, so the exp() arguments below stay <= 0 and cannot overflow.
        m_new = torch.maximum(log_f_h.amax(dim=-1) + m_prev, i_pre_h.amax(dim=-1))

        # Broadcast the per-head stabilizer back to per-channel.
        m_new_full = m_new.repeat_interleave(self.head_dim, dim=-1)
        m_prev_full = m_prev.repeat_interleave(self.head_dim, dim=-1)

        i = torch.exp(i_pre - m_new_full)
        f = torch.exp(log_f + m_prev_full - m_new_full)

        c = f * c_prev + i * z
        n = f * n_prev + i
        h_tilde = c / (n + 1e-8)
        h = o * h_tilde
        return h, (c, n, h, m_new)


class mLSTMCell(nn.Module):
    """Matrix-memory xLSTM cell (recurrent step form).

    Each head carries a memory matrix ``C`` of shape ``[head_dim, head_dim]``, a
    normalizer vector ``n`` of shape ``[head_dim]`` and a scalar stabilizer
    ``m``. State tuple is ``(C, n, m)`` batched on the leading axis. Query/key
    are scaled by ``1/sqrt(head_dim)`` as in scaled dot-product attention, which
    the mLSTM generalizes.
    """

    def __init__(self, input_size, hidden_size, num_heads=1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.input_size, self.hidden_size)
        self.k_proj = nn.Linear(self.input_size, self.hidden_size)
        self.v_proj = nn.Linear(self.input_size, self.hidden_size)
        self.o_proj = nn.Linear(self.input_size, self.hidden_size)
        # Scalar input/forget gate pre-activations, one per head.
        self.gate_proj = nn.Linear(self.input_size, 2 * self.num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_state(self, batch, device, dtype):
        c = _zeros(
            batch, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype,
        )
        n = _zeros(batch, self.num_heads, self.head_dim, device=device, dtype=dtype)
        m = _zeros(batch, self.num_heads, device=device, dtype=dtype)
        return (c, n, m)

    def forward(self, x, state):
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        c_prev, n_prev, m_prev = state
        batch = x.shape[0]

        def heads(proj):
            return proj(x).view(batch, self.num_heads, self.head_dim)

        q = heads(self.q_proj) * (self.head_dim ** -0.5)
        k = heads(self.k_proj)
        v = heads(self.v_proj)
        o = torch.sigmoid(self.o_proj(x)).view(batch, self.num_heads, self.head_dim)

        gate_pre = self.gate_proj(x).view(batch, self.num_heads, 2)
        i_pre, f_pre = gate_pre[..., 0], gate_pre[..., 1]
        log_f = nn.functional.logsigmoid(f_pre)

        # Stabilized exponential gates (scalar per head).
        m_new = torch.maximum(log_f + m_prev, i_pre)
        i = torch.exp(i_pre - m_new)
        f = torch.exp(log_f + m_prev - m_new)

        # Covariance update: C += i * (v outer k); n += i * k.
        vk = v.unsqueeze(-1) * k.unsqueeze(-2)  # [B, H, head_dim, head_dim]
        c = f.unsqueeze(-1).unsqueeze(-1) * c_prev + i.unsqueeze(-1).unsqueeze(-1) * vk
        n = f.unsqueeze(-1) * n_prev + i.unsqueeze(-1) * k

        # Readout: h = (C q) / max(|n . q|, exp(-m)).
        cq = torch.einsum("bhij,bhj->bhi", c, q)  # [B, H, head_dim]
        nq = torch.einsum("bhi,bhi->bh", n, q).abs()
        denom = torch.maximum(nq, torch.exp(-m_new)).clamp_min(1e-8)
        h = cq / denom.unsqueeze(-1)
        h = o * h
        h = h.reshape(batch, self.hidden_size)
        return h, (c, n, m_new)


CELLS = {"slstm": sLSTMCell, "mlstm": mLSTMCell}


class xLSTMBlock(nn.Module):
    """One residual xLSTM block: pre-norm + cell + residual, then pre-norm +
    position-wise feed-forward + residual."""

    def __init__(self, hidden_size, num_heads, kind, dropout=0.0, ffn_mult=4):
        super().__init__()
        kind = str(kind).lower()
        if kind not in CELLS:
            raise ValueError(f"cell kind must be one of {sorted(CELLS)}, got {kind!r}")
        self.kind = kind
        self.norm_cell = nn.LayerNorm(hidden_size)
        self.cell = CELLS[kind](hidden_size, hidden_size, num_heads=num_heads)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_mult * hidden_size),
            nn.GELU(),
            nn.Linear(ffn_mult * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state):
        h, new_state = self.cell(self.norm_cell(x), state)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, new_state


class xLSTMStack(nn.Module):
    """Input projection followed by ``num_blocks`` xLSTM blocks.

    ``cell_type`` is ``slstm`` | ``mlstm`` | ``mixed``; ``mixed`` alternates
    sLSTM and mLSTM blocks (sLSTM first). ``step(x, state)`` advances one
    timestep: ``x`` is ``[batch, num_inputs]``, ``state`` is the per-block list
    of cell states (``None`` to start from zeros). Returns ``(out, new_state)``
    with ``out`` of shape ``[batch, hidden_size]``.
    """

    def __init__(
        self,
        num_inputs,
        hidden_size,
        num_blocks,
        num_heads=1,
        cell_type="slstm",
        dropout=0.0,
    ):
        super().__init__()
        cell_type = str(cell_type).lower()
        if cell_type not in ("slstm", "mlstm", "mixed"):
            raise ValueError(
                f"cell_type must be 'slstm', 'mlstm', or 'mixed', got {cell_type!r}"
            )
        self.hidden_size = int(hidden_size)
        self.num_blocks = int(num_blocks)
        self.cell_type = cell_type
        self.input_proj = nn.Linear(int(num_inputs), self.hidden_size)
        kinds = []
        for idx in range(self.num_blocks):
            if cell_type == "mixed":
                kinds.append("slstm" if idx % 2 == 0 else "mlstm")
            else:
                kinds.append(cell_type)
        self.blocks = nn.ModuleList(
            xLSTMBlock(self.hidden_size, num_heads, kind, dropout=dropout)
            for kind in kinds
        )

    def step(self, x, state):
        if state is None:
            state = [None] * self.num_blocks
        h = self.input_proj(x)
        new_state = []
        for block, block_state in zip(self.blocks, state):
            h, s = block(h, block_state)
            new_state.append(s)
        return h, new_state

    forward = step
