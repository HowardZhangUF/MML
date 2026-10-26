"""
flow_density.py
===============
Exact log-densities of Euler-discretised 2-D conditional flows.

A flow-matching sampler with N_FE Euler steps is the map

    x_{s+1} = x_s + delta * u(x_s, tau_s),   tau_s = s * delta,   delta = 1 / N_FE.

Each step is a bijection of R^2 (as long as det(I + delta * du/dx) > 0), so the
sampler defines a proper density by change of variables:

    log p(x_N) = log p_0(x_0) - sum_s log |det(I + delta * J_s)|,

with J_s the 2x2 Jacobian of u at x_s. This is the density of the samples the
filter actually draws, not of the continuous-time ODE, so importance weights
built from it are exact for the sampler (no ODE-solver bias).

MLPField        u(x, tau) = MLP(cat[cond, x, tau]) with an analytic Jacobian;
                the conditioning term of the first layer is computed once.
euler_forward   pushes x_0 -> x_N and accumulates the log-determinant.
euler_inverse   recovers x_0 from x_N by Newton's method on every step
                (needed to evaluate a density at a point the flow did not
                generate, e.g. the motion-model density at a proposal sample).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _act_and_grad(act: nn.Module, h: torch.Tensor):
    if isinstance(act, nn.ReLU):
        return torch.relu(h), (h > 0).to(h.dtype)
    if isinstance(act, nn.SiLU):
        s = torch.sigmoid(h)
        return h * s, s * (1.0 + h * (1.0 - s))
    if isinstance(act, nn.Tanh):
        t = torch.tanh(h)
        return t, 1.0 - t * t
    raise TypeError(f"Unsupported activation {type(act).__name__}")


class MLPField:
    """
    Vector field u(x, tau) = mlp(cat[cond, x, tau]) for (B, 2) states x and
    fixed (B, E) conditioning; mlp alternates Linear and activation modules.
    """

    def __init__(self, mlp: nn.Sequential, cond: torch.Tensor):
        layers = list(mlp)
        first = layers[0]
        E = cond.shape[1]
        self.base = cond @ first.weight[:, :E].T + first.bias       # (B, H)
        self.w_x = first.weight[:, E:E + 2]                          # (H, 2)
        self.w_t = first.weight[:, E + 2]                            # (H,)
        self.rest = layers[1:]                                       # act, Linear, act, ...

    def _pre(self, x, tau):
        return self.base + x @ self.w_x.T + tau * self.w_t

    def __call__(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        h = self._pre(x, tau)
        for layer in self.rest:
            h = layer(h)
        return h

    def value_and_jacobian(self, x: torch.Tensor, tau: float):
        """u (B, 2) and du/dx (B, 2, 2)."""
        h = self._pre(x, tau)
        J = self.w_x.expand(x.shape[0], -1, -1)                      # (B, H, 2)
        for layer in self.rest:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                J = torch.einsum("oh,bhk->bok", layer.weight, J)
            else:
                h, g = _act_and_grad(layer, h)
                J = g.unsqueeze(-1) * J
        return h, J


def _step(field, x, tau, delta):
    """One Euler step g(x) = x + delta * u(x) and its Jacobian I + delta * du/dx."""
    u, J = field.value_and_jacobian(x, tau)
    J = delta * J
    J[:, 0, 0] += 1.0
    J[:, 1, 1] += 1.0
    return x + delta * u, J


def _logabsdet(J: torch.Tensor) -> torch.Tensor:
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    return torch.log(det.abs().clamp_min(1e-30))


def _solve(J: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Batched 2x2 solve J dx = r."""
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    det = a * d - b * c
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    return torch.stack([(d * r[:, 0] - b * r[:, 1]) / det,
                        (a * r[:, 1] - c * r[:, 0]) / det], dim=-1)


def euler_forward(field, x0: torch.Tensor, n_steps: int):
    """Push x0 through the Euler flow; returns (x_N, sum_s log|det step Jacobian|)."""
    delta = 1.0 / n_steps
    x = x0
    logdet = torch.zeros(x0.shape[0], dtype=x0.dtype, device=x0.device)
    for s in range(n_steps):
        x, J = _step(field, x, s * delta, delta)
        logdet = logdet + _logabsdet(J)
    return x, logdet


def euler_inverse(field, y: torch.Tensor, n_steps: int, iters: int = 20, tol: float = 1e-5):
    """
    Invert the Euler flow: find x0 with euler_forward(x0) = y.

    Returns (x0, sum_s log|det step Jacobian|, max Newton residual). The
    residual is |g(x_s) - x_{s+1}| of the worst step and row; a value well
    above tol means some row did not converge (its density is unreliable).
    """
    delta = 1.0 / n_steps
    x_next = y
    logdet = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    worst = 0.0
    for s in reversed(range(n_steps)):
        tau = s * delta
        x = x_next - delta * field(x_next, tau)          # fixed-point initial guess
        for _ in range(iters):
            g, J = _step(field, x, tau, delta)
            r = g - x_next
            if float(r.abs().max()) < tol:
                break
            x = x - _solve(J, r)
        g, J = _step(field, x, tau, delta)
        worst = max(worst, float((g - x_next).abs().max()))
        logdet = logdet + _logabsdet(J)
        x_next = x
    return x_next, logdet, worst


def gaussian_log_prob(x: torch.Tensor, mean, sigma: float) -> torch.Tensor:
    """log N(x; mean, sigma^2 I) for (B, 2) rows."""
    d2 = ((x - mean) ** 2).sum(-1)
    return -0.5 * d2 / sigma ** 2 - 2.0 * math.log(sigma) - math.log(2.0 * math.pi)
