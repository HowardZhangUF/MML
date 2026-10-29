"""
Training / inference scaffolding shared by the trajectory-forecasting ports.
Method logic (networks, losses, samplers) lives in each method's folder.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from mml_baselines.common.data import MinMax, build_trajectory_dataset, history_features
from mml_baselines.common.torch_utils import EMA, count_parameters, pick_device
from mml_research.motion.generate_synthetic_data import DT, N_TH


def load_split(horizon: int):
    """Normalized train/test tensors plus the fitted history / future normalizers."""
    d = build_trajectory_dataset(horizon=horizon)
    feats = history_features(d["X"])
    tr = d["train"]
    h_norm = MinMax().fit(feats[tr])
    q_norm = MinMax().fit(d["Q"][tr])
    split = {}
    for name, m in (("train", tr), ("test", ~tr)):
        split[name] = {
            "hist": torch.from_numpy(h_norm.norm(feats[m]).astype(np.float32)),
            "q": torch.from_numpy(d["Q"][m]),
            "v": torch.from_numpy(d["V"][m]),
        }
    return split, h_norm, q_norm


def train_loop(model, loss_fn, data, *, steps, batch, lr, weight_decay, ema_beta, grad_clip,
               device, log, warmup=0, cosine=False, eval_fn=None, eval_every=None):
    """
    AdamW + optional linear warmup / cosine decay + EMA; loss_fn(model, batch_dict) -> scalar.
    Returns (ema model, list of (step, train loss)).
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(step):
        f = min(1.0, (step + 1) / warmup) if warmup else 1.0
        if cosine:
            f *= 0.5 * (1 + np.cos(np.pi * step / steps))
        return lr * f

    ema = EMA(model, ema_beta)
    n = len(data["hist"])
    gen = torch.Generator().manual_seed(0)
    history, run, t0 = [], 0.0, time.time()
    eval_every = eval_every or max(1, steps // 10)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        b = {k: v[idx].to(device) for k, v in data.items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = loss_fn(model, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ema.update(model)
        run += loss.item()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            msg = f"step {step + 1:6d}/{steps}  loss {run / eval_every:9.4f}  {time.time() - t0:6.0f}s"
            if eval_fn is not None:
                msg += "  " + eval_fn(ema.model)
            log(msg)
            history.append((step + 1, run / eval_every))
            run = 0.0
    return ema.model.eval(), history


@torch.no_grad()
def forecast_metrics(sample_fn, data, device, n=2000, chunk=500):
    """
    minADE / minFDE over the returned modes (m), and the RMSE of the
    first-step velocity of a mode drawn by sample_fn's probabilities (m/s),
    the quantity the particle filter proposes.
    """
    hist, q = data["hist"][:n].to(device), data["q"][:n].to(device)
    ade, fde, vel = [], [], []
    for s in range(0, len(hist), chunk):
        trajs, probs = sample_fn(hist[s:s + chunk])          # [B, M, H, 2], [B, M]
        gt = q[s:s + chunk]
        err = (trajs - gt[:, None]).norm(dim=-1)
        ade.append(err.mean(-1).min(-1).values)
        fde.append(err[..., -1].min(-1).values)
        pick = torch.multinomial(probs, 1).squeeze(-1)
        v = trajs[torch.arange(len(gt), device=device), pick, 0] / DT
        vel.append(((v - gt[:, 0] / DT) ** 2).sum(-1))
    ade, fde, vel = torch.cat(ade), torch.cat(fde), torch.cat(vel)
    return {"minADE": ade.mean().item(), "minFDE": fde.mean().item(),
            "vel_rmse": vel.mean().sqrt().item(), "modes": trajs.shape[1]}


def save_checkpoint(path, model, config, h_norm, q_norm, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict(),
                "h_norm": h_norm.state(), "q_norm": q_norm.state(), **(extra or {})}, path)


class TrajectoryMotionModel:
    """
    Bootstrap-PF motion-model interface (propose / block_size / dt), shared by
    the forecaster ports: each particle's 5-step history -> the method's
    forecast set -> one trajectory drawn by the method's own mode
    probabilities -> its first-step velocity.
    """

    name = "trajectory"

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.h_norm = MinMax(**ckpt["h_norm"])
        self.q_norm = MinMax(**ckpt["q_norm"])
        self.device = torch.device(device)
        self.block_size = int(self.config.get("block_size", N_TH))   # absent in simulator checkpoints
        self.dt = float(self.config.get("dt", DT))
        self.model = self.build(ckpt).to(self.device).eval()
        self.n_params = count_parameters(self.model)
        print(f"[{type(self).__name__}] loaded <- {checkpoint_path}  ({self.n_params:,} params, "
              f"config={json.dumps(self.config)})")

    def build(self, ckpt):
        raise NotImplementedError

    def forecast(self, hist: torch.Tensor):
        """[B, P, 6] normalized history -> (trajs [B, M, H, 2] metric, probs [B, M])."""
        raise NotImplementedError

    @torch.no_grad()
    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        feats = self.h_norm.norm(history_features(c_flat, self.block_size, self.dt))
        hist = torch.from_numpy(feats.astype(np.float32)).to(self.device)
        trajs, probs = self.forecast(hist)
        pick = torch.multinomial(probs, 1).squeeze(-1)
        first = trajs[torch.arange(len(hist), device=self.device), pick, 0]
        return (first / self.dt).cpu().numpy().astype(np.float32)


__all__ = ["load_split", "train_loop", "forecast_metrics", "save_checkpoint",
           "TrajectoryMotionModel", "pick_device", "count_parameters"]
