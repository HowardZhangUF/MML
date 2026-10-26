"""TCFM as a bootstrap-PF motion model: one flow sample per particle, propose its first step."""

from __future__ import annotations

from mml_baselines.common.data import MinMax
from mml_baselines.common.scaffold import TrajectoryMotionModel
from mml_baselines.tcfm.model import TCFM, ConditionalUnet1D


def build_tcfm(config: dict, q_norm: MinMax) -> TCFM:
    net = ConditionalUnet1D(dim=config["dim"], dim_mults=tuple(config["dim_mults"]),
                            lstm_out_dim=config["lstm_out_dim"])
    return TCFM(net, q_norm, config["horizon"], config["n_timesteps"])


class TCFMMotionModel(TrajectoryMotionModel):
    def build(self, ckpt):
        model = build_tcfm(self.config, self.q_norm)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def forecast(self, hist):
        return self.model.sample(hist)
