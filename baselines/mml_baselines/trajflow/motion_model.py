"""TrajFlow as a bootstrap-PF motion model: sample a mode by its sigmoid score, propose its first step."""

from __future__ import annotations

import numpy as np

from mml_baselines.common.data import MinMax
from mml_baselines.common.scaffold import TrajectoryMotionModel
from mml_baselines.trajflow.model import TrajFlow, TrajFlowNet


def build_trajflow(config: dict, intention_points, q_norm: MinMax) -> TrajFlow:
    net = TrajFlowNet(config["horizon"], np.asarray(intention_points), d_model=config["d_model"],
                      heads=config["heads"], depth=config["depth"], dropout=config["dropout"],
                      n_th=config.get("block_size", 5))
    return TrajFlow(net, q_norm, num_motion_modes=config["num_motion_modes"],
                    nms_dist=config["nms_dist"], num_inter_layers=config["num_inter_layers"],
                    sampling_steps=config["sampling_steps"])


class TrajFlowMotionModel(TrajectoryMotionModel):
    def build(self, ckpt):
        model = build_trajflow(self.config, ckpt["intention_points"], self.q_norm)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def forecast(self, hist):
        trajs, scores = self.model.sample(hist)
        return trajs, scores.clamp_min(1e-12)
