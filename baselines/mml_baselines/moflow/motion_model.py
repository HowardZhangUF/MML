"""MoFlow (IMLE one-step student) as a bootstrap-PF motion model."""

from __future__ import annotations

from mml_baselines.common.data import MinMax
from mml_baselines.common.scaffold import TrajectoryMotionModel
from mml_baselines.moflow.model import MoFlowIMLE, MoFlowIMLENet, MoFlowTeacher, MoFlowTeacherNet


def _net_kwargs(c):
    return dict(num_query=c["num_query"], d=c["d_model"], social_hidden=c["social_hidden"],
                enc_layers=c["enc_layers"], dec_blocks=c["dec_blocks"], heads=c["heads"], dropout=c["dropout"])


def build_teacher(config: dict, q_norm: MinMax) -> MoFlowTeacher:
    return MoFlowTeacher(MoFlowTeacherNet(config["horizon"], **_net_kwargs(config)), q_norm,
                         config["horizon"], config["num_query"], config["sampling_steps"])


def build_student(config: dict, q_norm: MinMax) -> MoFlowIMLE:
    return MoFlowIMLE(MoFlowIMLENet(config["horizon"], **_net_kwargs(config)), q_norm,
                      config["horizon"], config["num_to_gen"])


class MoFlowMotionModel(TrajectoryMotionModel):
    """Draws one of the student's K set members uniformly (the IMLE set has no scores)."""

    def build(self, ckpt):
        model = build_student(self.config, self.q_norm)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def forecast(self, hist):
        return self.model.sample(hist)
