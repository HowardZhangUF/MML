"""Configuration loading and checkpoint selection shared by MML callers."""

import os
import sys
from pathlib import Path

import yaml


def project_root():
    """Simulator resource root; override when using installed wheels elsewhere."""
    override = os.environ.get("MML_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "mml_python_sim/pyproject.toml").is_file():
            return parent / "mml_python_sim"
    return Path(sys.prefix) / "share/mml-python-sim"


def artifact_root():
    """External checkpoints/data; configurable independently of package install."""
    override = os.environ.get("MML_ARTIFACT_ROOT")
    return Path(override).expanduser().resolve() if override else project_root().parent / "artifacts"


def motion_resource_dir():
    return artifact_root() / "motion"


def ros_package_dir():
    """Locate ROS resources without importing ROS."""
    prefixes = [Path(p) / "share/mml_guidance"
                for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    candidates = [project_root().parent / "mml_ros_ws/src/mml_guidance",
                  *prefixes, Path(sys.prefix) / "share/mml_guidance"]
    return next((p for p in candidates if p.is_dir()), candidates[0])


def research_data_dir():
    return motion_resource_dir() / "mml_guidance/sim_data/training_data"


def load_config(file_path=None):
    """Load an explicit configuration or the packaged, unchanged default."""
    with open(file_path if file_path is not None else Path(__file__).with_name("config.yaml")) as f:
        return yaml.safe_load(f)

def resolve_model_path(project_root: str,
                       prediction_method: str,
                       user_model_path: str | None = None) -> str | None:
    """
    Choose the correct model checkpoint for the selected prediction method.

    CFM:
        uses the original anchor-sigma checkpoint.

    PF+CFM / PF+CFM-AS:
        use the sigma=1.0 CFM checkpoint. PF+CFM-AS changes only the
        inference-time proposal spread after loading this velocity field.

    PF+NF / PF+Diffusion / DiffPF:
        use their trained standalone benchmark checkpoints.

    MultiPFTra / Transformer:
        uses the original ScratchTransformer checkpoint.

    MultiPFVel:
        no neural model checkpoint is needed.
    """
    if user_model_path is not None:
        return user_model_path

    package_dir = str(motion_resource_dir())
    model_dir = os.path.join(package_dir, "models")

    if prediction_method == "CFM":
        return os.path.join(
            model_dir,
            "cfm_motion_model.pth",
        )

    if prediction_method in {"PF+CFM", "PF+CFM-AS"}:
        return os.path.join(model_dir, "cfm_motion_model_anchor1.0.pth")

    if prediction_method == "PF+NF":
        return os.path.join(model_dir, "nf_motion_model.pth")

    if prediction_method == "PF+Diffusion":
        return os.path.join(model_dir, "diffusion_motion_model.pth")

    if prediction_method == "DiffPF":
        return os.path.join(model_dir, "diffpf_tracker.pth")

    if prediction_method in {"MultiPFTra", "Transformer"}:
        candidates = [
            os.path.join(
                package_dir,
                "mml_guidance",
                "mml_network",
                "models",
                "best_noisy_5_v2_velocities_occlusions_ScratchTransformer.pth",
            ),
            os.path.join(
                str(artifact_root() / "simulator"),
                "best_noisy_5_v2_velocities_occlusions_ScratchTransformer.pth",
            ),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # Return the first expected path even if missing so the error message
        # points to the most likely location.
        return candidates[0]

    # MultiPFVel / velocity baseline does not need a checkpoint.
    return None
