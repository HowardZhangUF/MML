"""Motion models and loaders; implementations are imported on demand."""

from importlib import import_module

_EXPORTS = {'CFMMotionModel': 'mml_core.motion.cfm_motion_model',
 'CFMShim': 'mml_core.motion.cfm_motion_model',
 'load_cfm_model': 'mml_core.motion.cfm_motion_model',
 'DiffusionMotionModel': 'mml_core.motion.diffusion_motion_model',
 'load_diffusion_model': 'mml_core.motion.diffusion_motion_model',
 'NFMotionModel': 'mml_core.motion.nf_motion_model',
 'load_nf_model': 'mml_core.motion.nf_motion_model',
 'GMMMotionModel': 'mml_core.motion.gmm_motion_model',
 'load_gmm_model': 'mml_core.motion.gmm_motion_model',
 'ScratchTransformer': 'mml_core.motion.scratch_transformer',
 'HistoryBuffer': 'mml_core.motion.utils'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
