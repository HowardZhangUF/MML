"""Tracking implementations; production, tutorial and DiffPF filters stay distinct."""

from importlib import import_module

_EXPORTS = {'ParticleFilter': 'mml_core.tracking.particle_filter',
 'KalmanFilter': 'mml_core.tracking.kalman_filter',
 'ParticleFilterGNNAssociator': 'mml_core.tracking.particle_gnn',
 'KalmanFilterGNNAssociator': 'mml_core.tracking.kalman_gnn',
 'DiffPFTracker': 'mml_core.tracking.diffpf_tracker',
 'CFMParticleFilter': 'mml_core.tracking.tutorial_cfm'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
