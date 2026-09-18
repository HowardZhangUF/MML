"""Field-of-view geometry with the original strict boundary comparisons."""

import numpy as np


def is_in_fov(sparticles, fov):
    """Check if the particles are in the FOV of the camera."""
    if sparticles.ndim == 1:
        return np.all(
            [
                sparticles[0] > fov[0],
                sparticles[0] < fov[1],
                sparticles[1] > fov[2],
                sparticles[1] < fov[3],
            ]
        )
    else:
        return np.logical_and.reduce(
            [
                sparticles[:, 0] > fov[0],
                sparticles[:, 0] < fov[1],
                sparticles[:, 1] > fov[2],
                sparticles[:, 1] < fov[3],
            ]
        )

def construct_fov(fov_center, fov_dims) -> np.ndarray:
    """Construct the FOV of the camera given the center
    of the FOV and the dimensions of the FOV
    Input: Center of the FOV
    Output: FOV dimensions in the form (x_min, x_max, y_min, y_max)
    """
    fov = np.array(
        [
            fov_center[0] - fov_dims[0] / 2,
            fov_center[0] + fov_dims[0] / 2,
            fov_center[1] - fov_dims[1] / 2,
            fov_center[1] + fov_dims[1] / 2,
        ]
    )
    return fov
