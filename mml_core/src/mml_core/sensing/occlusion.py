"""Occlusion geometry extracted without changing boundary conventions."""

import numpy as np


class Occlusions:
    def __init__(self, positions, widths):
        """List of occlusions with helper functions"""
        print("Occlusions initialized with centers:", positions)
        print("Occlusions initialized with widths:", widths)
        self.occlusions = [(positions[ii], widths[ii]) for ii in range(len(positions))]

    def in_occlusion(self, pos):
        """Return true if the position measurement is in occlusion zones for a single
        position but if it is an array of positions, return an array of booleans for
        particles inside occlusion
        Inputs: pos: position to check - numpy.array of shape (2,)
        """
        if pos.ndim == 1:
            for position, width in self.occlusions:
                if np.all(
                    [
                        pos[0] > position[0] - width / 2,
                        pos[0] < position[0] + width / 2,
                        pos[1] > position[1] - width / 2,
                        pos[1] < position[1] + width / 2,
                    ]
                ):
                    return True
            return False
        else:
            return np.array(
                [
                    any(
                        np.all(
                            [
                                pos[ii, 0] > position[0] - width / 2,
                                pos[ii, 0] < position[0] + width / 2,
                                pos[ii, 1] > position[1] - width / 2,
                                pos[ii, 1] < position[1] + width / 2,
                            ]
                        )
                        for position, width in self.occlusions
                    )
                    for ii in range(pos.shape[0])
                ]
            )
