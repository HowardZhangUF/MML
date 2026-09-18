"""Deterministic initialization helpers for full-system experiments."""



import numpy as np


LAB_BOUNDS = np.array([[-3.0, -3.0], [3.0, 3.0]], dtype=float)


def sample_visible_initial_target_positions(
    agent_position: np.ndarray,
    fov_dims: np.ndarray,
    number_targets: int,
    occlusion_centers,
    occlusion_widths,
    rng: np.random.Generator,
    fov_margin: float = 0.10,
    occlusion_margin: float = 0.05,
    minimum_separation: float = 0.35,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Sample non-occluded target positions strictly inside the initial FOV."""
    agent_position = np.asarray(agent_position, dtype=float).reshape(2)
    fov_dims = np.asarray(fov_dims, dtype=float).reshape(2)
    number_targets = int(number_targets)
    centers = np.asarray(occlusion_centers, dtype=float).reshape(-1, 2)
    widths = np.asarray(occlusion_widths, dtype=float).reshape(-1)

    if number_targets < 0:
        raise ValueError("number_targets must be non-negative.")
    if len(centers) != len(widths):
        raise ValueError("Occlusion centers and widths must have equal lengths.")
    if np.any(fov_dims <= 2.0 * fov_margin):
        raise ValueError("FOV dimensions are too small for the requested margin.")

    lower = np.maximum(
        agent_position - fov_dims / 2.0 + fov_margin,
        LAB_BOUNDS[0] + fov_margin,
    )
    upper = np.minimum(
        agent_position + fov_dims / 2.0 - fov_margin,
        LAB_BOUNDS[1] - fov_margin,
    )
    if np.any(lower >= upper):
        raise ValueError(
            "The agent FOV has no valid initialization area inside the lab bounds."
        )

    positions = []
    for _ in range(max_attempts):
        if len(positions) == number_targets:
            break

        candidate = rng.uniform(lower, upper)
        if np.linalg.norm(candidate - agent_position) < minimum_separation:
            continue

        is_occluded = any(
            np.all(
                np.abs(candidate - center)
                <= width / 2.0 + occlusion_margin
            )
            for center, width in zip(centers, widths)
        )
        if is_occluded:
            continue

        if any(
            np.linalg.norm(candidate - existing) < minimum_separation
            for existing in positions
        ):
            continue

        positions.append(candidate)

    if len(positions) != number_targets:
        raise RuntimeError(
            f"Could place only {len(positions)} of {number_targets} targets "
            "inside the visible initialization region."
        )

    return np.asarray(positions, dtype=np.float32).reshape(number_targets, 2)
