"""Shared visibility and occlusion geometry."""

from .field_of_view import construct_fov, is_in_fov
from .occlusion import Occlusions

__all__ = ["construct_fov", "is_in_fov", "Occlusions"]
