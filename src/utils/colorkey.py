"""
Color key generation module for crystallographic orientation mapping.

This module provides utilities to generate IPF (Inverse Pole Figure) colors
for crystallographic orientations, primarily for hexagonal crystals.
"""

import numpy as np
from numpy.typing import NDArray
from math import acos, atan2, sqrt

from src.utils.constants import (
    QUAT_SYM,
    PI_OVER_180,
    K_180_OVER_PI,
    SQRT3_INV,
    USE_INVERSION,
)


class ColorKeyGenerator:
    """Generates color keys for crystallographic orientation mapping.

    This class provides methods to map crystallographic directions to RGB
    colors, particularly for hexagonal crystal systems using the inverse
    pole figure (IPF) coloring scheme.
    """

    @staticmethod
    def in_unit_triangle(eta: float = 0, chi: float = 0) -> bool:
        """Check if the point (eta, chi) is inside the unit triangle.

        Args:
            eta: Azimuthal angle in radians.
            chi: Polar angle in radians.

        Returns:
            bool: True if the point is inside the unit triangle, False otherwise.
        """
        if eta < 0 or eta > 45.0 * PI_OVER_180 or chi < 0 or chi > acos(SQRT3_INV):
            return False
        return True

    @staticmethod
    def drgb(a: int = 0, r: int | list[int] = 0, g: int = 0, b: int = 0) -> int:
        """Convert ARGB values to a packed integer.

        Args:
            a: Alpha channel value (0-255).
            r: Red channel value (0-255) or a list of [r, g, b] values.
            g: Green channel value (0-255), ignored if r is a list.
            b: Blue channel value (0-255), ignored if r is a list.

        Returns:
            int: Packed ARGB value as a 32-bit integer.
        """
        if isinstance(r, list) and len(r) == 3:
            g = int(round(r[1]))
            b = int(round(r[2]))
            r = int(round(r[0]))

        return ((a & 0xFF) << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)

    def generate_ipf_color(self, zone_axis: NDArray | list[float]) -> list[int]:
        """Generate IPF coloring for a given zone axis.

        Args:
            zone_axis: 3D vector representing crystallographic direction.

        Returns:
            List[int]: RGB color as a list of 3 integers (0-255).
        """
        # Normalize the zone axis
        zone_axis = np.asarray(zone_axis) / np.linalg.norm(zone_axis)

        # Get 24 symmetric vectors, size = [24,3]
        equivalent_zone_axes = np.matmul(
            QUAT_SYM.as_matrix(), np.tile(zone_axis, [24, 1])[:, :, np.newaxis]
        ).squeeze()

        # Add the opposite direction to the equivalent zone axes
        equivalent_zone_axes = np.concatenate(
            [equivalent_zone_axes, -1 * equivalent_zone_axes], axis=0
        )

        chi = 0.0
        eta = 0.0
        found_in_triangle = False

        # Find the equivalent direction that lies in the unit triangle
        for zone_axis in equivalent_zone_axes:
            if zone_axis[2] < 0:
                if USE_INVERSION:
                    zone_axis = -zone_axis
                else:
                    continue

            chi = acos(zone_axis[2])
            eta = atan2(zone_axis[1], zone_axis[0])

            if self.in_unit_triangle(eta, chi):
                found_in_triangle = True
                break

        if not found_in_triangle:
            # Handle the case where no equivalent direction is in the unit triangle
            # This shouldn't happen with proper symmetry operators but is handled for safety
            pass

        # Convert to degrees for color mapping
        eta_min = 0
        eta_max = 45
        chi_max = acos(SQRT3_INV) * K_180_OVER_PI
        eta_deg = eta * K_180_OVER_PI
        chi_deg = chi * K_180_OVER_PI

        # Calculate RGB values
        rgb = [1 - chi_deg / chi_max, 0, abs(eta_deg - eta_min) / (eta_max - eta_min)]
        rgb[1] = 1 - rgb[2]
        rgb[1] *= chi_deg / chi_max
        rgb[2] *= chi_deg / chi_max

        # Apply gamma correction
        rgb = [sqrt(val) for val in rgb]

        # Normalize and convert to 8-bit
        max_val = max(rgb)
        rgb = [int(round(255 * val / max_val)) for val in rgb]

        return rgb
