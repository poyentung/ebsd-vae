"""Constants used throughout the package for crystallographic calculations."""

from math import pi
from numpy import sqrt
from scipy.spatial.transform import Rotation as R

PI_OVER_180 = pi / 180
K_180_OVER_PI = 180 / pi
SQRT2_INV = 1 / sqrt(2)
SQRT3_INV = 1 / sqrt(3)
USE_INVERSION = True

CUBIC_SYMMETRY = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5, -0.5],
    [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5, -0.5],
    [SQRT2_INV, SQRT2_INV, 0, 0],
    [SQRT2_INV, 0, SQRT2_INV, 0],
    [SQRT2_INV, 0, 0, SQRT2_INV],
    [SQRT2_INV, -SQRT2_INV, 0, 0],
    [SQRT2_INV, 0, -SQRT2_INV, 0],
    [SQRT2_INV, 0, 0, -SQRT2_INV],
    [0, SQRT2_INV, SQRT2_INV, 0],
    [0, -SQRT2_INV, SQRT2_INV, 0],
    [0, 0, SQRT2_INV, SQRT2_INV],
    [0, 0, -SQRT2_INV, SQRT2_INV],
    [0, SQRT2_INV, 0, SQRT2_INV],
    [0, -SQRT2_INV, 0, SQRT2_INV],
]
QUAT_SYM = R.from_quat(CUBIC_SYMMETRY)
