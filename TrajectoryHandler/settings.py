from collections import namedtuple
from dataclasses import dataclass
from math import pi


DockZoneNotchedCircle = namedtuple('DockZoneNotchedCircle', ['x', 'y', 'theta', 'radius'])


@dataclass
class DockZoneParams:
    LARGE_CIRCLE_RADIUS: float = 0.7
    SMALL_CIRCLE_RADIUS: float = LARGE_CIRCLE_RADIUS / 2
    COLLISION_DISTANCE: float = 0.25
    COLLISION_AZIMUTH: float = pi/3
    COLLISION_EPSILON: float = 1e-3

@dataclass
class BatKinematicParams:
    MAX_LINEAR_SPEED: float = 2.0
    MIN_TURNING_RADIUS: float = 0.5

