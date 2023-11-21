from collections import namedtuple
from dataclasses import dataclass
from math import pi, sqrt


DockZoneNotchedCircle = namedtuple('DockZoneNotchedCircle', ['x', 'y', 'theta', 'radius'])

GRAVI_ACCEL = 9.81

@dataclass
class DockZoneParams:
    LARGE_CIRCLE_RADIUS: float = 0.6
    SMALL_CIRCLE_RADIUS: float = LARGE_CIRCLE_RADIUS / 2
    COLLISION_DISTANCE: float = 0.2
    COLLISION_AZIMUTH: float = pi/3
    COLLISION_EPSILON: float = 1e-3

@dataclass
class BatKinematicParams:
    MAX_LINEAR_VELOCITY: float = 5.
    MIN_LINEAR_VELOCITY: float = MAX_LINEAR_VELOCITY / 50
    MAX_ANGULAR_VELOCITY: float = 10 * pi
    MAX_ANGULAR_ACCELERATION: float = 1*MAX_ANGULAR_VELOCITY
    DECELERATION_FACTOR = 1
    CENTRIPETAL_ACCELERATION: float = 3*GRAVI_ACCEL # a_c = v^2 / r, with v is the linear velocity and r is the turning radius
    MIN_TURNING_RADIUS = 0.05 # m this is arbitrary, but simmulate the turn on a dime capability of the bat (not on the spot)
    SHARP_TURN_VELOCITY = sqrt(CENTRIPETAL_ACCELERATION * MIN_TURNING_RADIUS)
    CHIRP_RATE = 40 # Hz


@dataclass
class OtherKinematicParams:
    TAU_K: float = 0.1
    LINEAR_DECEL_LIMIT = -1.1 * BatKinematicParams.MAX_LINEAR_VELOCITY
    LINEAR_ACCEL_LIMIT = 1 * GRAVI_ACCEL
    BAIL_DISTANCE_MULTIPLIER = 5
    APPROACH_STEER_DAMPING = 10
    
