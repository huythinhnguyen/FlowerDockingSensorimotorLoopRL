from collections import namedtuple
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple

from .settings import DockZoneParams

DockZoneNotchedCircle = namedtuple('DockZoneNotchedCircle', ['x', 'y', 'theta', 'radius'])

def get_dockzone_notched_circle_from_flower_pose(flower_pose: ArrayLike, dockzone_radius: float = DockZoneParams.LARGE_CIRCLE_RADIUS, **kwargs) -> DockZoneNotchedCircle:
    return DockZoneNotchedCircle(
        x=flower_pose[0],
        y=flower_pose[1],
        theta=flower_pose[2],
        radius=dockzone_radius
    )

def generate_dockzone_notched_circle_waypoints(dockzone_circle: DockZoneNotchedCircle, num_waypoints: int = 100, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    center = np.array([dockzone_circle.x, dockzone_circle.y])
    radius = dockzone_circle.radius
    left_center  = center + radius * 0.5 * np.array([np.cos(dockzone_circle.theta + np.pi/2), np.sin(dockzone_circle.theta + np.pi/2)])
    right_center = center + radius * 0.5 * np.array([np.cos(dockzone_circle.theta - np.pi/2), np.sin(dockzone_circle.theta - np.pi/2)])
    front_thetas = np.linspace(-np.pi*0.5, np.pi*0.5, int(num_waypoints/4)) + dockzone_circle.theta
    back_thetas = np.linspace(-np.pi*1.5, -np.pi*0.5, int(num_waypoints/2)) + dockzone_circle.theta
    x = np.concatenate([
        center[0] + radius * np.cos(back_thetas),
        right_center[0] + radius*0.5*np.cos(front_thetas),
        left_center[0] + radius*0.5*np.cos(front_thetas),
    ])
    y = np.concatenate([
        center[1] + radius * np.sin(back_thetas),
        right_center[1] + radius*0.5*np.sin(front_thetas),
        left_center[1] + radius*0.5*np.sin(front_thetas),
    ])
    return x, y

