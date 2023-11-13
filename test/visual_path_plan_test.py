""" 
This test script will build the same widget similarly to PoseEstimatorVisual app with some added features.
Basic features
    o arrange a 2x2 flower grid. init spacing maybe 3 meters apart and allow user to move flower pose about 1-2 meter range.
        -> flower init at (+-1.5,+-1.5). --> map range will be -3to3.
    + continuously draw a path to estimated flower pose. Maybe flowing this list of progresses first:
        o draw a circle around the estimated pose. --> with fix radius, with fix n_points.
        o draw the notched circle instead.
        - plan path with "dubin like curve" output should be like this.
            for S -> d(m)
            for C -> theta(degree), center of rotation, turning radius.
        - Convert of dubpin-bath to kinematics. [Not needed to draw but needed for control]
Wishlist features:
    + Run a episode --> May need to add abunch of constrant in here.
    + keep track of all planned paths, all estimated poses, as well as trajactor for later replay.
"""
import sys
import os
import time
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, List
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec
from collections import namedtuple

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting

from Perception.flower_pose_estimator import NaiveOneShotFlowerPoseEstimator, OnsetOneShotFlowerPoseEstimator, TwoShotFlowerPoseEstimator

FLOWER_ARROW_LENGTH = 0.2
BAT_ARROW_LENGTH = 0.3

DOCKZONE_RADIUS = 0.7
MIN_TURNING_RADIUS = 0.35

FONT = {'size': 10}

DockZoneCircle = namedtuple('DockZoneCircle', ['x', 'y', 'radius'])
DockZoneNotchedCircle = namedtuple('DockZoneNotchedCircle', ['x', 'y', 'theta', 'radius'])

def get_dockzone_circle(pose: ArrayLike,  radius:float=DOCKZONE_RADIUS) -> DockZoneCircle:
    return DockZoneCircle(pose[0], pose[1], radius)

def get_dockzone_notched_circle(pose: ArrayLike, radius:float=DOCKZONE_RADIUS) -> DockZoneNotchedCircle:
    return DockZoneNotchedCircle(pose[0], pose[1], pose[2], radius)

def circle_around_pose(pose:ArrayLike, radius:float=DOCKZONE_RADIUS, n_points:int=100) -> Tuple[ArrayLike, ArrayLike]:
    center = pose[:2]
    # generate x,y coordinate of a circle center (x,y) corrdincate and radius
    # x,y are array with shape of (n_points,)
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    return x, y

def circle_around_pose_notched(pose:ArrayLike, radius:float=DOCKZONE_RADIUS, n_points:int=100) -> Tuple[ArrayLike, ArrayLike]:
    center = pose[:2]
    # generate x,y coordinate of a circle center (x,y) corrdincate and radius
    # x,y are array with shape of (n_points,)
    left_center = center + radius/2*np.array([np.cos(pose[2] + np.pi/2), np.sin(pose[2] + np.pi/2)])
    right_center = center + radius/2*np.array([np.cos(pose[2] - np.pi/2), np.sin(pose[2] - np.pi/2)])
    theta0 = np.linspace(-np.pi*0.5, np.pi*0.5, int(n_points/4)) + pose[2]
    theta1 = np.linspace(-np.pi*1.5, -np.pi*0.5, int(n_points/2)) + pose[2]
    x = np.concatenate([
        center[0] + radius*np.cos(theta1),
        right_center[0] + radius/2*np.cos(theta0),
        left_center[0] + radius/2*np.cos(theta0),
        ])
    y = np.concatenate([
        center[1] + radius*np.sin(theta1),
        right_center[1] + radius/2*np.sin(theta0),
        left_center[1] + radius/2*np.sin(theta0),
        ])
    return x, y


class DubinsParams:
    def __init__(self, **kwargs):
        self.modes: List[str] = kwargs.get('mode', [])
        self.quantities: List[float] = kwargs.get('quantities', [])
        self.cost: float = kwargs.get('cost', 0.)
        self.radii: List[float] = kwargs.get('radii', [])
        # above are the bare minimum to reconstruct the path.
        # below are the optional parameters, can be helpful to reuse it for different task
        self.centers: List[ArrayLike] = kwargs.get('centers', [])
        self.tangent_points: List[ArrayLike] = kwargs.get('tangent_points', [])
        self.path_waypoints: Tuple[ArrayLike] = kwargs.get('path_waypoints', None)


class DubinsDockZonePathPlanner:
    def __init__(self, min_turn_radius:float=MIN_TURNING_RADIUS):
        self.min_turn_radius = min_turn_radius
        self.modes_collections = [
            ['L', 'S', 'L', 'L'],
            ['R', 'S', 'R', 'R'],
            ['L', 'S', 'R', 'R'],
            ['R', 'S', 'L', 'L'],
        ]
    
    def __call__(self, *args, **kwargs) -> DubinsParams:
        return self.run(*args, **kwargs)
    
    def run(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle) -> DubinsParams:
        
        candidate_paths = []
        best = np.inf
        best_idx = -1
        for i, modes in enumerate(self.modes_collections):
            path = self._solve_path(start_pose, dockzone_circle, self.min_turn_radius, modes)
            candidate_paths.append(path)
            if path.cost < best:
                best = path.cost
                best_idx = i
        return self.add_waypoints_to_path(candidate_paths[best_idx])
    
    # TODO: This one was copy from the vanila version. May need modification.
    def add_waypoints_to_path(self, path: DubinsParams, n_points:int=50) -> DubinsParams:
        if not (np.any(path.tangent_points[0])and np.any(path.tangent_points[1])): return None
        if path.modes[1] != 'S': raise NotImplementedError('Only support CSC for now.')
        alpha = np.arctan2(path.tangent_points[0][1] - path.centers[0][1], path.tangent_points[0][0] - path.centers[0][0])
        theta = np.linspace(alpha-path.quantities[0], alpha, n_points)
        segment0x = np.cos(theta)*path.radii[0] + path.centers[0][0]
        segment0y = np.sin(theta)*path.radii[0] + path.centers[0][1]
        segment1x = np.linspace(path.tangent_points[0][0], path.tangent_points[1][0], n_points)
        segment1y = np.linspace(path.tangent_points[0][1], path.tangent_points[1][1], n_points)
        alpha = np.arctan2(path.tangent_points[1][1] - path.centers[2][1], path.tangent_points[1][0] - path.centers[2][0])
        theta = np.linspace(alpha, alpha+path.quantities[2], n_points)
        segment2x = np.cos(theta)*path.radii[2] + path.centers[2][0]
        segment2y = np.sin(theta)*path.radii[2] + path.centers[2][1]
        if len(path.modes) >3:
            alpha = np.arctan2(path.tangent_points[2][1] - path.centers[3][1], path.tangent_points[2][0] - path.centers[3][0])
            theta = np.linspace(alpha, alpha+path.quantities[3], n_points)
            segment2x = np.concatenate((segment2x, np.cos(theta)*path.radii[3] + path.centers[3][0] ))
            segment2y = np.concatenate((segment2y, np.sin(theta)*path.radii[3] + path.centers[3][1] ))
        x = np.concatenate((segment0x, segment1x, segment2x))
        y = np.concatenate((segment0y, segment1y, segment2y))
        path.path_waypoints = (x, y)
        return path

    def _solve_path(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, min_turn_radius: float, modes: List[str]) -> DubinsParams:
        radii = [min_turn_radius, np.inf, dockzone_circle.radius*0.5, dockzone_circle.radius]
        end_pose = np.asarray([dockzone_circle.x, dockzone_circle.y, Spatializer.wrapToPi(dockzone_circle.theta + np.pi)])
        centers, tangent_points, radii, modes = self._find_key_points(start_pose, end_pose, modes, radii, dockzone_circle.theta)
        quantities = self._compute_quantities(start_pose, end_pose, centers, tangent_points, modes)
        cost = self._compute_cost(radii, quantities, modes)
        return DubinsParams(mode=modes, radii=radii, quantities=quantities, cost=cost, centers=centers, tangent_points=tangent_points)
    
    def _find_key_points(self, star_pose: ArrayLike, end_pose: ArrayLike, modes:List[str], radii:List[float], dockzone_theta:float) -> Tuple[List[ArrayLike],
                                                                                                                                 List[ArrayLike]]:
        centers = []
        centers.append(self._find_center_of_rotation(star_pose, modes[0], radii[0]))
        if modes[1] != 'S': raise NotImplementedError('Only support CSC for now.')
        else: centers.append(None)
        centers.append(self._find_center_of_rotation(end_pose, modes[2], radii[2]))
        centers.append(np.asarray([end_pose[0], end_pose[1]]))
        tangent_points, centers, radii, modes = self._find_tangent_points(centers, modes, radii, dockzone_theta=dockzone_theta)
        return centers, tangent_points, radii, modes
    
    def _find_center_of_rotation(self, pose:ArrayLike, mode:str, min_turn_radius:float) -> ArrayLike:
        if mode == 'L':
            center = pose[:2] + min_turn_radius*np.array([np.cos(pose[2]+0.5*np.pi), np.sin(pose[2]+0.5*np.pi)])
        if mode == 'R':
            center = pose[:2] + min_turn_radius*np.array([np.cos(pose[2]-0.5*np.pi), np.sin(pose[2]-0.5*np.pi)])
        return center
    

    def _find_tangent_points(self, centers: List[ArrayLike], modes: List[str], radii: List[float], dockzone_theta:float) -> Tuple[List[ArrayLike]]:
        candidate_tangent_points = []
        # compute tangent candidates from the small circle
        candidate_tangent_points.append(self._find_tangent_points_CSC(centers[:2]+[centers[2]], modes[:2]+[modes[2]], radii[:2]+[radii[2]]))
        # compute tangent candidates from the big circle
        candidate_tangent_points.append(self._find_tangent_points_CSC(centers[:2]+[centers[3]], modes[:2]+[modes[3]], radii[:2]+[radii[3]]))
        valids = []
        valids.append(self._is_tagent_points_valid(candidate_tangent_points[0], centers, dockzone_theta, large_circle=False))
        valids.append(self._is_tagent_points_valid(candidate_tangent_points[1], centers, dockzone_theta, large_circle=True))
        if np.sum(valids) == 0: return [None]*(len(centers)-1), centers, radii, modes
        if np.sum(valids) == 2: print('Warning: There are 2 valid tangent points. I did not anticipate this case.')
        if valids[0]:
            centers = centers[:2] + [centers[2]]
            radii = radii[:2] + [radii[2]]
            tangent_points = candidate_tangent_points[0]
            modes = modes[:2] + [modes[2]]
        else:
            centers = centers[:2] + [centers[3], centers[2]]
            radii = radii[:2] + [radii[3], radii[2]]
            # find a theta from large circle to small circle, small circle is centers[2], large circle is centers[3]
            # BUG: the intersection is on the opposite side.
            theta = np.arctan2(centers[3][1] - centers[2][1], centers[3][0] - centers[2][0])
            intersection_between_two_circles = centers[2] + radii[2]*np.array([np.cos(theta), np.sin(theta)])
            tangent_points = candidate_tangent_points[1] + [intersection_between_two_circles]
            modes = modes[:2] + [modes[3], modes[2]]
        return tangent_points, centers, radii, modes

    def _is_tagent_points_valid(self, tangent_points: List[ArrayLike], centers: List[ArrayLike], dockzone_theta:float, large_circle=True) -> bool:
        tangent_point = tangent_points[1]
        if not np.any(tangent_point): return False
        # BUG: When two circles are very close.
        # alpha = np.arctan2(tangent_point[1] - centers[2][1], tangent_point[0] - centers[2][0])
        # TypeError: 'NoneType' object is not subscriptable
        alpha = np.arctan2(tangent_point[1] - centers[2][1], tangent_point[0] - centers[2][0])
        if large_circle: # np.mod(x + np.pi, 2*np.pi) - np.pi
            if np.abs(np.mod(alpha - dockzone_theta + np.pi, 2*np.pi) - np.pi  ) >= np.pi/2: return True
        else:
            if np.abs(np.mod(alpha - dockzone_theta + np.pi, 2*np.pi) - np.pi  ) < np.pi/2: return True
        return False
        

    def _find_tangent_points_CSC(self, centers: List[ArrayLike], modes: List[str], radii: List[float]) -> List[ArrayLike]:
        # distance between 2 centers
        d = np.linalg.norm(centers[0] - centers[2])
        if d < np.abs(radii[0]-radii[2]): return [None, None] # no solution found, one circle is inside the other
        if d < (radii[0]+radii[2]) and (modes[0]!= modes[2]): return [None, None] # no solution found
        if d < np.abs(radii[0]-radii[2]): return [None, None] # no solution found
        # find the angle between 2 centers
        theta = np.arctan2(centers[2][1] - centers[0][1], centers[2][0] - centers[0][0])
        tangent_points = []
        if modes[0] == modes[2]:
            beta = np.pi - np.arccos(np.sign(radii[2]-radii[0])*(radii[2] - radii[0])/d)
            alpha = theta - beta if modes[0] == 'L' else theta + beta
            tangent_points.append(centers[0] + radii[0]*np.array([np.cos(alpha), np.sin(alpha)]))
            tangent_points.append(centers[2] + radii[2]*np.array([np.cos(alpha), np.sin(alpha)]))
        else:
            beta = np.arccos((radii[0] + radii[2])/d)
            if modes[0] == 'L': # modes[2] == 'R'
                tangent_points.append(centers[0] \
                    + radii[0]*np.array([np.cos(theta-beta), np.sin(theta-beta)]))
                tangent_points.append(centers[2] \
                    + radii[2]*np.array([np.cos(theta-beta-np.pi), np.sin(theta-beta-np.pi)]))
            elif modes[0] == 'R': # modes[2] == 'L'
                tangent_points.append(centers[0] \
                    + radii[0]*np.array([np.cos(theta+beta), np.sin(theta+beta)]))
                tangent_points.append(centers[2] \
                    + radii[2]*np.array([np.cos(theta+beta+np.pi), np.sin(theta+beta+np.pi)]))
            else: raise ValueError('modes[0] and modes[2] must be either L or R. but got {} and {}'.format(modes[0], modes[2]))
        return tangent_points

    # This function is Fine.
    def _compute_cost(self, radii: List[float], quantities:List[float], modes: List[str]) -> float:
        cost = 0.
        for i in range(len(modes)):
            if modes[i] == 'S': cost += quantities[i]
            else:
                cost += radii[i] * np.abs(quantities[i])
        return cost

    # This function will may need to be modified.
    def _compute_quantities(self, start_pose: ArrayLike, target_pose: ArrayLike,
                            centers: List[ArrayLike], tangent_points: List[ArrayLike], modes: List[str]) -> List[float]:
        for tangent_point in tangent_points:
            if not np.any(tangent_point): return [np.inf]*(len(modes))
        quantities = []
        for i, m in enumerate(modes):
            first_pose = tangent_points[i-1] if i>0 else start_pose
            second_pose = tangent_points[i] if i<(len(modes)-1) else target_pose
            if m == 'S': quantities.append(np.linalg.norm(second_pose - first_pose))
            else:
                # alpha is the relative theta of first pose to the center
                # beta is the relative theta of second pose to the center
                alpha = np.arctan2(first_pose[1] - centers[i][1], first_pose[0] - centers[i][0])
                beta = np.arctan2(second_pose[1] - centers[i][1], second_pose[0] - centers[i][0])
                if m=='L': quantities.append( (beta-alpha)%(2*np.pi) )
                else: quantities.append( -1*((alpha-beta)%(2*np.pi)) )
        #print(modes, quantities, len(centers), len(tangent_points))
        return quantities

class DubinsPathPlanner:
    def __init__(self,
                 min_turn_radius: float=MIN_TURNING_RADIUS,):
        self.min_turn_radius = min_turn_radius
        self.modes_collections = [
            ['L', 'S', 'L'],
            ['R', 'S', 'R'],
            ['L', 'S', 'R'],
            ['R', 'S', 'L'],
        ]

    # solve for each mode then take the best one.

    def __call__(self, *args, **kwargs) -> DubinsParams:
        return self.run(*args, **kwargs)
    
    def run(self, start_pose: ArrayLike, target_pose: ArrayLike) -> DubinsParams:
        candidate_paths = []
        best = np.inf
        best_idx = -1
        for i, modes in enumerate(self.modes_collections):
            path = self._solve_path(start_pose, target_pose, self.min_turn_radius, modes)
            candidate_paths.append(path)
            if path.cost < best:
                best = path.cost
                best_idx = i
        return self.add_waypoints_to_path(candidate_paths[best_idx])
    
    def add_waypoints_to_path(self, path: DubinsParams, n_points:int=50) -> DubinsParams:
        if not (np.any(path.tangent_points[0])and np.any(path.tangent_points[1])): return None
        if path.modes[1] != 'S': raise NotImplementedError('Only support CSC for now.')
        alpha = np.arctan2(path.tangent_points[0][1] - path.centers[0][1], path.tangent_points[0][0] - path.centers[0][0])
        theta = np.linspace(alpha-path.quantities[0], alpha, n_points)
        segment0x = np.cos(theta)*path.radii[0] + path.centers[0][0]
        segment0y = np.sin(theta)*path.radii[0] + path.centers[0][1]
        segment1x = np.linspace(path.tangent_points[0][0], path.tangent_points[1][0], n_points)
        segment1y = np.linspace(path.tangent_points[0][1], path.tangent_points[1][1], n_points)
        alpha = np.arctan2(path.tangent_points[1][1] - path.centers[2][1], path.tangent_points[1][0] - path.centers[2][0])
        theta = np.linspace(alpha, alpha+path.quantities[2], n_points)
        segment2x = np.cos(theta)*path.radii[2] + path.centers[2][0]
        segment2y = np.sin(theta)*path.radii[2] + path.centers[2][1]
        x = np.concatenate((segment0x, segment1x, segment2x))
        y = np.concatenate((segment0y, segment1y, segment2y))
        path.path_waypoints = (x, y)
        return path

    def _solve_path(self, start_pose: ArrayLike, end_pose: ArrayLike, min_turn_radius: float, modes: List[str]) -> DubinsParams:
        radii = [min_turn_radius, np.inf, min_turn_radius*4]
        centers, tangent_points = self._find_key_points(start_pose, end_pose, modes, radii)
        quantities = self._compute_quantities(start_pose, end_pose, centers, tangent_points, modes)
        cost = self._compute_cost(radii, quantities, modes)
        return DubinsParams(mode=modes, radii=radii, quantities=quantities, cost=cost, centers=centers, tangent_points=tangent_points)
    
    def _find_key_points(self, star_pose: ArrayLike, end_pose: ArrayLike, modes:List[str], radii:List[float]) -> Tuple[List[ArrayLike],
                                                                                                                                 List[ArrayLike]]:
        centers = []
        centers.append(self._find_center_of_rotation(star_pose, modes[0], radii[0]))
        if modes[1] != 'S': raise NotImplementedError('Only support CSC for now.')
        else: centers.append(None)
        centers.append(self._find_center_of_rotation(end_pose, modes[2], radii[2])) 
        tangent_points = self._find_tangent_points_CSC(centers, modes, radii)
        return centers, tangent_points

    def _find_center_of_rotation(self, pose:ArrayLike, mode:str, min_turn_radius:float) -> ArrayLike:
        if mode == 'L':
            center = pose[:2] + min_turn_radius*np.array([np.cos(pose[2]+0.5*np.pi), np.sin(pose[2]+0.5*np.pi)])
        if mode == 'R':
            center = pose[:2] + min_turn_radius*np.array([np.cos(pose[2]-0.5*np.pi), np.sin(pose[2]-0.5*np.pi)])
        return center

    def _find_tangent_points_CSC(self, centers: List[ArrayLike], modes: List[str], radii: List[float]) -> List[ArrayLike]:
        # distance between 2 centers
        d = np.linalg.norm(centers[0] - centers[2])
        if d < np.abs(radii[0]-radii[2]): return [None, None] # no solution found, one circle is inside the other
        if d < (radii[0]+radii[2]) and (modes[0]!= modes[2]): return [None, None] # no solution found
        if d < np.abs(radii[0]-radii[2]): return [None, None] # no solution found
        # find the angle between 2 centers
        theta = np.arctan2(centers[2][1] - centers[0][1], centers[2][0] - centers[0][0])
        tangent_points = []
        if modes[0] == modes[2]:
            beta = np.pi - np.arccos(np.sign(radii[2]-radii[0])*(radii[2] - radii[0])/d)
            alpha = theta - beta if modes[0] == 'L' else theta + beta
            tangent_points.append(centers[0] + radii[0]*np.array([np.cos(alpha), np.sin(alpha)]))
            tangent_points.append(centers[2] + radii[2]*np.array([np.cos(alpha), np.sin(alpha)]))
        else:
            beta = np.arccos((radii[0] + radii[2])/d)
            if modes[0] == 'L': # modes[2] == 'R'
                tangent_points.append(centers[0] \
                    + radii[0]*np.array([np.cos(theta-beta), np.sin(theta-beta)]))
                tangent_points.append(centers[2] \
                    + radii[2]*np.array([np.cos(theta-beta-np.pi), np.sin(theta-beta-np.pi)]))
            elif modes[0] == 'R': # modes[2] == 'L'
                tangent_points.append(centers[0] \
                    + radii[0]*np.array([np.cos(theta+beta), np.sin(theta+beta)]))
                tangent_points.append(centers[2] \
                    + radii[2]*np.array([np.cos(theta+beta+np.pi), np.sin(theta+beta+np.pi)]))
            else: raise ValueError('modes[0] and modes[2] must be either L or R. but got {} and {}'.format(modes[0], modes[2]))
        return tangent_points

    def _compute_cost(self, radii: List[float], quantities:List[float], modes: List[str]) -> float:
        cost = 0.
        for i in range(len(modes)):
            if modes[i] == 'S': cost += quantities[i]
            else:
                cost += radii[i] * np.abs(quantities[i])
        return cost

    def _compute_quantities(self, start_pose: ArrayLike, target_pose: ArrayLike,
                            centers: List[ArrayLike], tangent_points: List[ArrayLike], modes: List[str]) -> List[float]:
        if not (np.any(tangent_points[0])and np.any(tangent_points[1])): return [np.inf, np.inf, np.inf]
        quantities = []
        for i, m in enumerate(modes):
            first_pose = start_pose if i==0 else tangent_points[i-1]
            second_pose = target_pose if i==2 else tangent_points[i]
            if m == 'S': quantities.append(np.linalg.norm(second_pose - first_pose))
            else:
                # alpha is the relative theta of first pose to the center
                # beta is the relative theta of second pose to the center
                alpha = np.arctan2(first_pose[1] - centers[i][1], first_pose[0] - centers[i][0])
                beta = np.arctan2(second_pose[1] - centers[i][1], second_pose[0] - centers[i][0])
                if m=='L': quantities.append( (beta-alpha)%(2*np.pi) )
                else: quantities.append( -1*((alpha-beta)%(2*np.pi)) )
        return quantities

# possible dubin paths: CSC, CCC
# 1. CSC: LSL, RSR, LSR, RSL
# 2. CCC: LRL, RLR


def convert_polar_to_cartesian(bat_pose: ArrayLike,
                               flower_distance: float, flower_azimuth: float, flower_orientation: float) -> ArrayLike:
    flower_pose = np.zeros(3)
    flower_pose[0] = bat_pose[0] + flower_distance*np.cos(bat_pose[2] + flower_azimuth)
    flower_pose[1] = bat_pose[1] + flower_distance*np.sin(bat_pose[2] + flower_azimuth)
    flower_pose[2] = Spatializer.wrapToPi(bat_pose[2] + flower_azimuth + np.pi - flower_orientation)
    return flower_pose

def set_up_waveform_plot(ax, distances, data_left, data_right, title=None, fontsize=10):
    line_left, = ax.plot(distances, data_left, linewidth=0.5, alpha=0.5)
    line_right, = ax.plot(distances, data_right, linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Distance (m)', fontsize=fontsize)
    ax.set_ylabel('Amplitude', fontsize=fontsize)
    if title is not None: ax.set_title(title, fontsize=fontsize)
    return line_left, line_right, ax

def select_pose_estimator(estimator_type):
    if estimator_type not in ['Naive', 'Onset', 'TwoShot']:
        raise ValueError('estimator_type must be either Naive, Onset or TwoShot. but got {}'.format(estimator_type))
    if estimator_type == 'Naive':
        return NaiveOneShotFlowerPoseEstimator(cache_inputs=True)
    if estimator_type == 'Onset':
        return OnsetOneShotFlowerPoseEstimator(cache_inputs=True)
    if estimator_type == 'TwoShot':
        return TwoShotFlowerPoseEstimator(cache_inputs=True)
    
def utest_widget():
    matplotlib.rc('font', **FONT)
    render = Spatializer.Render()
    render.compression_filter = Spatializer.Compressing.Downsample512()
    pose_estimators = {'naive': select_pose_estimator('Naive'), 'onset': select_pose_estimator('Onset'), 'twoshot': select_pose_estimator('TwoShot')}
    for estimator in pose_estimators.values(): estimator.presence_detector.detection_threshold = 4.
    path_planer = DubinsDockZonePathPlanner()
    estimator_type = 'naive'
    compressed_distances = render.compression_filter(Setting.DISTANCE_ENCODING)
    bat_pose = np.asarray([0., 0., 0.])
    init_flowers_pose = [
        [-1.5, -1.5, 0., 3.],
        [1.5, -1.5, 0., 3.],
        [1.5, 1.5, 0., 3.],
        [-1.5, 1.5, 0., 3.],
    ]
    cartesian_objects_matrix = np.asarray(init_flowers_pose).astype(np.float32)
    render.run(bat_pose, cartesian_objects_matrix)
    inputs = np.concatenate([render.compress_left, render.compress_right]).reshape(1,2,-1)
    prediction = pose_estimators[estimator_type](inputs)
    fig = plt.figure(figsize=(12, 6), dpi=200)
    gs = GridSpec(10, 10, figure=fig)
    ax1 = fig.add_subplot(gs[:,:4])
    ax2 = fig.add_subplot(gs[:4,5:])
    ax3 = fig.add_subplot(gs[6:,5:])
    bat_arrow = ax1.arrow(bat_pose[0], bat_pose[1],
            BAT_ARROW_LENGTH*np.cos(bat_pose[2]), BAT_ARROW_LENGTH*np.sin(bat_pose[2]),
            width=0.05, head_width=0.1, head_length=0.05, fc='k', ec='k')
    if prediction[0]:
        est_flower_pose = convert_polar_to_cartesian(bat_pose, prediction[0], prediction[1], prediction[2])
    else: est_flower_pose = cartesian_objects_matrix[0,:3]
    flower0_est_arrow = ax1.arrow(est_flower_pose[0], est_flower_pose[1],
            FLOWER_ARROW_LENGTH*np.cos(est_flower_pose[2]), FLOWER_ARROW_LENGTH*np.sin(est_flower_pose[2]),
            width=0.1, head_width=0.1, head_length=0.05, fc='r', ec='r', alpha=0.5)
    flowers_arrows = []
    for i in range(len(cartesian_objects_matrix)):
        flowers_arrows.append(ax1.arrow(cartesian_objects_matrix[i,0], cartesian_objects_matrix[i,1],
                FLOWER_ARROW_LENGTH*np.cos(cartesian_objects_matrix[i,2]), FLOWER_ARROW_LENGTH*np.sin(cartesian_objects_matrix[i,2]),
                width=0.05, head_width=0.1, head_length=0.05, fc='m', ec='m'))
    
    dockzone_circle, = ax1.plot(*circle_around_pose_notched(est_flower_pose), 'k:', linewidth=1.0, alpha=0.5)
    dockzone = get_dockzone_notched_circle(est_flower_pose)
    path = path_planer(bat_pose, dockzone)
    path_line, = ax1.plot(*path.path_waypoints, 'b--', linewidth=1.0, alpha=0.5)
    for i in range(len(path.centers)):
        if i==1: continue
        centers_x = [path.centers[i][0]]
        centers_y = [path.centers[i][1]]
    tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
    tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
    path_keypoints, = ax1.plot(centers_x + tangent_points_x, centers_y + tangent_points_y, 'kx', markersize=5, alpha=0.5)
    ax1.set_xlim([-3,3])
    ax1.set_ylim([-3,3])
    
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend([bat_arrow, flower0_est_arrow, flowers_arrows[0]], ['Bat', 'Flower (Est)', 'Flower (GT)'],
               loc = 'upper left', bbox_to_anchor=(0., 1.25))

    envelope_left_, envelope_right_, ax2 = set_up_waveform_plot(ax2,
            compressed_distances, render.compress_left, render.compress_right, title='Echo envelope of the scene')
    inputs_left_, inputs_right_, ax3 = set_up_waveform_plot(ax3,
            compressed_distances,
            pose_estimators[estimator_type].cache['inputs'][0,0,:],
            pose_estimators[estimator_type].cache['inputs'][0,1,:], title='Inputs to the estimator')
    
    ax2.plot(compressed_distances, pose_estimators[estimator_type].presence_detector.profile*5, 'k--', linewidth=0.5, alpha=0.5)
    ax3.plot(compressed_distances, pose_estimators[estimator_type].presence_detector.profile, 'k--', linewidth=0.5, alpha=0.5)


    fig.subplots_adjust(left=.2, right=0.9)
    
    bat_slider_ax = []
    for i in range(len(bat_pose)):
        bat_slider_ax.append(fig.add_axes([0.2, 0.98 - i*0.03, 0.2, 0.01]) ) # x,y,theta
    flowers_checkbox_ax = fig.add_axes([ 0.92, 0.05, 0.05, 0.5 ])
    estimator_type_radio_ax = fig.add_axes([ 0.92, 0.6, 0.05, 0.2 ])
    flowers_slider_ax = []
    for i in range(len(init_flowers_pose)):
        flowers_slider_ax.append(fig.add_axes([0.02, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.06, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.10, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))

    bat_slider = []
    bat_slider.append(widgets.Slider(bat_slider_ax[0], 'x_bat', -1., 1., valstep=0.01, valinit=bat_pose[0]))
    bat_slider.append(widgets.Slider(bat_slider_ax[1], 'y_bat', -1., 1., valstep=0.01, valinit=bat_pose[1]))
    bat_slider.append(widgets.Slider(bat_slider_ax[2], '\u03B8_bat', -180, 180, valstep=0.1, valinit=np.degrees(bat_pose[2])))
    
    flowers_slider = []
    for i in range(len(init_flowers_pose)):
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3], f'\u0394x{i}', -1, 1, valstep=0.01, valinit=0., orientation='vertical') )
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3+1], f'\u0394y{i}', -1, 1, valstep=0.01, valinit=0., orientation='vertical') )
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3+2], f'\u0394\u03B8{i}', -180, 180, valstep=0.1, valinit=0., orientation='vertical') )
    flowers_checkbox_txt = []
    for i in range(len(init_flowers_pose)): flowers_checkbox_txt.append(f'F{i}')
    flowers_checkbox = widgets.CheckButtons(flowers_checkbox_ax, flowers_checkbox_txt, [True]*len(init_flowers_pose))
    estimator_type_radio = widgets.RadioButtons(estimator_type_radio_ax, ['naive', 'onset', 'twoshot'], active=0)



    def update(val):
        flower_status = flowers_checkbox.get_status()
        estimator_type = estimator_type_radio.value_selected
        bat_pose = np.asarray([
            bat_slider[0].val, bat_slider[1].val, np.radians(bat_slider[2].val)
        ]).astype(np.float32)
        for i in range(len(cartesian_objects_matrix)):
            cartesian_objects_matrix[i,0] = init_flowers_pose[i][0] + flowers_slider[i*3].val if flower_status[i] else -100.
            cartesian_objects_matrix[i,1] = init_flowers_pose[i][1] + flowers_slider[i*3+1].val if flower_status[i] else -0.
            cartesian_objects_matrix[i,2] = np.radians(flowers_slider[i*3+2].val) if flower_status[i] else 0.
        render.run(bat_pose, cartesian_objects_matrix)
        inputs = np.concatenate([render.compress_left, render.compress_right]).reshape(1,2,-1)
        prediction = pose_estimators[estimator_type](inputs)
        # if prediction[0]:
        #     print('gt orientation: {:.2f}'.format(np.degrees( render.viewer.filtered_objects_inview_polar[0,2] )))
        #     print('pred orientation: {:.2f}'.format(np.degrees(prediction[2])))
        if render.viewer.collision_status==True:
            bat_arrow.set_color('r')
            bat_pose = np.copy(render.viewer.bat_pose)
        else:
            bat_arrow.set_color('k')
        if prediction[0]:
            est_flower_pose = convert_polar_to_cartesian(bat_pose, prediction[0], prediction[1], prediction[2])
            flower0_est_arrow.set_color('r')
        else:
            est_flower_pose = cartesian_objects_matrix[0,:3]
            flower0_est_arrow.set_color('k')
        bat_arrow.set_data(x=bat_pose[0], y=bat_pose[1],
                dx=BAT_ARROW_LENGTH*np.cos(bat_pose[2]),
                dy=BAT_ARROW_LENGTH*np.sin(bat_pose[2]))
        for i in range(len(cartesian_objects_matrix)):
            flowers_arrows[i].set_data(x=cartesian_objects_matrix[i,0], y=cartesian_objects_matrix[i,1],
                dx=FLOWER_ARROW_LENGTH*np.cos(cartesian_objects_matrix[i,2]),
                dy=FLOWER_ARROW_LENGTH*np.sin(cartesian_objects_matrix[i,2]))
            flowers_arrows[i].set_visible(flower_status[i])
            for w in range(3): flowers_slider_ax[i*3 + w].set_visible(flower_status[i])

        dockzone = get_dockzone_notched_circle(est_flower_pose)
        path = path_planer(bat_pose, dockzone)
        d = np.linalg.norm(path.centers[2] - path.centers[0])
        if d < np.abs(path.radii[0]-path.radii[2]): path_line.set_visible(False)
        else: path_line.set_visible(True)

        path_line.set_data(*path.path_waypoints)
        for i in range(len(path.centers)):
            if i==1: continue
            centers_x = [path.centers[i][0]]
            centers_y = [path.centers[i][1]]
        tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
        tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
        path_keypoints.set_xdata(centers_x + tangent_points_x)
        path_keypoints.set_ydata(centers_y + tangent_points_y)
        flower0_est_arrow.set_data(x=est_flower_pose[0], y=est_flower_pose[1],
                dx=FLOWER_ARROW_LENGTH*np.cos(est_flower_pose[2]),
                dy=FLOWER_ARROW_LENGTH*np.sin(est_flower_pose[2]))
        envelope_left_.set_ydata(render.compress_left)
        envelope_right_.set_ydata(render.compress_right)
        inputs_left_.set_ydata(pose_estimators[estimator_type].cache['inputs'][0,0,:])
        inputs_right_.set_ydata(pose_estimators[estimator_type].cache['inputs'][0,1,:])
        dockzone_circle.set_data(*circle_around_pose_notched(est_flower_pose))
        
        fig.canvas.draw_idle()

    flowers_checkbox.on_clicked(update)
    estimator_type_radio.on_clicked(update)
    for slider in bat_slider+flowers_slider: slider.on_changed(update)

    plt.show()

def main():
    return utest_widget()

if __name__ == '__main__':
    main()