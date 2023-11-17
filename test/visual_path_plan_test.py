""" 
This test script will build the same widget similarly to PoseEstimatorVisual app with some added features.
Basic features
    o arrange a 2x2 flower grid. init spacing maybe 3 meters apart and allow user to move flower pose about 1-2 meter range.
        -> flower init at (+-1.5,+-1.5). --> map range will be -3to3.
    + continuously draw a path to estimated flower pose. Maybe flowing this list of progresses first:
        o draw a circle around the estimated pose. --> with fix radius, with fix n_points.
        o draw the notched circle instead.
        o plan path with "dubin like curve" output should be like this.
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
MIN_TURNING_RADIUS = 0.1

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
    def __init__(self, min_turn_radius:float=MIN_TURNING_RADIUS, collision_distance:float=0.25, collision_azimuth=np.pi/3):
        self.min_turn_radius = min_turn_radius
        self.modes_collections = [
            ['L', 'S', 'L', 'L'],
            ['R', 'S', 'R', 'R'],
            ['L', 'S', 'R', 'R'],
            ['R', 'S', 'L', 'L'],
        ]
        self.collision_distance = collision_distance
        self.collision_azimuth = collision_azimuth
    
    def __call__(self, *args, **kwargs) -> DubinsParams:
        return self.run2(*args, **kwargs)
    
    def run(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle) -> DubinsParams:
        candidate_paths = []
        best = np.inf
        best_idx = -1
        for i, modes in enumerate(self.modes_collections):
            path = self._solve_path2(start_pose, dockzone_circle, self.min_turn_radius, modes)
            candidate_paths.append(path)
            if path.cost < best:
                best = path.cost
                best_idx = i
        return self.add_waypoints_to_path(candidate_paths[best_idx])
    
    def add_waypoints_to_path(self, path:DubinsParams, n_points:int=50) -> DubinsParams:
        if not (np.any(path.tangent_points[0])and np.any(path.tangent_points[1])): return None
        if path.modes[1] != 'S': raise NotImplementedError('Only support CSC for now.')
        if path.modes[-1] == 'S': # add one more point to tangent points.
            theta = np.arctan2(path.tangent_points[-1][1] - path.centers[-2][1], path.tangent_points[-1][0] - path.centers[-2][0])
            theta += np.pi/2 if path.modes[-2]=='L' else -np.pi/2
            extra_tangent_point = path.tangent_points[-1] + path.quantities[-1]*np.array([np.cos(theta), np.sin(theta)])
            tangent_points = path.tangent_points + [extra_tangent_point]
        else:
            tangent_points = path.tangent_points
        # create the first segment
        alpha = np.arctan2(path.tangent_points[0][1] - path.centers[0][1], path.tangent_points[0][0] - path.centers[0][0])
        theta = np.linspace(alpha-path.quantities[0], alpha, n_points)
        x = np.cos(theta)*path.radii[0] + path.centers[0][0]
        y = np.sin(theta)*path.radii[0] + path.centers[0][1]
        for i in range(1, len(path.modes)):
            if path.modes[i] == 'S':
                x = np.concatenate((x, np.linspace(tangent_points[i-1][0], tangent_points[i][0], n_points)))
                y = np.concatenate((y, np.linspace(tangent_points[i-1][1], tangent_points[i][1], n_points)))
            else:
                alpha = np.arctan2(tangent_points[i-1][1] - path.centers[i][1], tangent_points[i-1][0] - path.centers[i][0])
                theta = np.linspace(alpha, alpha+path.quantities[i], n_points)
                x = np.concatenate((x, np.cos(theta)*path.radii[i] + path.centers[i][0]))
                y = np.concatenate((y, np.sin(theta)*path.radii[i] + path.centers[i][1]))
        path.path_waypoints = (x, y)
        return path


    def _solve_path(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, min_turn_radius: float, modes: List[str]) -> DubinsParams:
        radii = [min_turn_radius, np.inf, dockzone_circle.radius*0.5, dockzone_circle.radius]
        end_pose = np.asarray([dockzone_circle.x, dockzone_circle.y, Spatializer.wrapToPi(dockzone_circle.theta + np.pi)])
        centers, tangent_points, radii, modes = self._find_key_points(start_pose, end_pose, modes, radii, dockzone_circle.theta)
        quantities = self._compute_quantities(start_pose, end_pose, centers, tangent_points, modes)
        cost = self._compute_cost(radii, quantities, modes)
        return DubinsParams(mode=modes, radii=radii, quantities=quantities, cost=cost, centers=centers, tangent_points=tangent_points)
    
    ##################################################
    ##########  ALL NEW PROTOTYPES GO HERE  ##########

    def run2(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle) -> DubinsParams:
        candidate_paths = []
        best = np.inf
        best_idx = -1
        for i, modes in enumerate(self.modes_collections):
            path = self._solve_path2(start_pose, dockzone_circle, self.min_turn_radius, modes)
            candidate_paths.append(path)
            if path.cost < best:
                best = path.cost
                best_idx = i
        if best < np.inf: return self.add_waypoints_to_path(candidate_paths[best_idx])
        candidate_paths = []
        for i, modes in enumerate(self.modes_collections):
            for k in range(2):
                path = self._solve_path2(start_pose, dockzone_circle, self.min_turn_radius, modes, exception=True, exception_case=k)
                candidate_paths.append(path)
                if path.cost < best:
                    if self._collision_free_check(path, dockzone_circle):
                        best = path.cost
                        best_idx = 2*i + k
        return self.add_waypoints_to_path(candidate_paths[best_idx])
    
    def _collision_free_check(self, path: DubinsParams, dockzone_circle: DockZoneNotchedCircle, epsilon:float=1e-3):
        if not (np.any(path.tangent_points[0])and np.any(path.tangent_points[1])): return False
        cx = dockzone_circle.x - epsilon*np.cos(dockzone_circle.theta)
        cy = dockzone_circle.y - epsilon*np.sin(dockzone_circle.theta)
        path  = self.add_waypoints_to_path(path, n_points=50)
        distances = np.linalg.norm(np.asarray(( path.path_waypoints[0], path.path_waypoints[1] )) \
                                    - np.asarray([cx, cy]).reshape(-1,1), axis=0)
        thetas = np.arctan2(path.path_waypoints[1] - cy, path.path_waypoints[0] - cx)
        idx = np.where(distances < self.collision_distance)
        if np.sum(np.abs(Spatializer.wrapToPi(thetas[idx] - dockzone_circle.theta) ) > self.collision_azimuth) > 0:
            return False
        return True

    # def collision_check(waypoints:ArrayLike, dockzone_circle:DockZoneNotchedCircle, collision_distance:float, collision_azimuth:float):
    #     waypoints_x = waypoints[0]
    #     waypoints_y = waypoints[1]
    #     distances = np.linalg.norm(np.asarray((waypoints_x, waypoints_y)) - np.asarray([dockzone_circle.x, dockzone_circle.y]).reshape(-1,1), axis=0)
    #     thetas = np.arctan2(waypoints_y - dockzone_circle.y, waypoints_x - dockzone_circle.x)
    #     idx = np.where(distances < collision_distance)
    #     if np.sum(np.abs(Spatializer.wrapToPi(thetas[idx] - dockzone_circle.theta) ) > collision_azimuth) > 0:
    #         return False
    #     return True

    # def _solve_path2(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, min_turn_radius: float, modes: List[str], exception:bool=False) -> DubinsParams:
    #     radii = [min_turn_radius, np.inf, dockzone_circle.radius*0.5, dockzone_circle.radius]
    #     end_pose = np.asarray([dockzone_circle.x, dockzone_circle.y, Spatializer.wrapToPi(dockzone_circle.theta + np.pi)])
    #     #if self._is_key_points_exceptions(start_pose, end_pose, modes, radii, dockzone_circle) and exception:
    #     if exception:
    #         centers, tangent_points, new_radii, new_modes = self._find_key_points_w_exception(start_pose, end_pose, modes, radii, dockzone_circle.theta)
    #     else: centers, tangent_points, new_radii, new_modes = self._find_key_points(start_pose, end_pose, modes, radii, dockzone_circle.theta)
    #     quantities = self._compute_quantities(start_pose, end_pose, centers, tangent_points, new_modes)
    #     cost = self._compute_cost(new_radii, quantities, new_modes)
    #     return DubinsParams(mode=new_modes, radii=new_radii, quantities=quantities, cost=cost, centers=centers, tangent_points=tangent_points)

    def _solve_path2(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, min_turn_radius: float, modes: List[str], exception:bool=False, exception_case:int=0) -> DubinsParams:
        radii = [min_turn_radius, np.inf, dockzone_circle.radius*0.5, dockzone_circle.radius]
        end_pose = np.asarray([dockzone_circle.x, dockzone_circle.y, Spatializer.wrapToPi(dockzone_circle.theta + np.pi)])
        #if self._is_key_points_exceptions(start_pose, end_pose, modes, radii, dockzone_circle) and exception:
        if exception:
            centers, tangent_points, new_radii, new_modes = self._find_key_points_w_exception(start_pose, end_pose, modes, radii, dockzone_circle.theta, exception_case)
        else: centers, tangent_points, new_radii, new_modes = self._find_key_points(start_pose, end_pose, modes, radii, dockzone_circle.theta)
        quantities = self._compute_quantities(start_pose, end_pose, centers, tangent_points, new_modes)
        cost = self._compute_cost(new_radii, quantities, new_modes)
        return DubinsParams(mode=new_modes, radii=new_radii, quantities=quantities, cost=cost, centers=centers, tangent_points=tangent_points)

    def _is_key_points_exceptions(self, start_pose: ArrayLike, end_pose: ArrayLike, modes:List[str], radii:List[float], dockzone_circle: DockZoneNotchedCircle) -> bool:
        # 1. The first cirle is completely inside the dockzone circle.
        # 2. Modes needs to be CSC.
        if modes[0]==modes[-1]: 
            centers = [self._find_center_of_rotation(start_pose, modes[0], radii[0]), None]
            centers.append(self._find_center_of_rotation(end_pose, modes[2], radii[2]))
            centers.append(np.asarray([end_pose[0], end_pose[1]]))
            d_to_large_circle = np.linalg.norm(centers[0] - centers[3])
            d_to_small_circle = np.linalg.norm(centers[0] - centers[2])
            # first circle is not inside the large circle:
            if d_to_large_circle < radii[3] - radii[0]:
                alpha = np.arctan2(centers[0][1] - centers[3][1], centers[0][0] - centers[3][0])
                front_of_dockzone = np.abs(np.mod(alpha - dockzone_circle.theta + np.pi, 2*np.pi) - np.pi  ) < np.pi/2
                if d_to_small_circle < radii[2] - radii[0]: return True
                if not front_of_dockzone: return True
        return False
    
    def _find_key_points_w_exception(self, start_pose: ArrayLike, end_pose: ArrayLike, modes:List[str], radii:List[float], dockzone_theta:float, exception_case:int) -> Tuple[List[ArrayLike],
                                                                                                                                    List[ArrayLike]]:
        cache = {'modes': modes, 'radii': radii, }
        # if the first circle is in the front of the docking zone
        # --> Solve for this path [CSCC]: first circle, straight, intermediate circle, small circle, endpose.
        # if the first circle is in the back of the docking zone
        # --> Solve for this path [CSCCC]: first circle, straight, intermediate circle, large circle, small circle (pi/2), endpose.
        centers = [self._find_center_of_rotation(start_pose, modes[0], radii[0]), None]
        alpha = np.arctan2(centers[0][1] - end_pose[1], centers[0][0] - end_pose[0])
        small_circle_center = self._find_center_of_rotation(end_pose, modes[2], radii[2])
        large_circle_center = end_pose[:2]
        small_circle_radius = radii[2]
        large_circle_radius = radii[3]
        # phi = np.arctan2((end_pose[1] - start_pose[1]), (end_pose[0] - start_pose[0]))
        # beta = np.arctan2((centers[0][1] - start_pose[1]), (centers[0][0] - start_pose[0]))
        # if (np.abs(Spatializer.wrapToPi(beta - phi)) < np.pi/2 and modes[0] == modes[2]):
        #     print('Reject this solution with modes = {}'.format(modes))
        #     return centers, [None, None], radii, modes
        #if np.abs(np.mod(alpha - dockzone_theta + np.pi, 2*np.pi) - np.pi  ) < np.pi/2: # front of the flower
        if exception_case == 0:
            d = np.linalg.norm(centers[0] - small_circle_center)
            # if d > (small_circle_radius + radii[0]):
            #     # if np.abs(Spatializer.wrapToPi(beta - phi)) < np.pi/2:
            #     #     print('Reject this solution with modes = {}'.format(modes))
            #     #     return centers, [None, None], radii, modes
            #     return centers, [None, None], radii, modes
            # CSCC solution, contact to small circle first
            theta = np.arctan2(centers[0][1] - small_circle_center[1], centers[0][0] - small_circle_center[0])
            intermediate_circle_center = small_circle_center + (small_circle_radius - radii[0] )*np.array([np.cos(theta), np.sin(theta)])
            centers = [centers[0], None, intermediate_circle_center, small_circle_center]
            radii = [radii[0], np.inf, radii[0], radii[2]]
            modes = [modes[0], 'S', modes[2], modes[2]]
            tangent_points = self._find_tangent_points_CSC(centers[:3], modes[:3], radii[:3])
            tangent_points.append(small_circle_center + small_circle_radius*np.array([np.cos(theta), np.sin(theta)]))
            # if (not self._is_tangent_points_valid(tangent_points[1:3], centers[1:4], dockzone_theta, large_circle=False)): #or\
            # #     #np.linalg.norm(centers[0] - small_circle_center) > small_circle_radius + radii[0]:
            #     return centers, [None, None], radii, modes
            return centers, tangent_points, radii, modes
        if exception_case == 1:
            # if np.abs(Spatializer.wrapToPi(beta - phi)) < np.pi/2:
            #     print('Reject this solution with modes = {}'.format(modes))
            #     return centers, [None, None], radii, modes
            # TODO: DEBUG THIS CASE
            # CSCCC solution, contact to large circle first
            theta = np.arctan2(centers[0][1] - large_circle_center[1], centers[0][0] - large_circle_center[0])
            intermediate_circle_center = large_circle_center + (large_circle_radius - radii[0] )*np.array([np.cos(theta), np.sin(theta)])
            centers = [centers[0], None, intermediate_circle_center, large_circle_center, small_circle_center]
            radii = [radii[0], np.inf, radii[0], radii[3], radii[2]]
            modes = [modes[0], 'S', modes[2], modes[2], modes[2]]
            tangent_points = self._find_tangent_points_CSC(centers[:3], modes[:3], radii[:3])
            tangent_points.append(large_circle_center + large_circle_radius*np.array([np.cos(theta), np.sin(theta)]))
            alpha = np.arctan2(small_circle_center[1]-large_circle_center[1], small_circle_center[0] - large_circle_center[0])
            tangent_points.append(small_circle_center + small_circle_radius*np.array([np.cos(alpha), np.sin(alpha)]))

            if not self._is_tangent_points_valid(tangent_points[1:3], centers[1:4], dockzone_theta, large_circle=True):
                if not (np.any(tangent_points[0])and np.any(tangent_points[1])): 
                    #print('no solution in expception case 1')
                    return centers, [None, None], radii, modes
                new_start_pose_theta = np.arctan2(tangent_points[1][1] - centers[2][1], tangent_points[1][0] - centers[2][0])
                new_start_pose_theta += np.pi/2 if modes[2]=='L' else -np.pi/2
                new_start_pose = np.concatenate((tangent_points[1], np.asarray([ new_start_pose_theta ])))
                temp_centers, temp_tangent_points, temp_radii, temp_modes = self._find_key_points(new_start_pose, end_pose, [modes[2]]+cache['modes'][1:],
                                                                                                  [radii[2]]+cache['radii'][1:], dockzone_theta)
                if not (np.any(temp_tangent_points[0])and np.any(temp_tangent_points[1])): 
                    # print('Could not find solution after extending the CSCSC path.')
                    return centers, [None, None], radii, modes
                # print('centers', centers[:2]+temp_centers)
                # print('tangent_points', tangent_points[:2]+temp_tangent_points)
                # print('radii', radii[:2]+temp_radii)
                # print('modes', modes[:2]+temp_modes)
                return centers[:2]+temp_centers, tangent_points[:2]+temp_tangent_points, radii[:2]+temp_radii, modes[:2]+temp_modes
            return centers, tangent_points, radii, modes
            #print('FOUND INVALID TANGENT POINTS ON EXCEPTION CASE 1')
            #return centers, [None, None], radii, modes
            #return centers, tangent_points, radii, modes

    ##################################################

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
        valids.append(self._is_tangent_points_valid(candidate_tangent_points[0], centers, dockzone_theta, large_circle=False))
        valids.append(self._is_tangent_points_valid(candidate_tangent_points[1], centers, dockzone_theta, large_circle=True))
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

    def _is_tangent_points_valid(self, tangent_points: List[ArrayLike], centers: List[ArrayLike], dockzone_theta:float, large_circle=True) -> bool:
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


def collision_check(waypoints:ArrayLike, dockzone_circle:DockZoneNotchedCircle, collision_distance:float, collision_azimuth:float):
    waypoints_x = waypoints[0]
    waypoints_y = waypoints[1]
    distances = np.linalg.norm(np.asarray((waypoints_x, waypoints_y)) - np.asarray([dockzone_circle.x, dockzone_circle.y]).reshape(-1,1), axis=0)
    thetas = np.arctan2(waypoints_y - dockzone_circle.y, waypoints_x - dockzone_circle.x)
    idx = np.where(distances < collision_distance)
    if np.sum(np.abs(Spatializer.wrapToPi(thetas[idx] - dockzone_circle.theta) ) > collision_azimuth) > 0:
        return False
    return True


def test_collision_free_check():
    target_pose = np.asarray([0., 0., 0.])
    dockzone_circle = get_dockzone_notched_circle(target_pose)
    ref_dockzone_circle = get_dockzone_notched_circle(np.zeros(3))
    start_pose = np.asarray([1, 0.2, -np.pi/2])
    path_planner = DubinsDockZonePathPlanner()
    path_planner.collision_distance = 0.25 
    path_planner.collision_azimuth = np.pi/3
    fig, ax = plt.subplots(dpi = 100)
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    dockzone_circle_path, = ax.plot(*circle_around_pose_notched(target_pose), 'k:', linewidth=1.0)
    thetas = np.linspace(-np.pi, np.pi, 100)
    collision_circles = ( path_planner.collision_distance*np.cos(thetas) , path_planner.collision_distance*np.sin(thetas) )
    collision_circle_path, = ax.plot(*collision_circles, 'r:', linewidth=2.0)
    idx = np.where(np.abs(Spatializer.wrapToPi(thetas - dockzone_circle.theta)) < path_planner.collision_azimuth)
    opening = (np.concatenate(([target_pose[0]],collision_circles[0][idx], [target_pose[0]] )),
               np.concatenate(([target_pose[1]],collision_circles[1][idx], [target_pose[1]] )) )
    opening_path, = ax.plot(*opening, 'g-', linewidth=5.0, alpha=0.5)
    path = path_planner(start_pose, ref_dockzone_circle)
    path_line, = ax.plot(*path.path_waypoints, 'b--', linewidth=1.0)
    tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
    tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
    path_keypoints, = ax.plot(tangent_points_x, tangent_points_y, 'ko', markersize=3)

    # add widget slider for target_pose[2], path_planner.collision_distance, path_planner.collision_azimuth
    target_pose_slider_ax = fig.add_axes([0.2, 0.98, 0.2, 0.01])
    collision_distance_slider_ax = fig.add_axes([0.2, 0.95, 0.2, 0.01])
    collision_azimuth_slider_ax = fig.add_axes([0.2, 0.92, 0.2, 0.01])
    bat_x_slider_ax = fig.add_axes([0.2, 0.89, 0.2, 0.01])
    bat_y_slider_ax = fig.add_axes([0.2, 0.86, 0.2, 0.01])
    bat_theta_slider_ax = fig.add_axes([0.2, 0.83, 0.2, 0.01])

    target_pose_slider = widgets.Slider(target_pose_slider_ax, '\u03B8_target', -180, 180, valstep=0.1, valinit=np.degrees(target_pose[2]))
    collision_distance_slider = widgets.Slider(collision_distance_slider_ax, 'collision_distance', 0.05, 0.5, valstep=0.01, valinit=path_planner.collision_distance)
    collision_azimuth_slider = widgets.Slider(collision_azimuth_slider_ax, 'collision_azimuth', 0, 180, valstep=0.1, valinit=np.degrees(path_planner.collision_azimuth))
    bat_x_slider = widgets.Slider(bat_x_slider_ax, 'x_bat', -1, 1, valstep=0.01, valinit=start_pose[0])
    bat_y_slider = widgets.Slider(bat_y_slider_ax, 'y_bat', -1, 1, valstep=0.01, valinit=start_pose[1])
    bat_theta_slider = widgets.Slider(bat_theta_slider_ax, '\u03B8_bat', -180, 180, valstep=0.1, valinit=np.degrees(start_pose[2]))


    def update(val):
        start_pose = np.asarray([bat_x_slider.val, bat_y_slider.val, np.radians(bat_theta_slider.val)])
        target_pose[2] = np.radians(target_pose_slider.val)
        dockzone_circle = get_dockzone_notched_circle(target_pose)
        path_planner.collision_distance = collision_distance_slider.val
        path_planner.collision_azimuth = np.radians(collision_azimuth_slider.val)
        dockzone_circle_path.set_data(*circle_around_pose_notched(target_pose))
        thetas = Spatializer.wrapToPi(np.linspace(-np.pi + dockzone_circle.theta, np.pi + dockzone_circle.theta, 100))
        idx = np.where(np.abs(Spatializer.wrapToPi(thetas - dockzone_circle.theta)) < path_planner.collision_azimuth)
        collision_circles = ( path_planner.collision_distance*np.cos(thetas) , path_planner.collision_distance*np.sin(thetas) )
        collision_circle_path.set_data(*collision_circles)
        opening = (np.concatenate(([target_pose[0]],collision_circles[0][idx], [target_pose[0]] )),
                   np.concatenate(([target_pose[1]],collision_circles[1][idx], [target_pose[1]] )) )
        opening_path.set_data(*opening)
        path = path_planner(start_pose, ref_dockzone_circle)
        path_line.set_data(*path.path_waypoints)
        tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
        tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
        path_keypoints.set_xdata(tangent_points_x)
        path_keypoints.set_ydata(tangent_points_y)
        #path = path_planner(start_pose, dockzone_circle)
        #if collision_check(path.path_waypoints, dockzone_circle, path_planner.collision_distance, path_planner.collision_azimuth):
        if path_planner._collision_free_check(path, dockzone_circle):
            path_line.set_color('b')
        else:
            path_line.set_color('r')
        fig.canvas.draw_idle()

    target_pose_slider.on_changed(update)
    collision_distance_slider.on_changed(update)
    collision_azimuth_slider.on_changed(update)
    bat_x_slider.on_changed(update)
    bat_y_slider.on_changed(update)
    bat_theta_slider.on_changed(update)

    plt.show()

def main():
    return utest_widget()
    #return test_collision_free_check()

if __name__ == '__main__':
    main()