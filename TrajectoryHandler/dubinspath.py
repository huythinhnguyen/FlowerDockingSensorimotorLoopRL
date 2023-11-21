from typing import List, Tuple
from numpy.typing import ArrayLike
import numpy as np

from .settings import DockZoneParams, BatKinematicParams
from .dockzone import DockZoneNotchedCircle, get_dockzone_notched_circle_from_flower_pose

from Sensors.FlowerEchoSimulator.Spatializer import wrapToPi


class DubinsParams:
    def __init__(self, **kwargs):
        self.modes: List[str] = kwargs.get('modes', [])
        self.quantities: List[float] = kwargs.get('quantities', [])
        self.cost: float = kwargs.get('cost', 0.)
        self.radii: List[float] = kwargs.get('radii', [])
        self.centers: List[ArrayLike] = kwargs.get('centers', [])
        self.tangent_points: List[ArrayLike] = kwargs.get('tangent_points', [])
        self.path_waypoints: Tuple[ArrayLike] = kwargs.get('path_waypoints', None)


class DubinsDockZonePathPlanner:
    def __init__(self, min_turn_radius: float = BatKinematicParams.MIN_TURNING_RADIUS,
                 collision_distance: float = DockZoneParams.COLLISION_DISTANCE,
                 collision_azimuth: float = DockZoneParams.COLLISION_AZIMUTH,
                 modes_collections: List[List[str]] = None):
        self.min_turn_radius = min_turn_radius
        self.modes_collections = modes_collections if modes_collections is not None else [
            ['L', 'S', 'L', 'L'],
            ['R', 'S', 'R', 'R'],
            ['L', 'S', 'R', 'R'],
            ['R', 'S', 'L', 'L'],
        ]
        self.collision_distance = collision_distance
        self.collision_azimuth = collision_azimuth

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    
    def add_waypoints_to_path(self, path:DubinsParams, waypoint_segment_num_points:int=50, **kwargs) -> DubinsParams:
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
        theta = np.linspace(alpha-path.quantities[0], alpha, waypoint_segment_num_points)
        x = np.cos(theta)*path.radii[0] + path.centers[0][0]
        y = np.sin(theta)*path.radii[0] + path.centers[0][1]
        for i in range(1, len(path.modes)):
            if path.modes[i] == 'S':
                x = np.concatenate((x, np.linspace(tangent_points[i-1][0], tangent_points[i][0], waypoint_segment_num_points)))
                y = np.concatenate((y, np.linspace(tangent_points[i-1][1], tangent_points[i][1], waypoint_segment_num_points)))
            else:
                alpha = np.arctan2(tangent_points[i-1][1] - path.centers[i][1], tangent_points[i-1][0] - path.centers[i][0])
                theta = np.linspace(alpha, alpha+path.quantities[i], waypoint_segment_num_points)
                x = np.concatenate((x, np.cos(theta)*path.radii[i] + path.centers[i][0]))
                y = np.concatenate((y, np.sin(theta)*path.radii[i] + path.centers[i][1]))
        path.path_waypoints = (x, y)
        return path
    
    
    def run(self, start_pose: ArrayLike, target_pose: ArrayLike = None, dockzone_circle: DockZoneNotchedCircle = None, handle_exception:bool=True, **kwargs) -> DubinsParams:
        if target_pose is None and dockzone_circle is None:
            raise ValueError('Need to provide either target_pose or dockzone_circle. If dockzone_circle is provided, the target_pose will be ignored.')
        if dockzone_circle is None:
            dockzone_circle = get_dockzone_notched_circle_from_flower_pose(target_pose, **kwargs)
        #### SOLVE FOR SOLUTIONS WHEN FIRST ROTATION CIRCLE IS OUTSIDE OF DOCKZONE CIRCLE ####
        path = self.plan_path(start_pose, dockzone_circle, **kwargs)
        if path.cost < np.inf: return path
        if not handle_exception: return path
        #### AT THIS POINT, NO SOLUTION IS FOUND. WE WILL HANDLE THIS EXCEPTION ####
        #### SOLVE FOR SOLUTIONS WHEN FIRST ROTATION CIRCLE IS INSIDE OF DOCKZONE CIRCLE ####
        return self.plan_path_with_exception(start_pose, dockzone_circle, **kwargs)
        
    def plan_path(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, **kwargs) -> DubinsParams:
        candidate_paths = []
        best = np.inf
        best_idx = -1
        for i, modes, in enumerate(self.modes_collections):
            path = self._solve_path(start_pose, dockzone_circle, self.min_turn_radius, modes)
            candidate_paths.append(path)
            if path.cost < best:
                # collison check is not needed since if if the solution is found outside the dockzone, it's already collision free.
                best = path.cost
                best_idx = i
        # if solution is found, return it.
        if best < np.inf: return self.add_waypoints_to_path(candidate_paths[best_idx], **kwargs)
        return DubinsParams(cost=np.inf)
    
    def plan_path_with_exception(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, **kwargs) -> DubinsParams:
        candidate_paths = []
        best = np.inf
        best_idx = -1
        for i, modes in enumerate(self.modes_collections):
            for k in range(2):
                path = self._solve_path(start_pose, dockzone_circle, self.min_turn_radius, modes, exception=True, exception_case=k)
                candidate_paths.append(path)
                if path.cost < best:
                    if self._collision_free_check(path, dockzone_circle):
                        best = path.cost
                        best_idx = 2*i + k
        if best<np.inf: return self.add_waypoints_to_path(candidate_paths[best_idx], **kwargs)
        return DubinsParams(cost=np.inf)
    
    def _collision_free_check(self, path: DubinsParams, dockzone_circle: DockZoneNotchedCircle,
                              epsilon:float=1e-3, **kwargs) -> bool: # True is free of collision, False is not.
        if path.cost == np.inf: return False
        cx = dockzone_circle.x - epsilon*np.cos(dockzone_circle.theta)
        cy = dockzone_circle.y - epsilon*np.sin(dockzone_circle.theta)
        path  = self.add_waypoints_to_path(path, n_points=50)
        distances = np.linalg.norm(np.asarray(( path.path_waypoints[0], path.path_waypoints[1] )) \
                                    - np.asarray([cx, cy]).reshape(-1,1), axis=0)
        thetas = np.arctan2(path.path_waypoints[1] - cy, path.path_waypoints[0] - cx)
        idx = np.where(distances < self.collision_distance)
        if np.sum(np.abs(wrapToPi(thetas[idx] - dockzone_circle.theta) ) > self.collision_azimuth) > 0:
            return False
        return True
    
    def _solve_path(self, start_pose: ArrayLike, dockzone_circle: DockZoneNotchedCircle, min_turn_radius: float,
                    modes: List[str], exception:bool=False, exception_case:int=0, **kwargs) -> DubinsParams:
        radii = [min_turn_radius, np.inf, dockzone_circle.radius*0.5, dockzone_circle.radius]
        target_pose = np.asarray([dockzone_circle.x, dockzone_circle.y, wrapToPi(dockzone_circle.theta + np.pi)])
        if exception:
            centers, tangent_points, radii, modes = self._find_keypoints_for_exception(start_pose, target_pose, modes, radii, dockzone_circle, exception_case)
        else: centers, tangent_points, radii, modes = self._find_keypoints(start_pose, target_pose, modes, radii, dockzone_circle)
        quantities = self._compute_quantities(start_pose, target_pose, centers, tangent_points, modes)
        cost = self._compute_cost(radii, quantities, modes)
        return DubinsParams(modes=modes, quantities=quantities, cost=cost, radii=radii, centers=centers, tangent_points=tangent_points)
    
    def _find_keypoints(self, start_pose: ArrayLike, target_pose: ArrayLike, modes: List[str],
                        radii: List[float], dockzone_circle: DockZoneNotchedCircle) -> Tuple[List[ArrayLike], List[ArrayLike], List[float], List[str]]:
        centers = []
        for pose, mode, radius in zip([start_pose, start_pose, target_pose], modes, radii):
            centers.append(self._find_center_of_rotation(pose, mode, radius))
        centers.append(target_pose[:2])
        tangent_points, centers, radii, modes = self._find_tangent_points(centers, radii, modes, dockzone_circle)
        return centers, tangent_points, radii, modes
    
    def _find_keypoints_for_exception(self, start_pose: ArrayLike, target_pose: ArrayLike, modes: List[str],
                                      radii: List[float], dockzone_circle: DockZoneNotchedCircle,
                                      exception_case: int) -> Tuple[List[ArrayLike], List[ArrayLike], List[float], List[str]]:
        cache = {'modes': modes, 'radii': radii}
        centers = [self._find_center_of_rotation(start_pose, modes[0], radii[0]), None]
        small_circle_center = self._find_center_of_rotation(target_pose, modes[2], radii[2])
        small_circle_radius = radii[2]
        large_circle_center = target_pose[:2]
        large_circle_radius = radii[3]
        if exception_case == 0:
            theta = np.arctan2(centers[0][1] - small_circle_center[1], centers[0][0] - small_circle_center[0])
            # TODO: Uncomment this to test the new code!
            if not self._is_tangent_points_valid([centers[0]], dockzone_circle, large_circle=False): theta += np.pi
            intermediate_circle_center = small_circle_center + (small_circle_radius - radii[0] )*np.array([np.cos(theta), np.sin(theta)])
            centers = [centers[0], None, intermediate_circle_center, small_circle_center]
            radii = [radii[0], np.inf, radii[0], small_circle_radius]
            modes = [modes[0], 'S', modes[2], modes[2]]
            tangent_points = self._find_tangent_points_CSC(centers[:3], modes[:3], radii[:3])
            tangent_points.append(small_circle_center + small_circle_radius*np.array([np.cos(theta), np.sin(theta)]))
            return centers, tangent_points, radii, modes
        if exception_case == 1:
            theta = np.arctan2(centers[0][1] - large_circle_center[1], centers[0][0] - large_circle_center[0])
            # TODO: Uncomment this to test the new code! --> No, no this is bad idea!!! It will lead to collision path.
            # if not self._is_tangent_points_valid([centers[0]], dockzone_circle, large_circle=True): theta += np.pi
            intermediate_circle_center = large_circle_center + (large_circle_radius - radii[0] )*np.array([np.cos(theta), np.sin(theta)])
            centers = [centers[0], None, intermediate_circle_center, large_circle_center, small_circle_center]
            radii = [radii[0], np.inf, radii[0], large_circle_radius, small_circle_radius]
            modes = [modes[0], 'S', modes[2], modes[2], modes[2]]
            tangent_points = self._find_tangent_points_CSC(centers[:3], modes[:3], radii[:3])
            tangent_points.append(large_circle_center + large_circle_radius*np.asarray([np.cos(theta), np.sin(theta)]))
            alpha = np.arctan2(small_circle_center[1]-large_circle_center[1], small_circle_center[0] - large_circle_center[0])
            tangent_points.append(small_circle_center + small_circle_radius*np.array([np.cos(alpha), np.sin(alpha)]))
            if not self._is_tangent_points_valid(tangent_points[1:3], dockzone_circle, large_circle=True):
                if not (np.any(tangent_points[0])and np.any(tangent_points[1])): 
                    #print('no solution in expception case 1')
                    return centers, [None, None], radii, modes
                new_start_pose_theta = np.arctan2(tangent_points[1][1] - centers[2][1], tangent_points[1][0] - centers[2][0])
                new_start_pose_theta += np.pi/2 if modes[2]=='L' else -np.pi/2
                new_start_pose = np.concatenate((tangent_points[1], np.asarray([ new_start_pose_theta ])))
                temp_centers, temp_tangent_points, temp_radii, temp_modes = self._find_keypoints(new_start_pose, target_pose, [modes[2]]+cache['modes'][1:],
                                                                                                  [radii[2]]+cache['radii'][1:], dockzone_circle)
                if not (np.any(temp_tangent_points[0])and np.any(temp_tangent_points[1])): 
                    return centers, [None, None], radii, modes
                return centers[:2]+temp_centers, tangent_points[:2]+temp_tangent_points, radii[:2]+temp_radii, modes[:2]+temp_modes
            return centers, tangent_points, radii, modes

        
    def _find_tangent_points(self, centers: List[ArrayLike], radii: List[float],
                             modes: List[str], dockzone_circle: DockZoneNotchedCircle) -> Tuple[List[ArrayLike], List[ArrayLike], List[float], List[str]]:
        candidate_tangent_points = []
        # find candidates to the small circle.
        candidate_tangent_points.append(self._find_tangent_points_CSC(centers[:2]+[centers[2]], modes[:2]+[modes[2]], radii[:2]+[radii[2]]))
        # find candidates to the large circle.
        candidate_tangent_points.append(self._find_tangent_points_CSC(centers[:2]+[centers[3]], modes[:2]+[modes[3]], radii[:2]+[radii[3]]))
        #valids = [self._is_tangent_points_valid(candidate, dockzone_circle, lc)  for candidate, lc in zip(candidate_tangent_points, [False, True])]
        valids = []
        valids.append(self._is_tangent_points_valid(candidate_tangent_points[0], dockzone_circle, large_circle=False))
        valids.append(self._is_tangent_points_valid(candidate_tangent_points[1], dockzone_circle, large_circle=True))
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
            theta = np.arctan2(centers[3][1] - centers[2][1], centers[3][0] - centers[2][0])
            intersection_between_two_circles = centers[2] + radii[2]*np.array([np.cos(theta), np.sin(theta)])
            tangent_points = candidate_tangent_points[1] + [intersection_between_two_circles]
            modes = modes[:2] + [modes[3], modes[2]]
        return tangent_points, centers, radii, modes
        
    def _find_tangent_points_CSC(self, centers: List[ArrayLike], modes: List[str], radii: List[float]) -> List[ArrayLike]:
        # distance between 2 centers
        d = np.linalg.norm(centers[0] - centers[-1])
        if d < (radii[0]+radii[-1]) and (modes[0]!= modes[-1]): return [None, None] # no solution found
        if d < np.abs(radii[0]-radii[-1]): return [None, None] # no solution found
        # find the angle between 2 centers
        theta = np.arctan2(centers[-1][1] - centers[0][1], centers[-1][0] - centers[0][0])
        tangent_points = []
        if modes[0] == modes[-1]:
            phi = np.pi - np.arccos(np.sign(radii[-1]-radii[0])*(radii[-1] - radii[0])/d)
            alpha = theta - phi if modes[0] == 'L' else theta + phi
            beta = alpha
        else:
            phi = np.arccos((radii[0] + radii[-1])/d)
            if modes[0] == 'L': # modes[2] == 'R'
                alpha = theta - phi
                beta = theta - phi - np.pi
            elif modes[0] == 'R': # modes[2] == 'L'
                alpha = theta + phi
                beta = theta + phi + np.pi
        tangent_points.append(centers[0] + radii[0]*np.array([np.cos(alpha), np.sin(alpha)]))
        tangent_points.append(centers[-1] + radii[-1]*np.array([np.cos(beta), np.sin(beta)]))
        return tangent_points

    # Valid if on the front side of dockzone for small circle tangent point; backside of the dockzone for large circle tangent point.
    def _is_tangent_points_valid(self, tangent_points: List[ArrayLike], dockzone_circle: DockZoneNotchedCircle, large_circle:bool=True) -> bool:
        tangent_point = tangent_points[-1]
        if not np.any(tangent_point): return False
        alpha = np.arctan2(tangent_point[1] - dockzone_circle.y, tangent_point[0] - dockzone_circle.x)
        if not( (np.abs(wrapToPi(alpha - dockzone_circle.theta)) >= np.pi/2) ^ large_circle ): return True
        return False

    def _find_center_of_rotation(self, pose: ArrayLike, mode: str, turning_radius: float) -> ArrayLike:
        if mode == 'S': return None
        if mode == 'L': return pose[:2] + turning_radius * np.asarray([np.cos(pose[2]+0.5*np.pi), np.sin(pose[2]+0.5*np.pi)])
        if mode == 'R': return pose[:2] + turning_radius * np.asarray([np.cos(pose[2]-0.5*np.pi), np.sin(pose[2]-0.5*np.pi)])
        raise ValueError('mode must be one of "S", "L", "R" but got {}'.format(mode))


    def _compute_cost(self, radii: List[float], quantities:List[float], modes: List[str]) -> float:
        cost = 0.
        for i in range(len(modes)):
            if modes[i] == 'S': cost += quantities[i]
            else:
                cost += radii[i] * np.abs(quantities[i])
        return cost

    def _compute_quantities(self, start_pose: ArrayLike, target_pose: ArrayLike,
                            centers: List[ArrayLike], tangent_points: List[ArrayLike], modes: List[str]) -> List[float]:
        for tangent_point in tangent_points:
            if not np.any(tangent_point): return [np.inf]*(len(modes))
        quantities = []
        checkpoint_poses = [start_pose] + tangent_points + [target_pose]
        for i, m in enumerate(modes):
            first_pose = checkpoint_poses[i]
            second_pose = checkpoint_poses[i+1]
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