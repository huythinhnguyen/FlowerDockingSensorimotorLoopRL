import sys
import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import warnings


REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting
from Simulation.Motion import State

from Perception.flower_pose_estimator import NaiveOneShotFlowerPoseEstimator, OnsetOneShotFlowerPoseEstimator, TwoShotFlowerPoseEstimator
from SensorimotorLoops.home_in_flower import HomeInFlower

from TrajectoryHandler.dubinspath import DubinsDockZonePathPlanner, DubinsToKinematics, DubinsToKinematicsNoAccel, DubinsParams
from TrajectoryHandler.dockzone import generate_dockzone_notched_circle_waypoints, get_dockzone_notched_circle_from_flower_pose
from TrajectoryHandler.settings import BatKinematicParams, OtherKinematicParams, DockZoneParams

SAVE_DATA_PATH = os.path.join(REPO_PATH, 'experiments', 'data', 'exp_1')

FLOWER_DISTANCE_SETTINGS = {
    'close': 0.8, # meter
    'mid':   1.6, # meter
    'far':   2.4, # meter
}

BAT_AZIMUTH_RANGE: Tuple[float] = (-np.pi/3, np.pi/3)
FLOWER_COLLISION_RADIUS: float = 0.22
FLOWER_OPENING_ANGULAR_RANGE: Tuple[float] = (-4*np.pi/18, 4*np.pi/18)
BAT_FACING_ANGULAR_RANGE: Tuple[float] = (-7*np.pi/18, 7*np.pi/18)
ARENA_LIM = {'x': (-1., 4.), 'y': (-2.5, 2.5)}
N_TRIAL: int = 5_000

INIT_VELOCITY: float = 0.


def convert_polar_to_cartesian(bat_pose: ArrayLike,
                               flower_distance: float, flower_azimuth: float, flower_orientation: float,) -> ArrayLike:
    # return flower cartesian pose (x,y,theta)
    # bat_pose: (x,y,theta)
    # flower_distance: float
    # flower_azimuth: float
    # flower_orientation: float
    # return: (x,y,theta)
    flower_pose = np.zeros(3)
    flower_pose[0] = bat_pose[0] + flower_distance*np.cos(bat_pose[2] + flower_azimuth)
    flower_pose[1] = bat_pose[1] + flower_distance*np.sin(bat_pose[2] + flower_azimuth)
    flower_pose[2] = Spatializer.wrapToPi(bat_pose[2] + flower_azimuth + np.pi - flower_orientation)
    return flower_pose

# TESTED
def collision_check(bat_pose: ArrayLike, objects: ArrayLike,
                    collision_distance: float = FLOWER_COLLISION_RADIUS) -> bool:
    if np.any(np.linalg.norm(objects[:,:2]-bat_pose[:2], axis=1) <= collision_distance):
        return True
    return False

def check_out_of_arena(bat_pose: ArrayLike,
                       arena_lim = ARENA_LIM) -> bool:
    if not (arena_lim['x'][0] <= bat_pose[0] <= arena_lim['x'][1]):
        return True
    if not (arena_lim['y'][0] <= bat_pose[1] <= arena_lim['y'][1]):
        return True
    return False

# TESTED
def get_angle_of_arrival(bat_pose: ArrayLike, flower_pose: ArrayLike) -> float:
    flower_to_bat = bat_pose[:2] - flower_pose[:2]
    flower_to_bat_angle = np.arctan2(flower_to_bat[1], flower_to_bat[0])
    return Spatializer.wrapToPi(flower_to_bat_angle - flower_pose[2])

# TESTED  
def get_azimuth_of_flower(bat_pose: ArrayLike, flower_pose: ArrayLike) -> float:
    bat_to_flower = flower_pose[:2] - bat_pose[:2]
    bat_to_flower_angle = np.arctan2(bat_to_flower[1], bat_to_flower[0])
    return Spatializer.wrapToPi(bat_to_flower_angle - bat_pose[2])

# TESTED            
def check_docking(bat_pose: ArrayLike, flower_pose: ArrayLike,
                  angle_of_arrival: float = None, azimuth_of_flower: float = None,
                  collision_distance: float = FLOWER_COLLISION_RADIUS,
                  opening_angular_range: Tuple[float, float] = FLOWER_OPENING_ANGULAR_RANGE,
                  facing_angular_range: Tuple[float, float] = BAT_FACING_ANGULAR_RANGE) -> bool:
        if np.linalg.norm(bat_pose[:2]-flower_pose[:2]) > collision_distance:
            error_msg = f'Bat distance to flower is {np.round(np.linalg.norm(bat_pose[:2]-flower_pose[:2]),3)} \
                which is greater than the collision distance {collision_distance}.\n \
                check_docking() should only be used after collision_check() is True.'
            raise ValueError(error_msg)
        # compute angle of arrival and azimuth of flower if not provided.
        if angle_of_arrival is None: angle_of_arrival = get_angle_of_arrival(bat_pose, flower_pose)
        if azimuth_of_flower is None: azimuth_of_flower = get_azimuth_of_flower(bat_pose, flower_pose)
        # check if the bat is facing the flower.
        if facing_angular_range[0] <= azimuth_of_flower <= facing_angular_range[1]:
            if opening_angular_range[0] <= angle_of_arrival <= opening_angular_range[1]:
                # Docking is successful.
                return True
        # Docking is not successful.
        return False

def calculate_optimal_path_length(bat_pose: ArrayLike, flower_pose: ArrayLike) -> float:
    theta = np.pi - 2*np.arccos(FLOWER_COLLISION_RADIUS/2*DockZoneParams.SMALL_CIRCLE_RADIUS)
    
    path_planner = DubinsDockZonePathPlanner()
    dockzone = get_dockzone_notched_circle_from_flower_pose(flower_pose)
    path = path_planner(bat_pose, dockzone)

    if np.abs(path.quantities[-1]) > theta and path.modes[-1]!='S':
        return path.cost - theta*DockZoneParams.SMALL_CIRCLE_RADIUS
    if path.modes[-2]=='S' and path.modes[-1]!='S':
        phi = path.quantities[-1]
        b = DockZoneParams.SMALL_CIRCLE_RADIUS
        a = FLOWER_COLLISION_RADIUS
        c = b*(1-np.cos(phi))/np.cos(phi)
        y = np.sqrt((b+c)**2 - b**2)
        h = c*np.cos(phi)
        d = np.sqrt(a**2 - h**2) + np.sqrt(c**2 - h**2)
        x = d - y
        return path.cost - phi*DockZoneParams.SMALL_CIRCLE_RADIUS - x
    warnings.warn('The optimal path length is not calculated correctly.')
    return path.cost


def run_1_trial(bat_pose: ArrayLike, flower_pose: ArrayLike,
                timeout = 10, track_bat_pose = False) -> Dict[str, Any]:
    # return a dictionary of the trial result.
    # bat_pose: (x,y,theta)
    # flower_pose: (x,y,theta)
    # results to track:
    # initial config:
    #   bat_pose: (x,y,theta)
    #   flower_pose: (x,y,theta)
    #   initial estimated flower pose: (x,y,theta)
    #   optimal path length
    # ending:
    #   bat_pose: (x,y,theta)
    #   estimated flower pose at the ending bat pose.
    #   angle of arrival
    #   ending azimuth of the flower
    #   outcomes
    #   travel distance
    # the steps leading to the ending:
    #   bat_pose: (x,y,theta)
    #   estimated flower pose
    result = {
        'init_bat_pose': bat_pose, #
        'init_flower_pose': flower_pose, #
        'init_estimated_flower_pose': np.array([np.nan, np.nan, np.nan]), #
        'optimal_path_length': calculate_optimal_path_length(bat_pose, flower_pose), #
        'ending_bat_pose': np.array([np.nan, np.nan, np.nan]), #
        'ending_estimated_flower_pose': np.array([np.nan, np.nan, np.nan]), #
        'angle_of_arrival': np.nan, #
        'ending_azimuth_of_flower': np.nan, #
        'outcomes': 'miss', #
        'travel_distance': np.nan, #
        'final_leading_bat_pose': np.array([np.nan, np.nan, np.nan]), #
        'final_estimated_flower_pose': np.array([np.nan, np.nan, np.nan]), #
    }
    #############################################
    render = Spatializer.Render()
    render.compression_filter = Spatializer.Compressing.Downsample512()
    state = State(pose = bat_pose, kinematic=[INIT_VELOCITY, 0.], dt = 1/BatKinematicParams.CHIRP_RATE,
                  max_linear_velocity=BatKinematicParams.MAX_LINEAR_VELOCITY,
                  max_angular_velocity=BatKinematicParams.MAX_ANGULAR_VELOCITY*1,
                  max_linear_acceleration=OtherKinematicParams.LINEAR_ACCEL_LIMIT,
                  max_angular_acceleration=BatKinematicParams.MAX_ANGULAR_ACCELERATION*1,
                  max_linear_deceleration=OtherKinematicParams.LINEAR_DECEL_LIMIT,
                  )
    objects = np.hstack([flower_pose, 3.]).astype(np.float32).reshape(1,4)
    control_loop = HomeInFlower(pose_estimator=OnsetOneShotFlowerPoseEstimator(),
                                init_v=INIT_VELOCITY, caching=True)
    distance_traveled = 0.
    if track_bat_pose: bat_poses = [state.pose]
    while not collision_check(state.pose, np.array([flower_pose])):
        if check_out_of_arena(state.pose): break
        envelope_left, envelope_right = render(state.pose, objects).values()
        control_loop.init_v = state.kinematic[0]
        vs, ws = control_loop(envelope_left, envelope_right)
        if control_loop.cache['prediction'][0]:
            est_flower_pose = convert_polar_to_cartesian(state.pose, *control_loop.cache['prediction'])
            if np.isnan(result['init_estimated_flower_pose'][0]):
                result['init_estimated_flower_pose'] = est_flower_pose
        else: est_flower_pose = np.array([np.nan, np.nan, np.nan])
        leading_bat_pose = state.pose
        for v, w in zip(vs, ws):
            state(v=v, w=w)
            distance_traveled += state.kinematic[0]*state.dt
            if track_bat_pose: bat_poses.append(state.pose)
            if collision_check(state.pose, objects):
                angle_of_arrival = get_angle_of_arrival(state.pose, flower_pose)
                azimuth_of_flower = get_azimuth_of_flower(state.pose, flower_pose)
                if check_docking(state.pose, flower_pose, angle_of_arrival, azimuth_of_flower):
                    result['outcomes'] = 'dock'
                else: result['outcomes'] = 'hit'
                result['angle_of_arrival'] = angle_of_arrival
                result['ending_azimuth_of_flower'] = azimuth_of_flower
                result['final_leading_bat_pose'] = leading_bat_pose
                result['final_estimated_flower_pose'] = est_flower_pose
                result['ending_bat_pose'] = state.pose
                result['travel_distance'] = distance_traveled
                envelope_left, envelope_right = render(state.pose, objects).values()
                control_loop(envelope_left, envelope_right)
                if control_loop.cache['prediction'][0]:
                    result['ending_estimated_flower_pose'] = convert_polar_to_cartesian(state.pose, *control_loop.cache['prediction'])
                break
    if track_bat_pose: result['bat_poses'] = np.asarray(bat_poses)
    return result

def test2(bat_angle_degree, flower_angle_degree, distance_setting):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    bat_init_pose = np.array([0., 0., np.radians(bat_angle_degree)])
    flower_pose = np.array([FLOWER_DISTANCE_SETTINGS[distance_setting], 0., np.radians(flower_angle_degree)])
    result = run_1_trial(bat_init_pose, flower_pose, track_bat_pose=True)
    fig, ax = plt.subplots()
    ax.set_xlim(ARENA_LIM['x'][0], ARENA_LIM['x'][1])
    ax.set_ylim(ARENA_LIM['y'][0], ARENA_LIM['y'][1])
    ax.set_aspect('equal')
    ax.grid(True)
    if result['outcomes'] == 'dock' or result['outcomes'] == 'hit':
        bat_end_arrow = FancyArrowPatch((result['ending_bat_pose'][0], result['ending_bat_pose'][1]),
                                    (result['ending_bat_pose'][0]+0.3*np.cos(result['ending_bat_pose'][2]),
                                     result['ending_bat_pose'][1]+0.3*np.sin(result['ending_bat_pose'][2])),
                                arrowstyle='simple', mutation_scale=10, color='blue', alpha=0.5)
        ax.add_patch(bat_end_arrow)
    bat_arrow = FancyArrowPatch((result['init_bat_pose'][0], result['init_bat_pose'][1]),
                            (result['init_bat_pose'][0]+0.3*np.cos(result['init_bat_pose'][2]),
                                result['init_bat_pose'][1]+0.3*np.sin(result['init_bat_pose'][2])),
                            arrowstyle='simple', mutation_scale=10, color='black', alpha=0.5)
    
    flower_arrow = FancyArrowPatch((result['init_flower_pose'][0], result['init_flower_pose'][1]),
                                (result['init_flower_pose'][0]+0.3*np.cos(result['init_flower_pose'][2]),
                                result['init_flower_pose'][1]+0.3*np.sin(result['init_flower_pose'][2])),
                                arrowstyle='simple', mutation_scale=20, color='magenta', alpha=0.3)
    bat_trajectory_line = ax.plot(result['bat_poses'][:,0], result['bat_poses'][:,1], color='blue', alpha=0.5)
    ax.add_patch(bat_arrow)
    ax.add_patch(flower_arrow)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.text(ARENA_LIM['x'][1]+0.1, ARENA_LIM['y'][1]-0.5,
            'ending=' + result['outcomes'], ha='left', va='center', fontsize=10)
    aoa = np.round(np.degrees(result['angle_of_arrival']), 1)
    ax.text(ARENA_LIM['x'][1]+0.1, ARENA_LIM['y'][1]-0.8,
            'aoa={}'.format(aoa), ha='left', va='center', fontsize=10)
    print((result['travel_distance'])/result['optimal_path_length'])
    plt.show()

def test3(bat_angle_degree, flower_angle_degree, distance_setting):
    for i in range(5):
        test2(bat_angle_degree, flower_angle_degree, distance_setting)


def run(num_trial, distance_setting,
        save_path = SAVE_DATA_PATH, save_overwrite = True, save_file_interval = 100):
    import pandas as pd
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_file = os.path.join(save_path, f'results_{distance_setting}.pkl')
    # load save file if exists and save_overwrite is False
    if os.path.exists(save_file) and not save_overwrite:
        results = pd.read_pickle(save_file).to_dict(orient='list')
    else:
        results = {
            'init_bat_pose': [],
            'init_flower_pose': [],
            'init_estimated_flower_pose': [],
            'optimal_path_length': [],
            'ending_bat_pose': [],
            'ending_estimated_flower_pose': [],
            'angle_of_arrival': [],
            'ending_azimuth_of_flower': [],
            'outcomes': [],
            'travel_distance': [],
            'final_leading_bat_pose': [],
            'final_estimated_flower_pose': [],
            'bat_poses': [],
        }
    for i in range(len(results['init_bat_pose']), num_trial):
        bat_init_pose = np.array([0.,0.,
            np.random.uniform(low=BAT_AZIMUTH_RANGE[0], high= BAT_AZIMUTH_RANGE[1])])
        flower_pose = np.array([FLOWER_DISTANCE_SETTINGS[distance_setting], 0.,
                np.random.uniform(-np.pi, np.pi)])
        result = run_1_trial(bat_init_pose, flower_pose, track_bat_pose=True)
        for key in results.keys():
            results[key].append(result[key])
        if i % save_file_interval == 0:
            df = pd.DataFrame(results)
            df.to_pickle(save_file)
            print(f'{i} trials are saved.')
    print(f'{num_trial} trials are completed.')
    df = pd.DataFrame(results)
    df.to_pickle(save_file)
    print('All trials are saved.')
    return None    



def main():
    # return test2(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
    # return test3(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
    if len(sys.argv) > 2:
        overwrite = True if sys.argv[2] in ['ow', 'overwrite'] else False
    else: overwrite = False
    return run(N_TRIAL, sys.argv[1])

if __name__=='__main__':
    main()



# def test1():
#     # make a interactive widget to plot arrows showing the bat and flower pose.
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import FancyArrowPatch
#     from matplotlib.widgets import Slider
#     from matplotlib import animation
#     from matplotlib.animation import FuncAnimation
#     from matplotlib import patches
#     from matplotlib import transforms
#     from matplotlib import cm

#     fig, ax = plt.subplots()
#     ax.set_xlim(-1, 2)
#     ax.set_ylim(-1, 2)
#     ax.set_aspect('equal')
#     ax.grid(True)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title('Bat and Flower Pose')

#     bat_pose = np.array([0., 0., 0.])
#     flower_pose = np.array([1., 0., 0.])
#     bat_arrow = FancyArrowPatch((bat_pose[0], bat_pose[1]), (bat_pose[0]+0.3*np.cos(bat_pose[2]), bat_pose[1]+0.3*np.sin(bat_pose[2])),
#                                 arrowstyle='simple', mutation_scale=20, color='blue')
#     flower_arrow = FancyArrowPatch((flower_pose[0], flower_pose[1]), (flower_pose[0]+0.3*np.cos(flower_pose[2]), flower_pose[1]+0.3*np.sin(flower_pose[2])),
#                                       arrowstyle='simple', mutation_scale=20, color='red')
#     ax.add_patch(bat_arrow)
#     ax.add_patch(flower_arrow)
#     fig.subplots_adjust(bottom=0.3)
#     axcolor = 'lightgoldenrodyellow'
#     ax_x = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
#     ax_y = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
#     ax_theta = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)
#     ax_flower_theta = plt.axes([0.25, 0.16, 0.65, 0.03], facecolor=axcolor)
    
#     # add a changeable textbox show the sign of bat pose [2]

#     status_text = ax.text(2.2, 1.8, 'STATUS: ??', ha='left', va='center', fontsize=14)
#     distance_text = ax.text(2.2, 1.5, 'd = ??', ha='left', va='center', fontsize=14)
#     aoa_text = ax.text(2.2, 1.2, 'aoa = ??', ha='left', va='center', fontsize=14)
#     azi_text = ax.text(2.2, 0.8, 'azi = ??', ha='left', va='center', fontsize=14)

#     s_x = Slider(ax_x, 'x', ARENA_LIM['x'][0], ARENA_LIM['x'][1], valinit=bat_pose[0])
#     s_y = Slider(ax_y, 'y', ARENA_LIM['y'][0], ARENA_LIM['y'][1], valinit=bat_pose[1])
#     s_theta = Slider(ax_theta, 'theta', -np.pi, np.pi, valinit=bat_pose[2])
#     s_flower_theta = Slider(ax_flower_theta, 'flower_theta', -np.pi, np.pi, valinit=flower_pose[2])

#     def update(val):
#         bat_pose[0] = s_x.val
#         bat_pose[1] = s_y.val
#         bat_pose[2] = s_theta.val
#         flower_pose[2] = s_flower_theta.val
#         bat_arrow.set_positions((bat_pose[0], bat_pose[1]), (bat_pose[0]+0.3*np.cos(bat_pose[2]), bat_pose[1]+0.3*np.sin(bat_pose[2])))
#         flower_arrow.set_positions((flower_pose[0], flower_pose[1]), (flower_pose[0]+0.3*np.cos(flower_pose[2]), flower_pose[1]+0.3*np.sin(flower_pose[2])))
#         fig.canvas.draw_idle()
#         distance = np.linalg.norm(bat_pose[:2]-flower_pose[:2])
#         distance_text.set_text(f'd = {np.round(distance, 3)}')
#         aoa = get_angle_of_arrival(bat_pose, flower_pose)
#         aoa_text.set_text(f'aoa = {np.round(np.degrees(aoa), 1)}')
#         azi = get_azimuth_of_flower(bat_pose, flower_pose)
#         azi_text.set_text(f'azi = {np.round(np.degrees(azi), 1)}')
#         if collision_check(bat_pose, np.array([flower_pose])):
#             if check_docking(bat_pose, flower_pose):
#                 status_text.set_text('STATUS: Dock')
#             else:
#                 status_text.set_text('STATUS: Hit')
#         else: status_text.set_text('STATUS: OK')


#     s_x.on_changed(update)
#     s_y.on_changed(update)
#     s_theta.on_changed(update)
#     s_flower_theta.on_changed(update)

#     plt.show()
