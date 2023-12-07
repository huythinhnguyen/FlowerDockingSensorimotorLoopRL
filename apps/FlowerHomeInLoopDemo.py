import sys
import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec
import pickle
import time

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting
from Simulation.Motion import State

from Perception.flower_pose_estimator import NaiveOneShotFlowerPoseEstimator, OnsetOneShotFlowerPoseEstimator, TwoShotFlowerPoseEstimator
from SensorimotorLoops.home_in_flower import HomeInFlower

PRESENCE_DETECTION_THRESHOLD = 1.
JITTER = 0.2
FLOWER_XY_ANCHOR = 1.5

EXECUTION_STEPS_PER_ESTIMATION = 20
# TODO: Dynamic execution steps based on distance estimation is probably WRONG.
# I will leeds to bat looping a round the flowers in some edge cases.
# Here is a better proposal: --> Will need to modify DubinsToKinematics to support this.
# 1. Allow DubinsToKinematics to track the len (number of points) for the kinematic sequence in each segment ('L', 'R', 'S')
# 2. Execute N steps in the kinematic sequence,
# 3. N = len(sequence) - int(len(last_segment_sequence)*0.5)
ENDING_COLLISION_DISTANCE = 0.25
DISTANCE_MEMORY_SIZE = 5

FLOWER_ARROW_LENGTH = 0.2
BAT_ARROW_LENGTH = 0.3

FONT = {'size': 10}

from TrajectoryHandler.dubinspath import DubinsDockZonePathPlanner, DubinsToKinematics, DubinsToKinematicsNoAccel, DubinsParams
from TrajectoryHandler.dockzone import generate_dockzone_notched_circle_waypoints, get_dockzone_notched_circle_from_flower_pose
from TrajectoryHandler.settings import BatKinematicParams, OtherKinematicParams

def transform_coordinates(Xs, Ys, a, b, theta):
    # Transformation matrix
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), a],
                                [np.sin(theta), np.cos(theta), b],
                                [0, 0, 1]])

    # Homogeneous coordinates of input points
    homogeneous_coords = np.vstack([Xs, Ys, np.ones_like(Xs)])

    # Apply transformation
    transformed_coords = np.dot(transform_matrix, homogeneous_coords)

    # Extract transformed Xs and Ys
    transformed_Xs = transformed_coords[0, :]
    transformed_Ys = transformed_coords[1, :]

    return transformed_Xs, transformed_Ys

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

def set_up_waveform_plot(ax: plt.Axes, distances: ArrayLike, data_left: ArrayLike, data_right: ArrayLike, title: Optional[str]=None, fontsize: int=10):
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
    
def collision_check(bat_pose: ArrayLike, objects: ArrayLike, collision_distance: float = ENDING_COLLISION_DISTANCE, bounds: float= 10.) -> bool:
    # return True if collision happens, False otherwise
    # bat_pose: (x,y,theta)
    # objects: (n,3) array, each row is (x,y,theta)
    # collision_distance: float
    # return: bool
    if np.any(np.linalg.norm(objects[:,:2]-bat_pose[:2], axis=1) <= collision_distance):
        return True
    elif np.abs(bat_pose[0]) >= bounds or np.abs(bat_pose[1]) >= bounds:
        return True
    else:
        return False
 
def run_episode(bat_pose: ArrayLike, objects: ArrayLike,
                renderer: Spatializer.Render, pose_estimator: NaiveOneShotFlowerPoseEstimator,
                init_v:float = 0., timeout: float = 5., **kwargs):
    tic = time.time()
    state = State(pose = bat_pose, kinematic=[init_v, 0.], dt=1/BatKinematicParams.CHIRP_RATE,
                  max_linear_velocity=BatKinematicParams.MAX_LINEAR_VELOCITY,
                  max_angular_velocity=BatKinematicParams.MAX_ANGULAR_VELOCITY*1,
                  max_linear_acceleration=OtherKinematicParams.LINEAR_ACCEL_LIMIT,
                  max_angular_acceleration=BatKinematicParams.MAX_ANGULAR_ACCELERATION*1,
                  max_linear_deceleration=OtherKinematicParams.LINEAR_DECEL_LIMIT,
                  )
    
    control_loop = HomeInFlower(pose_estimator, init_v=init_v, caching=True)
    bat_trajectory = np.asarray(state.pose).astype(np.float32).reshape(1,3)
    envelope_left_record = np.asarray([]).astype(np.float32).reshape(0,512)
    envelope_right_record = np.asarray([]).astype(np.float32).reshape(0,512)
    inputs_left_record = np.asarray([]).reshape(0,512)
    inputs_right_record = np.asarray([]).reshape(0,512)
    flower_pose_record = np.asarray([]).astype(np.float32).reshape(0,3)
    paths = []
    use_estimation = []
    marker_record = []
    predicted_distances = []
    v_record = []
    w_record = []
    
    while not collision_check(state.pose, objects):
        while time.time() - tic < timeout: break
        envelope_left, envelope_right = renderer(state.pose, objects).values()
        control_loop.init_v = state.kinematic[0]
        vs, ws = control_loop(envelope_left, envelope_right)
        est_flower_pose = convert_polar_to_cartesian(state.pose, *control_loop.cache['prediction']) if control_loop.cache['prediction'][0] else np.asarray([np.nan, np.nan, np.nan])
        for v, w in zip(vs, ws):
            state(v=v, w=w)
            bat_trajectory = np.vstack((bat_trajectory, state.pose))
            v_record.append(state.kinematic[0])
            w_record.append(state.kinematic[1])
            if time.time() - tic > timeout: break
            if collision_check(state.pose, objects): break
        marker_record.append(len(bat_trajectory))
        predicted_distances.append(control_loop.cache['prediction'][0] if control_loop.cache['prediction'][0] else np.nan)
        envelope_left_record = np.vstack((envelope_left_record, envelope_left))
        envelope_right_record = np.vstack((envelope_right_record, envelope_right))
        inputs_left_record = np.vstack((inputs_left_record, control_loop.pose_estimator.cache['inputs'][0,0,:]))
        inputs_right_record = np.vstack((inputs_right_record, control_loop.pose_estimator.cache['inputs'][0,1,:] ))
        flower_pose_record = np.vstack((flower_pose_record, est_flower_pose))
        paths.append(control_loop.cache['path'])
        use_estimation.append(control_loop.cache['use_prediction'])
    return {'bat_trajectory': bat_trajectory,
            'envelope_left_record': envelope_left_record,
            'envelope_right_record': envelope_right_record,
            'inputs_left_record': inputs_left_record,
            'inputs_right_record': inputs_right_record,
            'flower_pose_record': flower_pose_record,
            'predicted_distances': np.asarray(predicted_distances),
            'marker_record': np.asarray(marker_record).astype(int),
            'v_record': np.asarray(v_record).astype(np.float32),
            'w_record': np.asarray(w_record).astype(np.float32),
            'paths': paths,
            'use_estimation': use_estimation,
            }


def mainapp():
    matplotlib.rc('font', **FONT)
    render = Spatializer.Render()
    render.compression_filter = Spatializer.Compressing.Downsample512()
    pose_estimators = {'naive': select_pose_estimator('Naive'),
                       'onset': select_pose_estimator('Onset'),
                       'twoshot': select_pose_estimator('TwoShot')}
    for estimator in pose_estimators.values(): estimator.presence_detector.detection_threshold = PRESENCE_DETECTION_THRESHOLD
    path_planner = DubinsDockZonePathPlanner()
    kinematic_converter = DubinsToKinematics()
    estimator_type = 'twoshot'
    compressed_distances = render.compression_filter(Setting.DISTANCE_ENCODING)
    bat_pose = np.asarray([0., 0., np.random.uniform(-np.pi, np.pi)])
    bat_pose[:2] = np.random.uniform(-JITTER, JITTER, size=2)
    init_flowers_pose = np.asarray([
        [-1., -1., 0., 3.],
        [1., -1., 0., 3.],
        [1., 1., 0., 3.],
        [-1., 1., 0., 3.],
    ])
    init_flowers_pose[:,2] = np.random.uniform(-np.pi, np.pi, size=len(init_flowers_pose))
    init_flowers_pose[:,:2] *= FLOWER_XY_ANCHOR
    init_flowers_pose[:, :2] += np.random.uniform(-JITTER, JITTER, size=(len(init_flowers_pose), 2))

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
    bat_trajectory_arrow = ax1.arrow(bat_pose[0], bat_pose[1],
            0.5*BAT_ARROW_LENGTH*np.cos(bat_pose[2]), 0.5*BAT_ARROW_LENGTH*np.sin(bat_pose[2]),
            width=0.02, head_width=0.05, head_length=0.02, fc='g', ec='g', alpha=0.8)
    bat_trajectory_arrow.set_visible(False)
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
    dockzone = get_dockzone_notched_circle_from_flower_pose(est_flower_pose)
    dockzone_circle, = ax1.plot(*generate_dockzone_notched_circle_waypoints(dockzone), 'k:', linewidth=1.0, alpha=0.5)
    path = path_planner(bat_pose, dockzone)
    path_line, = ax1.plot(*path.path_waypoints, 'b--', linewidth=1.0, alpha=0.5)
    for i in range(len(path.centers)):
        if i==1: continue
        centers_x = [path.centers[i][0]]
        centers_y = [path.centers[i][1]]
    tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
    tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
    path_keypoints, = ax1.plot(centers_x + tangent_points_x, centers_y + tangent_points_y, 'kx', markersize=5, alpha=0.5)
    bat_trajectory_line, = ax1.plot([], [], 'c-', linewidth=1.5, alpha=0.5)

    ax1.set_xlim([-2*FLOWER_XY_ANCHOR, 2*FLOWER_XY_ANCHOR])
    ax1.set_ylim([-2*FLOWER_XY_ANCHOR, 2*FLOWER_XY_ANCHOR])
    
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
    eps_progress_slider_ax = fig.add_axes([0.2, 0.1, 0.3, 0.01])
    flowers_checkbox_ax = fig.add_axes([ 0.92, 0.05, 0.05, 0.5 ])
    estimator_type_radio_ax = fig.add_axes([ 0.92, 0.6, 0.05, 0.2 ])
    flowers_slider_ax = []
    for i in range(len(init_flowers_pose)):
        flowers_slider_ax.append(fig.add_axes([0.02, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.06, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.10, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))

    reset_button_ax = fig.add_axes([0.92, 0.85, 0.05, 0.05])
    run_episode_button_ax = fig.add_axes([0.92, 0.9, 0.05, 0.05])

    bat_slider = []
    bat_slider.append(widgets.Slider(bat_slider_ax[0], 'x_bat', -1., 1., valstep=0.01, valinit=bat_pose[0]))
    bat_slider.append(widgets.Slider(bat_slider_ax[1], 'y_bat', -1., 1., valstep=0.01, valinit=bat_pose[1]))
    bat_slider.append(widgets.Slider(bat_slider_ax[2], '\u03B8_bat', -180, 180, valstep=0.1, valinit=np.degrees(bat_pose[2])))
    eps_progress_slider = widgets.Slider(eps_progress_slider_ax, 'Episode progress', 0, 10, valstep=1, valinit=0)
    
    flowers_slider = []
    for i in range(len(init_flowers_pose)):
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3], f'\u0394x{i}',
                                             -1, 1, valstep=0.01, valinit=init_flowers_pose[i,0] - np.sign(init_flowers_pose[i,0])*FLOWER_XY_ANCHOR, orientation='vertical') )
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3+1], f'\u0394y{i}',
                                             -1, 1, valstep=0.01, valinit=init_flowers_pose[i,1] - np.sign(init_flowers_pose[i,1])*FLOWER_XY_ANCHOR, orientation='vertical') )
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3+2], f'\u0394\u03B8{i}',
                                             -180, 180, valstep=0.1, valinit=np.degrees(init_flowers_pose[i,2]), orientation='vertical') )
    flowers_checkbox_txt = []
    for i in range(len(init_flowers_pose)): flowers_checkbox_txt.append(f'F{i}')
    flowers_checkbox = widgets.CheckButtons(flowers_checkbox_ax, flowers_checkbox_txt, [True]*len(init_flowers_pose))
    estimator_type_radio = widgets.RadioButtons(estimator_type_radio_ax, ['naive', 'onset', 'twoshot'], active=0)
    reset_button = widgets.Button(reset_button_ax, 'Reset')
    run_episode_button = widgets.Button(run_episode_button_ax, 'Run Episode')
    

    def update(val):
        if os.path.exists('result.pkl'):
            os.remove('result.pkl')
        flower0_est_arrow.set_visible(True)
        dockzone_circle.set_visible(True)
        path_line.set_visible(True)
        path_keypoints.set_visible(True)
        bat_trajectory_arrow.set_visible(False)
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

        dockzone = get_dockzone_notched_circle_from_flower_pose(est_flower_pose)
        path = path_planner(bat_pose, dockzone)
        if path.cost<np.inf:
            path_line.set_visible(True)
            path_keypoints.set_visible(True)
            path_line.set_data(*path.path_waypoints)
            for i in range(len(path.centers)):
                if path.modes[i]=='S': continue
                centers_x = [path.centers[i][0]]
                centers_y = [path.centers[i][1]]
            tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
            tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
            path_keypoints.set_xdata(centers_x + tangent_points_x)
            path_keypoints.set_ydata(centers_y + tangent_points_y)
        else:
            path_line.set_visible(False)
            path_keypoints.set_visible(False)

        flower0_est_arrow.set_data(x=est_flower_pose[0], y=est_flower_pose[1],
                dx=FLOWER_ARROW_LENGTH*np.cos(est_flower_pose[2]),
                dy=FLOWER_ARROW_LENGTH*np.sin(est_flower_pose[2]))
        envelope_left_.set_ydata(render.compress_left)
        envelope_right_.set_ydata(render.compress_right)
        inputs_left_.set_ydata(pose_estimators[estimator_type].cache['inputs'][0,0,:])
        inputs_right_.set_ydata(pose_estimators[estimator_type].cache['inputs'][0,1,:])
        
        dockzone_circle.set_data(*generate_dockzone_notched_circle_waypoints(dockzone))
        bat_trajectory_line.set_data([],[])
        fig.canvas.draw_idle()

    def reset(event):
        if os.path.exists('result.pkl'):
            os.remove('result.pkl')
        flower0_est_arrow.set_visible(True)
        dockzone_circle.set_visible(True)
        path_line.set_visible(True)
        path_keypoints.set_visible(True)
        bat_trajectory_arrow.set_visible(False)
        sys.stdout.writelines('Resetting...                                          \r')
        # reset bat pose and init_flowers_pose and cartseian_objects_matrix
        bat_pose = np.asarray([0., 0., np.random.uniform(-np.pi, np.pi)])
        bat_pose[:2] += np.random.uniform(-JITTER, JITTER, size=2)
        init_flowers_pose = np.asarray([
            [-1., -1., 0., 3.],
            [1., -1., 0., 3.],
            [1., 1., 0., 3.],
            [-1., 1., 0., 3.],
        ])
        init_flowers_pose[:,2] = np.random.uniform(-np.pi, np.pi, size=len(init_flowers_pose))
        init_flowers_pose[:,:2] *= FLOWER_XY_ANCHOR
        init_flowers_pose[:, :2] += np.random.uniform(-JITTER, JITTER, size=(len(init_flowers_pose), 2))
        # reset all slider and reset change the init_values on the slider.
        for i in range(len(bat_pose)):
            if i!=2: bat_slider[i].set_val(bat_pose[i])
            else: bat_slider[i].set_val(np.degrees(bat_pose[i]))
        for i in range(len(init_flowers_pose)):
            for w in range(3):
                if w!=2: flowers_slider[i*3 + w].set_val(init_flowers_pose[i,w] - np.sign(init_flowers_pose[i,w])*FLOWER_XY_ANCHOR)
                else: flowers_slider[i*3 + w].set_val(np.degrees(init_flowers_pose[i,w]))
        update(0)

    def run_eps_event(event):
        if os.path.exists('result.pkl'):
            os.remove('result.pkl')
        sys.stdout.writelines('Running episode...                                          \r')
        flower0_est_arrow.set_visible(False)
        dockzone_circle.set_visible(False)
        path_line.set_visible(False)
        path_keypoints.set_visible(False)
        bat_trajectory_arrow.set_visible(False)
        flower_status = flowers_checkbox.get_status()
        estimator_type = estimator_type_radio.value_selected
        bat_pose = np.asarray([
            bat_slider[0].val, bat_slider[1].val, np.radians(bat_slider[2].val)
        ]).astype(np.float32)
        for i in range(len(cartesian_objects_matrix)):
            cartesian_objects_matrix[i,0] = init_flowers_pose[i][0] + flowers_slider[i*3].val if flower_status[i] else -100.
            cartesian_objects_matrix[i,1] = init_flowers_pose[i][1] + flowers_slider[i*3+1].val if flower_status[i] else -0.
            cartesian_objects_matrix[i,2] = np.radians(flowers_slider[i*3+2].val) if flower_status[i] else 0.
        
        dockzone_circle.set_data(*generate_dockzone_notched_circle_waypoints(dockzone))
        #v_seq, w_seq = kinematic_converter(path)
        trajectories_dict =  run_episode(bat_pose=bat_pose, objects=cartesian_objects_matrix,
                                         renderer=render, pose_estimator=pose_estimators[estimator_type],
                                         path_planner=path_planner, kinematic_converter=kinematic_converter,)
        with open('result.pkl', 'wb') as f:
            pickle.dump(trajectories_dict, f)
        if len(trajectories_dict['marker_record']) > 1:
            eps_progress_slider.valmax = len(trajectories_dict['marker_record']) - 1
            eps_progress_slider.valmin = 0
        else:
            eps_progress_slider.valmax = 0
            eps_progress_slider.valmin = -1
        eps_progress_slider.ax.set_xlim([eps_progress_slider.valmin, eps_progress_slider.valmax])
        eps_progress_slider.reset()
        bat_trajectory_line.set_data(trajectories_dict['bat_trajectory'][:,0], trajectories_dict['bat_trajectory'][:,1])
        bat_trajectory_line.set_color('c')
        bat_trajectory_line.set_linewidth(1.5)
        fig.canvas.draw_idle()

    def progress_slider_update(val):
        if os.path.exists('result.pkl'):
            with open('result.pkl', 'rb') as f:
                trajectories_dict = pickle.load(f)
        else: raise FileNotFoundError('result.pkl not found. Need to run episode first.')
        flower0_est_arrow.set_visible(True)
        dockzone_circle.set_visible(True)
        path_line.set_visible(True) 
        path_keypoints.set_visible(True)
        bat_trajectory_arrow.set_visible(True)
        bat_trajectory_line.set_color('m')
        bat_trajectory_line.set_linewidth(3.)
        progress_step = int(eps_progress_slider.val)
        marker = trajectories_dict['marker_record'][progress_step]
        envelope_left_.set_ydata(trajectories_dict['envelope_left_record'][progress_step])
        envelope_right_.set_ydata(trajectories_dict['envelope_right_record'][progress_step])
        inputs_left_.set_ydata(trajectories_dict['inputs_left_record'][progress_step])
        inputs_right_.set_ydata(trajectories_dict['inputs_right_record'][progress_step])
        if progress_step <= 0: bat_pose = trajectories_dict['bat_trajectory'][0]
        else: bat_pose = trajectories_dict['bat_trajectory'][trajectories_dict['marker_record'][progress_step-1]]
        bat_trajectory_arrow.set_data(x=bat_pose[0], y=bat_pose[1],
                dx=0.5*BAT_ARROW_LENGTH*np.cos(bat_pose[2]),
                dy=0.5*BAT_ARROW_LENGTH*np.sin(bat_pose[2]))
        est_flower_pose = trajectories_dict['flower_pose_record'][progress_step]
        estimated_distance = trajectories_dict['predicted_distances'][progress_step]
        #print(est_flower_pose)
        if np.isnan(est_flower_pose[0]):
            dockzone_circle.set_visible(False)
            flower0_est_arrow.set_visible(False)
            path_line.set_visible(False)
            path_keypoints.set_visible(False)
            #print('There should not be any est flower arrow and dockzone circle nor path!!!!')
        else:
            #print('predicted distance: {:.2f}'.format(estimated_distance))
            if np.isnan(estimated_distance):
                flower0_est_arrow.set_visible(False)
                dockzone_circle.set_visible(False)
                #print('There should not be any est flower arrow and dockzone circle!!!!')
            else:
                flower0_est_arrow.set_visible(True)
                dockzone_circle.set_visible(True)
                dockzone = get_dockzone_notched_circle_from_flower_pose(est_flower_pose)
                #path = path_planner(bat_pose, dockzone)
                #path = trajectories_dict['paths'][progress_step]
                if progress_step <= 0: ref_pose = trajectories_dict['bat_trajectory'][0]
                else:
                    k = 0
                    while not trajectories_dict['marker_record'][progress_step - k]: k += 1
                    ref_pose =  trajectories_dict['bat_trajectory'][trajectories_dict['marker_record'][progress_step - k -1]]
                #if hasattr(path, 'path_waypoints'): path_waypoints = transform_coordinates(*path.path_waypoints, *ref_pose)
                #else: path_waypoints = ([],[])
                path = path_planner(ref_pose, dockzone)
                if path.cost<np.inf:
                    path_line.set_visible(True)
                    path_keypoints.set_visible(True)
                    path_line.set_data(*path.path_waypoints)
                    for i in range(len(path.centers)):
                        if path.modes[i]=='S': continue
                        centers_x = [path.centers[i][0]]
                        centers_y = [path.centers[i][1]]
                    tangent_points_x = [path.tangent_points[i][0] for i in range(len(path.tangent_points))]
                    tangent_points_y = [path.tangent_points[i][1] for i in range(len(path.tangent_points))]
                    path_keypoints.set_xdata(centers_x + tangent_points_x)
                    path_keypoints.set_ydata(centers_y + tangent_points_y)
                else:
                    path_line.set_visible(False)
                    path_keypoints.set_visible(False)
                flower0_est_arrow.set_data(x=est_flower_pose[0], y=est_flower_pose[1],
                        dx=FLOWER_ARROW_LENGTH*np.cos(est_flower_pose[2]),
                        dy=FLOWER_ARROW_LENGTH*np.sin(est_flower_pose[2]))
                dockzone_circle.set_data(*generate_dockzone_notched_circle_waypoints(dockzone))
        bat_trajectory_line.set_data(trajectories_dict['bat_trajectory'][:marker,0], trajectories_dict['bat_trajectory'][:marker,1])
        print('predicted distance: {:.2f}'.format(estimated_distance))
        fig.canvas.draw_idle()


    flowers_checkbox.on_clicked(update)
    estimator_type_radio.on_clicked(update)
    for slider in bat_slider+flowers_slider: slider.on_changed(update)
    reset_button.on_clicked(reset)
    run_episode_button.on_clicked(run_eps_event)
    eps_progress_slider.on_changed(progress_slider_update)
    # I love robots.
    plt.show()
    if os.path.exists('result.pkl'):
        os.remove('result.pkl')

    
def main():
    return mainapp()

if __name__ == '__main__':
    main()
