import sys
import os
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple, Dict, Callable, Union, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting
from Simulation.Motion import State

from Perception.flower_pose_estimator import NaiveOneShotFlowerPoseEstimator, OnsetOneShotFlowerPoseEstimator, TwoShotFlowerPoseEstimator

PRESENCE_DETECTION_THRESHOLD = 2.
JITTER = 0.2
FLOWER_XY_ANCHOR = 1.5

FLOWER_ARROW_LENGTH = 0.2
BAT_ARROW_LENGTH = 0.3

FONT = {'size': 10}

from TrajectoryHandler.dubinspath import DubinsDockZonePathPlanner, DubinsToKinematics, DubinsToKinematicsNoAccel
from TrajectoryHandler.dockzone import generate_dockzone_notched_circle_waypoints, get_dockzone_notched_circle_from_flower_pose
from TrajectoryHandler.settings import BatKinematicParams, OtherKinematicParams


def convert_polar_to_cartesian(bat_pose: ArrayLike,
                               flower_distance: float, flower_azimuth: float, flower_orientation: float) -> ArrayLike:
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
    
# run kinematic sequence (v,w array) and return (trajactories) --> enough for now.
def run_kinematic_sequence(bat_pose: ArrayLike, v_sequence: ArrayLike, w_sequence: ArrayLike, **kwargs) -> ArrayLike:
    state = State(pose = bat_pose, kinematic= [v_sequence[0], w_sequence[0]], dt=1/BatKinematicParams.CHIRP_RATE,
                  max_linear_velocity=BatKinematicParams.MAX_LINEAR_VELOCITY,
                  max_angular_velocity=BatKinematicParams.MAX_ANGULAR_VELOCITY*1,
                  max_linear_acceleration=OtherKinematicParams.LINEAR_ACCEL_LIMIT,
                  max_angular_acceleration=BatKinematicParams.MAX_ANGULAR_ACCELERATION*1,
                  max_linear_deceleration=OtherKinematicParams.LINEAR_DECEL_LIMIT,
                  )
    #render = Spatializer.Render()
    bat_trajectory = np.asarray([]).astype(np.float32).reshape(0, 3)
    v_records = []
    w_records = []
    for v, w in zip(v_sequence, w_sequence):
        bat_trajectory = np.vstack((bat_trajectory, state.pose))
        state(v=v, w=w)
        v_records.append(state.kinematic[0])
        w_records.append(state.kinematic[1])
    bat_trajectory = np.vstack((bat_trajectory, state.pose))
    #print('report: v_seq: {}, w_seq: {}, traj: {}, fin_v: {}, fin_kin: {}'.format(len(v_sequence), len(w_sequence), len(bat_trajectory), v_sequence[-1], state.kinematic[0]))
    return bat_trajectory, np.asarray(v_records).astype(float), np.asarray(w_records).astype(float)

def utest1():
    matplotlib.rc('font', **FONT)
    render = Spatializer.Render()
    render.compression_filter = Spatializer.Compressing.Downsample512()
    pose_estimators = {'naive': select_pose_estimator('Naive'), 'onset': select_pose_estimator('Onset'), 'twoshot': select_pose_estimator('TwoShot')}
    for estimator in pose_estimators.values(): estimator.presence_detector.detection_threshold = PRESENCE_DETECTION_THRESHOLD
    path_planer = DubinsDockZonePathPlanner()
    kinematic_converter = DubinsToKinematics()
    estimator_type = 'naive'
    compressed_distances = render.compression_filter(Setting.DISTANCE_ENCODING)
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
    dockzone = get_dockzone_notched_circle_from_flower_pose(est_flower_pose)
    dockzone_circle, = ax1.plot(*generate_dockzone_notched_circle_waypoints(dockzone), 'k:', linewidth=1.0, alpha=0.5)
    path = path_planer(bat_pose, dockzone)
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
    flowers_checkbox_ax = fig.add_axes([ 0.92, 0.05, 0.05, 0.5 ])
    estimator_type_radio_ax = fig.add_axes([ 0.92, 0.6, 0.05, 0.2 ])
    flowers_slider_ax = []
    for i in range(len(init_flowers_pose)):
        flowers_slider_ax.append(fig.add_axes([0.02, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.06, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.10, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))

    reset_button_ax = fig.add_axes([0.92, 0.85, 0.05, 0.05])
    run_path_button_ax = fig.add_axes([0.92, 0.9, 0.05, 0.05])

    bat_slider = []
    bat_slider.append(widgets.Slider(bat_slider_ax[0], 'x_bat', -1., 1., valstep=0.01, valinit=bat_pose[0]))
    bat_slider.append(widgets.Slider(bat_slider_ax[1], 'y_bat', -1., 1., valstep=0.01, valinit=bat_pose[1]))
    bat_slider.append(widgets.Slider(bat_slider_ax[2], '\u03B8_bat', -180, 180, valstep=0.1, valinit=np.degrees(bat_pose[2])))
    
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
    run_path_button = widgets.Button(run_path_button_ax, 'Run Path')

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

        dockzone = get_dockzone_notched_circle_from_flower_pose(est_flower_pose)
        path = path_planer(bat_pose, dockzone)
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

    def run_path(event):
        sys.stdout.writelines('Running path...                                          \r')
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
        path = path_planer(bat_pose, dockzone)
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
        v_seq, w_seq = kinematic_converter(path)
        trajectories, _,_ =  run_kinematic_sequence(bat_pose, v_seq, w_seq)
        bat_trajectory_line.set_data(trajectories[:,0], trajectories[:,1])
        if path.cost == np.inf: bat_trajectory_line.set_color('m')
        else: bat_trajectory_line.set_color('c')
                
        fig.canvas.draw_idle()


    flowers_checkbox.on_clicked(update)
    estimator_type_radio.on_clicked(update)
    for slider in bat_slider+flowers_slider: slider.on_changed(update)
    reset_button.on_clicked(reset)
    run_path_button.on_clicked(run_path)
    plt.show()

def dummy_test_1():
    from TrajectoryHandler.dubinspath import DubinsParams
    path = DubinsParams(modes = ['L','S','R','S','L','L','L'],
                        radii=[0.1, np.inf, 0.1, np.inf, 0.1, 0.6, 0.3],
                        quantities=[0.5*np.pi, 0.1, -0.5*np.pi, 0.2, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi],
                        cost = 1.)
    #path = DubinsParams(modes = ['S'], radii=[np.inf], quantities=[.0001,], cost = 1.)
    #path = DubinsParams(modes = ['L'], radii=[1.], quantities=[1.5*np.pi], cost = 1.)
    kinematic_converter1 = DubinsToKinematicsNoAccel()
    kinematic_converter2 = DubinsToKinematics()
    v_seq1, w_seq1, = kinematic_converter1(path)
    v_seq2, w_seq2 = kinematic_converter2(path)
    update_rate = BatKinematicParams.CHIRP_RATE
    bat_pose = np.asarray([0., 0., np.radians(45)])
    trajectories1, v_rec1, w_rec1 = run_kinematic_sequence(bat_pose, v_seq1, w_seq1)
    trajectories2, v_rec2, w_rec2 = run_kinematic_sequence(bat_pose, v_seq2, w_seq2)
    #print('error {}'.format(np.linalg.norm(trajectories1[:2]-trajectories2[:2])))
    print('travel 1: {}, travel 2: {}'.format(np.sum(v_seq1/update_rate), np.sum(v_seq2/update_rate)))
    print('actual travel 1: {}, actual travel 2: {}'.format(np.linalg.norm(trajectories1[-1] - trajectories1[0]),
                                                            np.linalg.norm(trajectories2[-1] - trajectories2[0])))
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(trajectories1[:,0], trajectories1[:,1], linewidth=5., alpha=0.6, label='Traj1')
    ax[0].plot(trajectories2[:,0], trajectories2[:,1], alpha=0.6, label='Traj2')
    ax[0].legend()
    ax[0].set_aspect('equal', 'box')
    ax[0].grid()
    ax[1].plot(v_seq1, 'o-', label='v1')
    ax[1].plot(v_seq2, 'o-', label='v2')
    ax[1].plot(v_rec1, 'x:', label='v1_rec')
    ax[1].plot(v_rec2, 'x:', label='v2_rec')
    ax[1].legend()
    ax[1].grid()
    ax[2].plot(w_seq1, 'o-', label='w1')
    ax[2].plot(w_seq2, 'o-', label='w2')
    ax[2].plot(w_rec1, 'x:', label='w1_rec')
    ax[2].plot(w_rec2, 'x:', label='w2_rec')
    ax[2].grid()
    plt.show()


def main():
    return utest1()
    #return dummy_test_1()

if __name__=='__main__':
    main()