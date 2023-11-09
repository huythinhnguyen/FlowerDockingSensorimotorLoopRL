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
        self.mode: List[str] = kwargs.get('mode', [])
        self.quantities: List[float] = kwargs.get('quantities', [])
        self.cost: float = kwargs.get('cost', 0.)
        self.radii: List[float] = kwargs.get('radii', [])

class DubinsPathPlanner:
    def __init__(self,
                 min_turn_radius: float=MIN_TURNING_RADIUS,):
        self.min_turn_radius = min_turn_radius

    # solve for each mode then take the best one.

    def _solve_LSL(self, start_pose: ArrayLike, end_pose: ArrayLike, min_turn_radius: float=MIN_TURNING_RADIUS) -> DubinsParams:
        mode = ['L', 'S', 'L']
        radii = [min_turn_radius, np.inf, min_turn_radius]
        # FOR CSC.
        # find center of rotation for each curves.
        # find straight tagent line on both curves
        # find the intersection between tagent line and circles.
        # find the amount of angle for first rotation and last rotation.
        # fint he distance between 2 tagent points.

    def _solve_LRL(self, start_pose: ArrayLike, end_pose: ArrayLike, min_turn_radius: float=MIN_TURNING_RADIUS) -> DubinsParams:
        mode = ['L', 'R', 'L']
        radii = [min_turn_radius, min_turn_radius, min_turn_radius]
        # FOR CCC.
        # find center of rotation for each curves.
        # if the distance between 2 centers > 4*min_turn_radius, then there is no solution.
        # find the center of the tagent curve.
        # find the intersection between tagent curve and circles.
        # find the amount of angle for first rotation and last rotation.
        # fint he distance between 2 tagent points.


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