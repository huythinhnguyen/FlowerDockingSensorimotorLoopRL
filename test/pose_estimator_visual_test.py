"""
Description:
- This meta test will produce a matplotlib widget visualizing top-down view.
- Bat pose will be fixed at (0,0,0) for convinience.
- Flower pose will be able to be changed by the user.
- A render of estimated flower power will be showed.
Basic features:
    o Add 1 flower to the widgets.
    o Hardcoded the estimator to use. (Use NaiveEstimator for now.)
    o Make sure this is free of bug before moving on.
Wish-list features:
    + Add 2 or more flower to the widgets.
    + Allow user to select which estimator to use.
    + Maybe allow user to move bat pose around also.
"""
import sys
import os
import time
import numpy as np
from numpy.typing import ArrayLike
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

from Perception.flower_pose_estimator import NaiveOneShotFlowerPoseEstimator, OnsetOneShotFlowerPoseEstimator, TwoShotFlowerPoseEstimator

FLOWER_ARROW_LENGTH = 0.2
BAT_ARROW_LENGTH = 0.3

FONT = {'size': 10}


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
    
########################################
# Write a function to convert prediction to cartesian coordinates (x,y,theta)

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
    # THIS IS NOT CORRECT. FIX IT.
    flower_pose[2] = Spatializer.wrapToPi(bat_pose[2] + flower_azimuth + np.pi - flower_orientation)

    return flower_pose

    

def utest_render_1(estimator_type='Naive'):
    matplotlib.rc('font', **FONT)
    render = Spatializer.Render()
    render.compression_filter = Spatializer.Compressing.Downsample512()
    pose_estimator = select_pose_estimator(estimator_type)
    pose_estimator.presence_detector.detection_threshold = 1.
    compressed_distances = render.compression_filter(Setting.DISTANCE_ENCODING)
    bat_pose = np.asarray([0., 0., 0.])
    init_flower_pose = [1., 0., 0., 3.]
    cartesian_objects_matrix = np.asarray([init_flower_pose,]).astype(np.float32)
    render.run(bat_pose, cartesian_objects_matrix)
    inputs = np.concatenate([render.compress_left, render.compress_right]).reshape(1,2,-1)
    prediction = pose_estimator(inputs)
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
    
    flower0_arrow = ax1.arrow(cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1],
            FLOWER_ARROW_LENGTH*np.cos(cartesian_objects_matrix[0,2]), FLOWER_ARROW_LENGTH*np.sin(cartesian_objects_matrix[0,2]),
            width=0.05, head_width=0.1, head_length=0.05, fc='m', ec='m')
    
    
    ax1.set_xlim([-1,4])
    ax1.set_ylim([-2.5,2.5])
    
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend([bat_arrow, flower0_arrow, flower0_est_arrow], ['Bat', 'Flower (GT)', 'Flower (Est)'],
               loc = 'upper left', bbox_to_anchor=(0., 1.25))

    envelope_left_, envelope_right_, ax2 = set_up_waveform_plot(ax2,
            compressed_distances, render.compress_left, render.compress_right, title='Echo envelope of the scene')
    inputs_left_, inputs_right_, ax3 = set_up_waveform_plot(ax3,
            compressed_distances,
            pose_estimator.cache['inputs'][0,0,:],
            pose_estimator.cache['inputs'][0,1,:], title='Inputs to the estimator')
    
    ax2.plot(compressed_distances, pose_estimator.presence_detector.profile*5, 'k--', linewidth=0.1, alpha=0.2)
    ax3.plot(compressed_distances, pose_estimator.presence_detector.profile, 'k--', linewidth=0.1, alpha=0.2)

    flower_slider_ax = []
    flower_slider_ax.append(fig.add_axes([0.02, 0.1, 0.01, 0.8]))
    flower_slider_ax.append(fig.add_axes([0.06, 0.1, 0.01, 0.8]))
    flower_slider_ax.append(fig.add_axes([0.1, 0.1, 0.01, 0.8]))
    fig.subplots_adjust(left=.15,)

    flower_slider = []
    flower_slider.append(widgets.Slider(flower_slider_ax[0], 'x', -1, 3, valstep=0.01, valinit=init_flower_pose[0], orientation='vertical') )
    flower_slider.append(widgets.Slider(flower_slider_ax[1], 'y', -2.5, 2.5, valstep=0.01, valinit=init_flower_pose[1], orientation='vertical') )
    flower_slider.append(widgets.Slider(flower_slider_ax[2], '\u03B8', -180, 180, valstep=0.1, valinit=np.degrees(init_flower_pose[2]), orientation='vertical') )

    def update(val):
        bat_pose = np.zeros(3).astype(float)
        cartesian_objects_matrix[0,0] = flower_slider[0].val
        cartesian_objects_matrix[0,1] = flower_slider[1].val
        cartesian_objects_matrix[0,2] = np.radians(flower_slider[2].val)
        render.run(bat_pose, cartesian_objects_matrix)
        inputs = np.concatenate([render.compress_left, render.compress_right]).reshape(1,2,-1)
        prediction = pose_estimator(inputs)
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
        flower0_arrow.set_data(x=cartesian_objects_matrix[0,0], y=cartesian_objects_matrix[0,1],
                dx=FLOWER_ARROW_LENGTH*np.cos(cartesian_objects_matrix[0,2]),
                dy=FLOWER_ARROW_LENGTH*np.sin(cartesian_objects_matrix[0,2]))
        flower0_est_arrow.set_data(x=est_flower_pose[0], y=est_flower_pose[1],
                dx=FLOWER_ARROW_LENGTH*np.cos(est_flower_pose[2]),
                dy=FLOWER_ARROW_LENGTH*np.sin(est_flower_pose[2]))
        envelope_left_.set_ydata(render.compress_left)
        envelope_right_.set_ydata(render.compress_right)
        inputs_left_.set_ydata(pose_estimator.cache['inputs'][0,0,:])
        inputs_right_.set_ydata(pose_estimator.cache['inputs'][0,1,:])

        fig.canvas.draw_idle()
    for slider in flower_slider: slider.on_changed(update)

    plt.show()


def main():
    return utest_render_1()

if __name__ == '__main__':
    main()