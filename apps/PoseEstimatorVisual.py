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
    
def mainapp(init_estimator_type: str):
    matplotlib.rc('font', **FONT)
    render = Spatializer.Render()
    render.compression_filter = Spatializer.Compressing.Downsample512()
    pose_estimators = {'naive': select_pose_estimator('Naive'), 'onset': select_pose_estimator('Onset'), 'twoshot': select_pose_estimator('TwoShot')}
    for estimator in pose_estimators.values(): estimator.presence_detector.detection_threshold = 4.
    estimator_type = init_estimator_type
    compressed_distances = render.compression_filter(Setting.DISTANCE_ENCODING)
    bat_pose = np.asarray([0., 0., 0.])
    init_flowers_pose = [
        [1., 1., 0., 3.],
        [1., -1., 0., 3.],
        [2., 1., 0., 3.],
        [2., -1., 0., 3.],
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
    
    
    ax1.set_xlim([-1,4])
    ax1.set_ylim([-2.5,2.5])
    
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
    
    bat_ori_slider_ax = fig.add_axes([0.08, 0.95, 0.3, 0.01])
    flowers_checkbox_ax = fig.add_axes([ 0.92, 0.05, 0.05, 0.5 ])
    estimator_type_radio_ax = fig.add_axes([ 0.92, 0.6, 0.05, 0.2 ])
    flowers_slider_ax = []
    for i in range(len(init_flowers_pose)):
        flowers_slider_ax.append(fig.add_axes([0.02, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.06, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))
        flowers_slider_ax.append(fig.add_axes([0.10, 0.05 + i*(0.6/len(init_flowers_pose)+0.08), 0.01, 0.6/len(init_flowers_pose)]))

    bat_ori_slider = widgets.Slider(bat_ori_slider_ax, '\u03B8_bat', -180, 180, valstep=0.1, valinit=np.degrees(bat_pose[2]))
    flowers_slider = []
    for i in range(len(init_flowers_pose)):
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3], f'x{i}', -1, 3, valstep=0.01, valinit=init_flowers_pose[i][0], orientation='vertical') )
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3+1], f'y{i}', -2.5, 2.5, valstep=0.01, valinit=init_flowers_pose[i][1], orientation='vertical') )
        flowers_slider.append(widgets.Slider(flowers_slider_ax[i*3+2], f'\u03B8{i}', -180, 180, valstep=0.1, valinit=np.degrees(init_flowers_pose[i][2]), orientation='vertical') )
    flowers_checkbox_txt = []
    for i in range(len(init_flowers_pose)): flowers_checkbox_txt.append(f'F{i}')
    flowers_checkbox = widgets.CheckButtons(flowers_checkbox_ax, flowers_checkbox_txt, [True]*len(init_flowers_pose))
    estimator_type_radio = widgets.RadioButtons(estimator_type_radio_ax, ['naive', 'onset', 'twoshot'], active=0)



    def update(val):
        flower_status = flowers_checkbox.get_status()
        estimator_type = estimator_type_radio.value_selected
        bat_pose = np.zeros(3).astype(float)
        bat_pose[2] = np.radians(bat_ori_slider.val)
        for i in range(len(cartesian_objects_matrix)):
            cartesian_objects_matrix[i,0] = flowers_slider[i*3].val if flower_status[i] else -100.
            cartesian_objects_matrix[i,1] = flowers_slider[i*3+1].val if flower_status[i] else -0.
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
        
        
        fig.canvas.draw_idle()
    bat_ori_slider.on_changed(update)
    flowers_checkbox.on_clicked(update)
    estimator_type_radio.on_clicked(update)
    for slider in flowers_slider: slider.on_changed(update)

    plt.show()

def main():
    return mainapp(sys.argv[1] if len(sys.argv)>1 else 'naive')

if __name__ == '__main__':
    main()