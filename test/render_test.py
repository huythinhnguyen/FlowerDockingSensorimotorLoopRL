import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting



def utest_render_1():
    render = Spatializer.Render()
    bat_pose = np.asarray([-0., 0., np.radians(45)])
    cartesian_objects_matrix = np.asarray([[-0.5, 0.25, np.radians(90), 3.],
                                           [0.25, 0.25, np.radians(-135), 3.],
                                           [0.25, -0.5, np.radians(135), 3.],
                                           ])
    render.run(bat_pose, cartesian_objects_matrix)

    fig, ax = plt.subplots(2,1, dpi=200, sharex=True)
    ax[0].plot(Setting.DISTANCE_ENCODING, render.waveform_left, linewidth=0.5, alpha=0.5)
    ax[0].plot(Setting.DISTANCE_ENCODING, render.waveform_right, linewidth=0.5, alpha=0.5)
    ax[1].plot(Setting.DISTANCE_ENCODING, render.envelope_left, linewidth=.5)
    ax[1].plot(Setting.DISTANCE_ENCODING, render.envelope_right, linewidth=.5)
    ax[1].stem(Setting.COMPRESSED_DISTANCE_ENCODING, render.compress_left,basefmt=':', markerfmt='b.', linefmt='-',)
    ax[1].stem(Setting.COMPRESSED_DISTANCE_ENCODING, render.compress_right,basefmt=':', markerfmt='r.', linefmt='-',)
    print(render.snippet_left.shape)
    print(render.snippet_right.shape)
    print(render.viewer.filtered_objects_inview_polar)
    plt.show()

def utest_render_2():
    render = Spatializer.Render()
    init_bat_orientation = np.radians(90)
    init_dist = 1.0
    bat_pose = np.asarray([0., 0., init_bat_orientation])
    init_flower_orientation = np.radians(-90)
    cartesian_objects_matrix = np.asarray([[0., init_dist, init_flower_orientation, 3.],
                                           ])
    render.run(bat_pose, cartesian_objects_matrix)
    flower_arrow_length = 0.2
    bat_arrow_length = 0.3

 

    fig = plt.figure(figsize=(12, 6), dpi=200)
    gs = GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    bat_arrow = ax1.arrow(bat_pose[0], bat_pose[1],
            bat_arrow_length*np.cos(bat_pose[2]), bat_arrow_length*np.sin(bat_pose[2]),
            width=0.05, head_width=0.1, head_length=0.05, fc='k', ec='k')
    flower_arrow = ax1.arrow(cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1],
            flower_arrow_length*np.cos(cartesian_objects_matrix[0,2]), flower_arrow_length*np.sin(cartesian_objects_matrix[0,2]),
            width=0.05, head_width=0.1, head_length=0.05, fc='m', ec='m')
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-1,3])
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    #set legend outside and on top of the plot in one row
    ax1.legend(['Bat', 'Flower'], loc='upper left', bbox_to_anchor=(0., 1.2))

    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title('Waveforms')
    ax2.set_xlabel('Distance (m)')
    #ax2.set_ylabel('Amplitude')
    waveform_left_, = ax2.plot(Setting.DISTANCE_ENCODING, render.waveform_left, linewidth=0.5, alpha=0.5)
    waveform_right_, = ax2.plot(Setting.DISTANCE_ENCODING, render.waveform_right, linewidth=0.5, alpha=0.5)
    ax2.set_ylim([-1500, 1500])

    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.set_title('Envelopes')
    ax3.set_xlabel('Distance (m)')
    #ax3.set_ylabel('Amplitude')
    envelope_left_, = ax3.plot(Setting.DISTANCE_ENCODING, render.envelope_left, linewidth=.5)
    envelope_right_, = ax3.plot(Setting.DISTANCE_ENCODING, render.envelope_right, linewidth=.5)

    axdistance = plt.axes([0.2, 0.05, 0.7, 0.01])
    axbatorient = plt.axes([0.2, 0.1, 0.7, 0.01])
    axflowerorient = plt.axes([0.2, 0.15, 0.7, 0.01])
    
    ax0 = fig.add_subplot(gs[2, 1:])
    ax0.set_title('Snippet')
    snippet_left_, = ax0.plot(render.snippet_left[0], linewidth=0.5, alpha=0.5)
    snippet_right_, = ax0.plot(render.snippet_right[0], linewidth=0.5, alpha=0.5)
    ax0.set_ylim([-1500, 1500])

    fig.subplots_adjust(bottom=0.25)

    sdistance = widgets.Slider(axdistance, 'Distance (m)', 0.1, 1.5, valinit=init_dist, valstep=0.002, orientation='horizontal')
    sbatorient = widgets.Slider(axbatorient, 'Bat Orient (\u00b0)', -180, 180, valinit=0., valstep=0.1, orientation='horizontal')
    sflowerorient = widgets.Slider(axflowerorient, 'Flower Orient (\u00b0)', -180, 180, valinit=0., valstep=0.1, orientation='horizontal')

    def update(val):
        bat_pose = np.zeros(3).astype(np.float32)
        bat_pose[2] = Spatializer.wrapToPi(np.radians(sbatorient.val) + init_bat_orientation)
        cartesian_objects_matrix[0,1] = sdistance.val
        cartesian_objects_matrix[0,2] = Spatializer.wrapToPi(np.radians(sflowerorient.val) + init_flower_orientation)
        render.run(bat_pose, cartesian_objects_matrix)
        if render.viewer.collision_status==True:
            bat_arrow.set_color('r')
            bat_pose = np.copy(render.viewer.bat_pose)
            #print(render.viewer.bat_pose)
            if len(render.viewer.filtered_objects_inview_polar) > 0:
                print('dist = {}, angle = {}, orient = {}'.format(render.viewer.filtered_objects_inview_polar[0,0],
                                                              np.degrees(render.viewer.filtered_objects_inview_polar[0,1]),
                                                              np.degrees(render.viewer.filtered_objects_inview_polar[0,2])))
        else:
            bat_arrow.set_color('k')
        bat_arrow.set_data(x=bat_pose[0], y=bat_pose[1],
            dx=bat_arrow_length*np.cos(bat_pose[2]), dy=bat_arrow_length*np.sin(bat_pose[2]))
        flower_arrow.set_data(x=cartesian_objects_matrix[0,0], y=cartesian_objects_matrix[0,1],
            dx=flower_arrow_length*np.cos(cartesian_objects_matrix[0,2]), dy=flower_arrow_length*np.sin(cartesian_objects_matrix[0,2]))
        waveform_left_.set_ydata(render.waveform_left)
        waveform_right_.set_ydata(render.waveform_right)
        envelope_left_.set_ydata(render.envelope_left)
        envelope_right_.set_ydata(render.envelope_right)
        if len(render.snippet_left) > 0:
            snippet_left_.set_ydata(render.snippet_left)
        else: snippet_left_.set_ydata(np.zeros(7000))
        if len(render.snippet_right) > 0:
            snippet_right_.set_ydata(render.snippet_right)
        else: snippet_right_.set_ydata(np.zeros(7000))
        if np.sum(np.isnan(render.compress_left)) > 0 or np.sum(np.isnan(render.compress_right)) > 0:
            sys.stdout.writelines('nan detected @ D={}, B={}, F={}         \r'.format(sdistance.val, sbatorient.val, sflowerorient.val))
        fig.canvas.draw_idle()

    sdistance.on_changed(update)
    sbatorient.on_changed(update)
    sflowerorient.on_changed(update)
    plt.show()



def main():
    #return utest_render_1()
    return utest_render_2()

if __name__ == '__main__':
    main()
