import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec
from matplotlib import image as mpimg
from scipy import ndimage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting

BAT_ASSET_PATH = os.path.join(REPO_PATH, 'assets', 'bat.png')
BAT_HIT_ASSET_PATH = os.path.join(REPO_PATH, 'assets', 'bat_hit.png')
FLOWER_ASSET_PATH = os.path.join(REPO_PATH, 'assets', 'flower.png')
FLOWER_EST_ASSET_PATH = os.path.join(REPO_PATH, 'assets', 'flower_est.png')

ASSET_SCALE = 0.09

TRANSLATED_LABELS = {'left': 'left', 'right': 'right', 'waveform': 'Waveforms', 'envelope': 'Envelopes',
                     'distance': 'Distance (m)',
}

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

    fig = plt.figure(figsize=(12, 6), dpi=200)
    gs = GridSpec(2, 5, figure=fig)
    ax1 = fig.add_subplot(gs[:, :2])

    
    ##### Add image here ##################
    flower_artists = []
    bat_artists = []
    flower_asset = mpimg.imread(FLOWER_ASSET_PATH)
    bat_asset = mpimg.imread(BAT_ASSET_PATH)
    bat_hit_asset = mpimg.imread(BAT_HIT_ASSET_PATH)

    rotated_flower_asset = ndimage.rotate(flower_asset, np.degrees(cartesian_objects_matrix[0,2]))
    flower_imgbox = OffsetImage(rotated_flower_asset, zoom=ASSET_SCALE)
    flower_annobox = AnnotationBbox(flower_imgbox, (cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1]), frameon = False)
    flower_artists.append(ax1.add_artist(flower_annobox))

    rotated_bat_asset = ndimage.rotate(bat_asset, np.degrees(bat_pose[2]))
    bat_imgbox = OffsetImage(rotated_bat_asset, zoom=ASSET_SCALE)
    bat_annobox = AnnotationBbox(bat_imgbox, (bat_pose[0], bat_pose[1]), frameon = False)
    bat_artists.append(ax1.add_artist(bat_annobox))

    ######################################

    ax1.set_xlim([-2,2])
    ax1.set_ylim([-1,3])
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    #set legend outside and on top of the plot in one row
    #ax1.legend(['Bat', 'Flower'], loc='upper left', bbox_to_anchor=(0., 1.2))

    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_title(TRANSLATED_LABELS['waveform'])
    ax2.set_xlabel(TRANSLATED_LABELS['distance'])
    #ax2.set_ylabel('Amplitude')
    waveform_left_, = ax2.plot(Setting.DISTANCE_ENCODING, render.waveform_left, linewidth=0.5, alpha=0.5, label=TRANSLATED_LABELS['left'])
    waveform_right_, = ax2.plot(Setting.DISTANCE_ENCODING, render.waveform_right, linewidth=0.5, alpha=0.5, label=TRANSLATED_LABELS['right'])
    ax2.set_ylim([-1500, 1500])
    ax2.legend()
    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.set_title(TRANSLATED_LABELS['envelope'])
    ax3.set_xlabel(TRANSLATED_LABELS['distance'])
    #ax3.set_ylabel('Amplitude')
    envelope_left_, = ax3.plot(Setting.DISTANCE_ENCODING, render.envelope_left, linewidth=1.5, alpha=0.5, label=TRANSLATED_LABELS['left'])
    envelope_right_, = ax3.plot(Setting.DISTANCE_ENCODING, render.envelope_right, linewidth=1.5, alpha=0.5, label=TRANSLATED_LABELS['right'])
    ax3.legend()
    axdistance = plt.axes([0.2, 0.05, 0.7, 0.01])
    axbatorient = plt.axes([0.2, 0.1, 0.7, 0.01])
    axflowerorient = plt.axes([0.2, 0.15, 0.7, 0.01])

    fig.subplots_adjust(bottom=0.25)

    sdistance = widgets.Slider(axdistance, 'Distance (m)', 0.01, 3.0, valinit=init_dist, valstep=0.01, orientation='horizontal')
    sbatorient = widgets.Slider(axbatorient, 'Bat Orient (\u00b0)', -180, 180, valinit=0., valstep=0.1, orientation='horizontal')
    sflowerorient = widgets.Slider(axflowerorient, 'Flower Orient (\u00b0)', -180, 180, valinit=0., valstep=0.1, orientation='horizontal')

    def update(val):
        bat_pose = np.zeros(3).astype(float)
        bat_pose[2] = Spatializer.wrapToPi(init_bat_orientation- np.radians(sbatorient.val) )
        cartesian_objects_matrix[0,1] = sdistance.val
        cartesian_objects_matrix[0,2] = Spatializer.wrapToPi(init_flower_orientation - np.radians(sflowerorient.val))
        render.run(bat_pose, cartesian_objects_matrix)
        
        
        ##### Edit image here ################
        flower_artists[0].remove()
        rotated_flower_asset = ndimage.rotate(flower_asset, np.degrees(cartesian_objects_matrix[0,2]))
        flower_imgbox = OffsetImage(rotated_flower_asset, zoom=ASSET_SCALE)
        flower_annobox = AnnotationBbox(flower_imgbox, (cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1]), frameon = False)
        flower_artists[0] = ax1.add_artist(flower_annobox)

        bat_artists[0].remove()
        if render.viewer.collision_status==True:
            bat_pose = np.copy(render.viewer.bat_pose)
            rotated_bat_asset = ndimage.rotate(bat_hit_asset, np.degrees(bat_pose[2]))
        else:
            rotated_bat_asset = ndimage.rotate(bat_asset, np.degrees(bat_pose[2]))
        bat_imgbox = OffsetImage(rotated_bat_asset, zoom=ASSET_SCALE)
        bat_annobox = AnnotationBbox(bat_imgbox, (bat_pose[0], bat_pose[1]), frameon = False)
        bat_artists[0] = ax1.add_artist(bat_annobox)
        
        ######################################
        
        waveform_left_.set_ydata(render.waveform_left)
        waveform_right_.set_ydata(render.waveform_right)
        envelope_left_.set_ydata(render.envelope_left)
        envelope_right_.set_ydata(render.envelope_right)
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
