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

BAT_ASSET_PATH = os.path.join(REPO_PATH, 'assets', 'bat.png')
FLOWER_ASSET_PATH = os.path.join(REPO_PATH, 'assets', 'flower.png')

def moving_around_plot():
    init_bat_orientation = np.radians(90)
    init_dist = 1.0
    bat_pose = np.asarray([0., 0., init_bat_orientation])
    init_flower_orientation = np.radians(-90)
    cartesian_objects_matrix = np.asarray([[0., init_dist, init_flower_orientation, 3.],
                                           ])
    #flower_arrow_length = 0.2
    bat_arrow_length = 0.3
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot()
    # bat_arrow = ax1.arrow(bat_pose[0], bat_pose[1],
    #         bat_arrow_length*np.cos(bat_pose[2]), bat_arrow_length*np.sin(bat_pose[2]),
    #         width=0.05, head_width=0.1, head_length=0.05, fc='k', ec='k')
    # flower_arrow = ax1.arrow(cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1],
    #         flower_arrow_length*np.cos(cartesian_objects_matrix[0,2]), flower_arrow_length*np.sin(cartesian_objects_matrix[0,2]),
    #         width=0.05, head_width=0.1, head_length=0.05, fc='m', ec='m')
    ######################################
    ##### Add image here ##################
    flower_artists = []
    bat_artists = []
    flower_asset = mpimg.imread(FLOWER_ASSET_PATH)
    bat_asset = mpimg.imread(BAT_ASSET_PATH)

    rotated_flower_asset = ndimage.rotate(flower_asset, np.degrees(cartesian_objects_matrix[0,2]))
    flower_imgbox = OffsetImage(rotated_flower_asset, zoom=0.1)
    flower_annobox = AnnotationBbox(flower_imgbox, (cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1]), frameon = False)
    flower_artists.append(ax1.add_artist(flower_annobox))

    rotated_bat_asset = ndimage.rotate(bat_asset, np.degrees(bat_pose[2]))
    bat_imgbox = OffsetImage(rotated_bat_asset, zoom=0.1)
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

        # bat_arrow.set_data(x=bat_pose[0], y=bat_pose[1],
        #     dx=bat_arrow_length*np.cos(bat_pose[2]), dy=bat_arrow_length*np.sin(bat_pose[2]))
        # flower_arrow.set_data(x=cartesian_objects_matrix[0,0], y=cartesian_objects_matrix[0,1],
        #     dx=flower_arrow_length*np.cos(cartesian_objects_matrix[0,2]), dy=flower_arrow_length*np.sin(cartesian_objects_matrix[0,2]))
        ######################################
        ##### Edit image here ################
        flower_artists[0].remove()
        rotated_flower_asset = ndimage.rotate(flower_asset, np.degrees(cartesian_objects_matrix[0,2]))
        flower_imgbox = OffsetImage(rotated_flower_asset, zoom=0.1)
        flower_annobox = AnnotationBbox(flower_imgbox, (cartesian_objects_matrix[0,0], cartesian_objects_matrix[0,1]), frameon = False)
        flower_artists[0] = ax1.add_artist(flower_annobox)

        bat_artists[0].remove()
        rotated_bat_asset = ndimage.rotate(bat_asset, np.degrees(bat_pose[2]))
        bat_imgbox = OffsetImage(rotated_bat_asset, zoom=0.1)
        bat_annobox = AnnotationBbox(bat_imgbox, (bat_pose[0], bat_pose[1]), frameon = False)
        bat_artists[0] = ax1.add_artist(bat_annobox)
        
        ######################################
        
        fig.canvas.draw_idle()
    sdistance.on_changed(update)
    sbatorient.on_changed(update)
    sflowerorient.on_changed(update)
    plt.show()



def main():
    return moving_around_plot()

if __name__ == '__main__':
    main()