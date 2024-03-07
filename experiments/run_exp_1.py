import sys
import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List, Dict, Any, Union, Callable


REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting
from Simulation.Motion import State

from Perception.flower_pose_estimator import NaiveOneShotFlowerPoseEstimator, OnsetOneShotFlowerPoseEstimator, TwoShotFlowerPoseEstimator
from SensorimotorLoops.home_in_flower import HomeInFlower

FLOWER_DISTANCE_SETTINGS = {
    'close': 0.8, # meter
    'mid':   1.6, # meter
    'far':   2.4, # meter
}

BAT_AZIMUTH_RANGE = (-np.pi/3, np.pi/3)
FLOWER_COLLISION_RADIUS = 0.25
FLOWER_OPENING_ANGULAR_RANGE = (-np.pi/6, np.pi/6)
BAT_FACING_ANGULAR_RANGE = (-np.pi/3, np.pi/3)
ARENA_LIM = {'x': (-1., 4.), 'y': (-1., 4.)}
N_TRIAL = 1_000

# TESTED
def collision_check(bat_pose: ArrayLike, objects: ArrayLike,
                    collision_distance: float = FLOWER_COLLISION_RADIUS) -> bool:
    if np.any(np.linalg.norm(objects[:,:2]-bat_pose[:2], axis=1) <= collision_distance):
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

# make some graphical test. --> I think it's GOOD
def test1():
    # make a interactive widget to plot arrows showing the bat and flower pose.
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.widgets import Slider
    from matplotlib import animation
    from matplotlib.animation import FuncAnimation
    from matplotlib import patches
    from matplotlib import transforms
    from matplotlib import cm

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Bat and Flower Pose')

    bat_pose = np.array([0., 0., 0.])
    flower_pose = np.array([1., 0., 0.])
    bat_arrow = FancyArrowPatch((bat_pose[0], bat_pose[1]), (bat_pose[0]+0.3*np.cos(bat_pose[2]), bat_pose[1]+0.3*np.sin(bat_pose[2])),
                                arrowstyle='simple', mutation_scale=20, color='blue')
    flower_arrow = FancyArrowPatch((flower_pose[0], flower_pose[1]), (flower_pose[0]+0.3*np.cos(flower_pose[2]), flower_pose[1]+0.3*np.sin(flower_pose[2])),
                                      arrowstyle='simple', mutation_scale=20, color='red')
    ax.add_patch(bat_arrow)
    ax.add_patch(flower_arrow)
    fig.subplots_adjust(bottom=0.3)
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
    ax_theta = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)
    ax_flower_theta = plt.axes([0.25, 0.16, 0.65, 0.03], facecolor=axcolor)
    
    # add a changeable textbox show the sign of bat pose [2]

    status_text = ax.text(2.2, 1.8, 'STATUS: ??', ha='left', va='center', fontsize=14)
    distance_text = ax.text(2.2, 1.5, 'd = ??', ha='left', va='center', fontsize=14)
    aoa_text = ax.text(2.2, 1.2, 'aoa = ??', ha='left', va='center', fontsize=14)
    azi_text = ax.text(2.2, 0.8, 'azi = ??', ha='left', va='center', fontsize=14)

    s_x = Slider(ax_x, 'x', ARENA_LIM['x'][0], ARENA_LIM['x'][1], valinit=bat_pose[0])
    s_y = Slider(ax_y, 'y', ARENA_LIM['y'][0], ARENA_LIM['y'][1], valinit=bat_pose[1])
    s_theta = Slider(ax_theta, 'theta', -np.pi, np.pi, valinit=bat_pose[2])
    s_flower_theta = Slider(ax_flower_theta, 'flower_theta', -np.pi, np.pi, valinit=flower_pose[2])

    def update(val):
        bat_pose[0] = s_x.val
        bat_pose[1] = s_y.val
        bat_pose[2] = s_theta.val
        flower_pose[2] = s_flower_theta.val
        bat_arrow.set_positions((bat_pose[0], bat_pose[1]), (bat_pose[0]+0.3*np.cos(bat_pose[2]), bat_pose[1]+0.3*np.sin(bat_pose[2])))
        flower_arrow.set_positions((flower_pose[0], flower_pose[1]), (flower_pose[0]+0.3*np.cos(flower_pose[2]), flower_pose[1]+0.3*np.sin(flower_pose[2])))
        fig.canvas.draw_idle()
        distance = np.linalg.norm(bat_pose[:2]-flower_pose[:2])
        distance_text.set_text(f'd = {np.round(distance, 3)}')
        aoa = get_angle_of_arrival(bat_pose, flower_pose)
        aoa_text.set_text(f'aoa = {np.round(np.degrees(aoa), 1)}')
        azi = get_azimuth_of_flower(bat_pose, flower_pose)
        azi_text.set_text(f'azi = {np.round(np.degrees(azi), 1)}')
        if collision_check(bat_pose, np.array([flower_pose])):
            if check_docking(bat_pose, flower_pose):
                status_text.set_text('STATUS: Dock')
            else:
                status_text.set_text('STATUS: Hit')
        else: status_text.set_text('STATUS: OK')


    s_x.on_changed(update)
    s_y.on_changed(update)
    s_theta.on_changed(update)
    s_flower_theta.on_changed(update)

    plt.show()


def run(distance_setting: str):
    print(sys.argv)
    print('distance_setting:', distance_setting)


def main():
    #return run(sys.argv[1])
    return test1()

if __name__=='__main__':
    main()