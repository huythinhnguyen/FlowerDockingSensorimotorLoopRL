import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

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
    plt.show()

def main():
    return utest_render_1()

if __name__ == '__main__':
    main()
