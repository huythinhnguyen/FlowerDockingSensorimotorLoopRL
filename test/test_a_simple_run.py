import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import SeriesEncoding as Setting

from Simulation.Motion import State

GRAVI_ACCEL = 9.8

MAX_LINEAR_VELOCITY = 5 # robot max velo is 0.5 m/s
MIN_LINEAR_VELOCITY = 0.01 # robot min velo is 0.01 m/s
MAX_ANGULAR_VELOCITY = 100*np.pi
MAX_ANGULAR_ACCELERATION = 1*MAX_ANGULAR_VELOCITY
LINEAR_VELOCITY_OFFSET = 0.
DECELERATION_FACTOR = 1 # Choose between 1 to 5 the higher the steeper the deceleration
CENTRIFUGAL_ACCEL = 3 * GRAVI_ACCEL
CHIRP_RATE = 40 # Hz

### OTHERS KINEMATIC PARAMETERS  ##################################
LINEAR_DECEL_LIMIT = -1.1 * MAX_LINEAR_VELOCITY
LINEAR_ACCEL_LIMIT = 1 * GRAVI_ACCEL

def simple_collision_check_from_polar_objects(polar_objects, limit = 0.18):
    """
    Check if there is any object within the limit distance.
    """
    if len(polar_objects) == 0: return False
    return np.any(polar_objects[:,0] <= limit)

def initialize():
    bat_pose = np.asarray([0., -1.5, np.radians(30)])
    objects = np.asarray([[0., 0., np.radians(90), 3.],])
    linear_velocities = [1.]
    angular_velocities = np.linspace(np.pi/20, np.pi*1.5, 500)
    return bat_pose, objects, linear_velocities, angular_velocities


def run_simple_episode(nstep = 100):
    bat_pose, objects, linear_velocities, angular_velocities = initialize()
    bat_trajectory = np.asarray([]).astype(np.float32).reshape(0,3)
    waveforms_trajectory = {'left': np.asarray([]).astype(np.float32).reshape(0, Setting.DISTANCE_ENCODING.shape[0]),
                            'right': np.asarray([]).astype(np.float32).reshape(0, Setting.DISTANCE_ENCODING.shape[0]),}
    envelope_trajectory = {'left': np.asarray([]).astype(np.float32).reshape(0, Setting.DISTANCE_ENCODING.shape[0]),
                            'right': np.asarray([]).astype(np.float32).reshape(0, Setting.DISTANCE_ENCODING.shape[0]),}
    compress_trajectory = {'left': np.asarray([]).astype(np.float32).reshape(0, Setting.COMPRESSED_DISTANCE_ENCODING.shape[0]),
                            'right': np.asarray([]).astype(np.float32).reshape(0, Setting.COMPRESSED_DISTANCE_ENCODING.shape[0]),}
    polar_objects_trajectory = []
    state = State(pose=bat_pose, kinematic=[linear_velocities[0], angular_velocities[0]], dt=1/CHIRP_RATE,
                 max_linear_velocity=MAX_LINEAR_VELOCITY,
                 max_angular_velocity=MAX_ANGULAR_VELOCITY,
                 max_linear_acceleration=LINEAR_ACCEL_LIMIT,
                 max_angular_acceleration=MAX_ANGULAR_ACCELERATION,
                 max_linear_deceleration=LINEAR_DECEL_LIMIT,
                 )
    render = Spatializer.Render()
    for i in range(nstep):
        bat_trajectory = np.vstack((bat_trajectory, state.pose))
        render.run(state.pose, objects)
        waveforms_trajectory['left'] = np.vstack((waveforms_trajectory['left'], render.waveform_left))
        waveforms_trajectory['right'] = np.vstack((waveforms_trajectory['right'], render.waveform_right))

        envelope_trajectory['left'] = np.vstack((envelope_trajectory['left'], render.envelope_left))
        envelope_trajectory['right'] = np.vstack((envelope_trajectory['right'], render.envelope_right))
        compress_trajectory['left'] = np.vstack((compress_trajectory['left'], render.compress_left))
        state(v=linear_velocities[i%len(linear_velocities)], w=angular_velocities[i%len(angular_velocities)])
        polar_objects_trajectory.append(render.viewer.filtered_objects_inview_polar)
        #if len(polar_objects_trajectory[-1]) > 0:
        #    print('step={}, dist={:.2f}, angle={:.2f}'.format(i,
        #    polar_objects_trajectory[-1][0,0], 
        #    np.degrees(polar_objects_trajectory[-1][0,1])))
        #collision check
        if simple_collision_check_from_polar_objects(render.viewer.filtered_objects_inview_polar):
            print('Collision!')
            break
    return bat_trajectory, objects, waveforms_trajectory, envelope_trajectory, compress_trajectory, polar_objects_trajectory


def plot_trajectory(bat_trajectory, objects, waveforms_trajectory, envelope_trajectory, compress_trajectory, polar_objects_trajectory):
    from matplotlib.widgets import Slider, Button, RadioButtons
    from matplotlib.gridspec import GridSpec
    init_timestep = 0
    fig = plt.figure(figsize=(12, 6), dpi=200)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    trajactory_line_, = ax1.plot(bat_trajectory[:init_timestep,0], bat_trajectory[:init_timestep,1], linewidth=2, alpha=1)
    #plot the last bat pose as an arrow
    bat_arrow = ax1.arrow(bat_trajectory[init_timestep,0], bat_trajectory[init_timestep,1],
            np.cos(bat_trajectory[init_timestep,2]), np.sin(bat_trajectory[init_timestep,2]),
            width=0.01,length_includes_head=0.1, head_width=0.1, head_length=0.05, fc='k', ec='k')
    ax1.scatter(objects[:,0], objects[:,1])
    ax1.set_xlim([-5,5])
    ax1.set_ylim([-5,5])
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    ax1.set_title('Bat trajectory')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Waveforms')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Amplitude')
    waveform_left_, = ax2.plot(Setting.DISTANCE_ENCODING, waveforms_trajectory['left'][init_timestep, :], linewidth=0.5, alpha=0.5)
    waveform_right_, = ax2.plot(Setting.DISTANCE_ENCODING, waveforms_trajectory['right'][init_timestep, :], linewidth=0.5, alpha=0.5)
    ax2.set_ylim([-1500, 1500])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Envelopes')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Amplitude')
    envelope_left_, = ax3.plot(Setting.DISTANCE_ENCODING, envelope_trajectory['left'][init_timestep, :], linewidth=0.5, alpha=0.5)
    envelope_right_, = ax3.plot(Setting.DISTANCE_ENCODING, envelope_trajectory['right'][init_timestep, :], linewidth=0.5, alpha=0.5)
    #compress_left_, = ax3.stem(Setting.COMPRESSED_DISTANCE_ENCODING, compress_trajectory['left'][init_timestep, :],basefmt=':', markerfmt='b.', linefmt='-',)
    #compress_right_,= ax3.stem(Setting.COMPRESSED_DISTANCE_ENCODING, compress_trajectory['right'][init_timestep, :],basefmt=':', markerfmt='r.', linefmt='-',)

    axtime = plt.axes([0.1, 0.1, 0.8, 0.01])
    stime = Slider(axtime, 'Time', 0, bat_trajectory.shape[0]-1, valinit=init_timestep, valstep=1)

    def update(val):
        timestep = int(stime.val)
        trajactory_line_.set_xdata(bat_trajectory[:timestep,0])
        trajactory_line_.set_ydata(bat_trajectory[:timestep,1])
        bat_arrow.set_data(x=bat_trajectory[timestep,0], y=bat_trajectory[timestep,1],
            dx=np.cos(bat_trajectory[timestep,2]), dy=np.sin(bat_trajectory[timestep,2]))
        waveform_left_.set_ydata(waveforms_trajectory['left'][timestep, :])
        waveform_right_.set_ydata(waveforms_trajectory['right'][timestep, :])
        envelope_left_.set_ydata(envelope_trajectory['left'][timestep, :])
        envelope_right_.set_ydata(envelope_trajectory['right'][timestep, :])
        if len(polar_objects_trajectory[timestep]) > 0:
            sys.stdout.writelines('step={}, dist={:.2f}, angle={:.2f}, orient={:.2f}     \r'.format(timestep,
                                    np.degrees(polar_objects_trajectory[timestep][0,0]), 
                                    np.degrees(polar_objects_trajectory[timestep][0,1]),
                                    np.degrees(polar_objects_trajectory[timestep][0,2])))
        else: sys.stdout.writelines('EMPTY VIEW                                          \r')
        #compress_left_.set_ydata(compress_trajectory['left'][timestep, :])
        #compress_right_.set_ydata(compress_trajectory['right'][timestep, :])
        fig.canvas.draw_idle()
    stime.on_changed(update)
    plt.show()


def test1():
    bat_trajectory, objects, waveforms_trajectory, envelope_trajectory, compress_trajectory, polar_objects_trajectory = run_simple_episode(nstep=200)
    plot_trajectory(bat_trajectory, objects, waveforms_trajectory, envelope_trajectory, compress_trajectory, polar_objects_trajectory)



def main():
    #return print(simple_collision_check_from_polar_objects(np.asarray([[float(sys.argv[1]), np.pi/10, np.pi/5],
    #                                                             [float(sys.argv[2]), np.pi/10, np.pi/5]])))
    return test1()


if __name__=='__main__':
    main()