import numpy as np
import os
import sys

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)


from Sensors.FlowerEchoSimulator import Spatializer
from Controller.SimpleController import DiscreteActions1, ContinuousActions1
#from Control.SensorimotorLoops.BatEcho import AvoidApproach
from Sensors.FlowerEchoSimulator import Setting as sensorconfig
#from Control.SensorimotorLoops import Setting as controlconfig
from Simulation.Motion import State



OBSERVATION_SIZE = 300
MEMORY_SIZE = 3

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

def init_bat_pose(level):
    if level<1:
        return np.asarray([0, 0.5, 
                           np.radians(90+ np.random.uniform(-10, 10))]).astype(np.float64)
    elif level<2:
        return np.asarray([0, 0.8, 
                           np.radians(90+ np.random.uniform(-10, 10))]).astype(np.float64)
    else:
        return np.asarray([0, 1.2, 
                           np.radians(90+ np.random.uniform(-10, 10))]).astype(np.float64)


def init_objects(level):
    if level<3:
        return np.asarray([[0, 0., 
                           np.radians(-90+ np.random.uniform(-10, 10))]]).astype(np.float64)
    elif level<4:
        return np.asarray([[0, 0., 
                           np.radians(-90+ np.random.uniform(-40, 40))]]).astype(np.float64)
    else:
        return np.asarray([[0, 0., 
                           np.radians(-90+ np.random.uniform(-70, 70))]]).astype(np.float64)

def check_collision(polar_objects, colision_distance=0.35):
    if len(polar_objects)==0: return False
    else: return np.any(polar_objects[:,0]<colision_distance)

def get_flower_idx(polar_objects):
    return np.where(polar_objects[:,3]==3)

def check_facing_flower(polar_objects, bat_opening_angle=np.radians(30), flower_opening_angle=np.radians(30)):
    if len(polar_objects)==0: return False
    flower_idx = get_flower_idx(polar_objects)
    if len(polar_objects[flower_idx])==0: return False
    answer = np.any(np.abs(polar_objects[flower_idx][:,1])<bat_opening_angle)
    answer = answer and np.any(np.abs(polar_objects[flower_idx][:,2])<flower_opening_angle)
    return answer


class SimpleFlowerDocking(py_environment.PyEnvironment):
    def __init__(self,
                 continuous_action=False, 
                 start_penalty_step=50, 
                 max_step=300, time_penalty=True,
                 max_v=1.6, max_w=np.pi*4, init_level=0.):
        self.start_penalty_step = start_penalty_step
        self.max_step = max_step
        self.max_v = max_v
        self.max_w = max_w
        self.level = init_level
        if time_penalty: self.time_penalty = 2/(self.max_step - self.start_penalty_step)
        else: self.time_penalty = 0
        init_pose = init_bat_pose(self.level)
        self.objects = init_objects(self.level)
        self.motion = State(pose = init_pose, kinematic=[0., 0.], dt=1/CHIRP_RATE,
                            max_linear_velocity=MAX_LINEAR_VELOCITY,
                            max_angular_velocity=MAX_ANGULAR_VELOCITY,
                            max_linear_acceleration=LINEAR_ACCEL_LIMIT,
                            max_angular_acceleration=MAX_ANGULAR_ACCELERATION,
                            max_linear_deceleration=LINEAR_DECEL_LIMIT,
                            )

        self.render = Spatializer.Render()
        if continuous_action:
            self.controller = ContinuousActions1(self.max_v, self.max_w, [0, 1], [0, 1])
            self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float64, minimum=0, maximum=1, name='action')
        else:
            self.controller = DiscreteActions1(v=self.max_v*0.5, w=self.max_w*0.5)
            self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(OBSERVATION_SIZE,), dtype=np.float64, minimum=0, name='observation')
        self.render(self.motion.pose, self.objects)
        self._state = np.empty((OBSERVATION_SIZE,), dtype=np.float64)
        for i in range(MEMORY_SIZE):
            self._state[i*100:(i+1)*100] = np.concatenate((self.render.compress_left[:50], self.render.compress_right[:50])).astype(np.float64)
        self._episode_ended = False
        self.step_count = 0
    
    @property
    def level(self):
        return self._level
    
    @level.setter
    def level(self, value):
        self._level = value

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _step(self, action, **kwargs):
        reward = 0
        if self._episode_ended:
            return self._reset()
        self.step_count += 1
        # Move according to action
        v, omega = self.controller(action)
        self.motion(v=v, w=omega)
        # Open your eyes, see where you are
        self.render(self.motion.pose, self.objects)
        # If see nothing, end episode, penalize heavily
        # We want agents to learn to always see something
        if len(self.render.viewer.filtered_objects_inview_polar)==0:
            reward -= 2
            self._episode_ended = True
        if check_collision(self.render.viewer.filtered_objects_inview_polar):
            self._episode_ended = True
            if check_facing_flower(self.render.viewer.filtered_objects_inview_polar): 
                reward += 2    
            else: reward -= 1
        if self.step_count >= self.max_step: self._episode_ended = True
        if self.step_count>self.start_penalty_step: reward -= self.time_penalty
        compress_left = self.render.compress_left[:50]
        compress_left[np.isnan(compress_left)] = 0
        compress_left[np.isinf(compress_left)] = 0
        compress_right = self.render.compress_right[:50]
        compress_right[np.isnan(compress_right)] = 0
        compress_right[np.isinf(compress_right)] = 0
        for i in range(MEMORY_SIZE-1):
            self._state[(i+1)*100:(i+2)*100] = self._state[i*100:(i+1)*100]
        self._state[:100] = np.concatenate((compress_left, compress_right)).astype(np.float64)
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.)

    def _reset(self, **kwargs):
        self._episode_ended=False
        self.step_count = 0
        init_pose = init_bat_pose(self.level)
        objects = init_objects(self.level)
        self.motion.reset(pose=init_pose)
        self.objects = objects
        self._state = np.empty((OBSERVATION_SIZE,), dtype=np.float64)
        for i in range(MEMORY_SIZE):
            self._state[i*100:(i+1)*100] = np.concatenate((self.render.compress_left[:50], self.render.compress_right[:50])).astype(np.float64)
        return ts.restart(self._state)
