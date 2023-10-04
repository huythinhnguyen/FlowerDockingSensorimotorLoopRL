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


OBSERVATION_SIZE = int(2*sensorconfig.AcousticSetting.RAW_DATA_LENGTH/sensorconfig.CompressionSetting.N_SAMPLE)
CHIRP_RATE = 40 #Hz

def init_bat_pose():
    return np.asarray([0, -2, np.radians(90)]).astype(np.float32)

def init_objects():
    return np.asarray([[0, 0, np.radians(90), 3]]).astype(np.float32)

def check_collision(polar_objects, colision_distance=0.18):
    if len(polar_objects)==0: return False
    else: return np.any(polar_objects[:,0]<colision_distance)

def get_flower_idx(polar_objects):
    return np.where(polar_objects[:,3]==3)

def check_facing_flower(polar_objects, bat_opening_angle=np.radians(10), flower_opening_angle=np.radians(30)):
    if len(polar_objects)==0: return False
    flower_idx = get_flower_idx(polar_objects)
    if len(polar_objects[flower_idx])==0: return False
    answer = np.any(np.abs(polar_objects[flower_idx][:,1])<bat_opening_angle)
    answer = answer and np.any(np.abs(polar_objects[flower_idx][:,2])<flower_opening_angle)
    return answer


class SimpleFlowerDocking(py_environment.PyEnvironment):
    def __init__(self,
                 continuous_action=False, 
                 start_penalty_step=200, 
                 max_step=600, time_penalty=False,
                 max_v=2., max_w=np.pi*4,):
        self.start_penalty_step = start_penalty_step
        self.max_step = max_step
        self.max_v = max_v
        self.max_w = max_w
        if time_penalty: self.time_penalty = 1/(self.max_step - self.start_penalty_step)
        else: self.time_penalty = 0
        init_pose = init_bat_pose()
        self.objects = init_objects()
        self.motion = State(pose = init_pose, dt=1/CHIRP_RATE)
        self.render = Spatializer.Render()
        if continuous_action:
            self.controller = ContinuousActions1(self.max_v, self.max_w, [0, 1], [0, 1])
            self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=0, maximum=1, name='action')
        else:
            self.controller = DiscreteActions1(max_v=self.max_v*0.5, max_w=self.max_w*0.5)
            self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(OBSERVATION_SIZE,), dtype=np.float32, minimum=0, name='observation')
        self._state = np.zeros(OBSERVATION_SIZE)
        self._episode_ended = False
        self.step_count = 0
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _step(self, action, **kwargs):
        reward = 0
        if self._episode_ended:
            return self._reset()
        self.step_count += 1
        # Open your eyes, see where you are
        self.render.run(pose=self.motion.pose, objects=self.objects)
        self._state = np.concatenate((self.render.compress_left, self.render.compress_right))
        # Move according to action
        v, omega = self.controller(action)
        self.motion(v=v, w=omega)
        # If see nothing, end episode, penalize heavily
        # We want agents to learn to always see something
        if len(self.render.filtered_objects_inview_polar)==0:
            reward -= 2
            self._episode_ended = True
        if check_collision(self.render.filtered_objects_inview_polar):
            self._episode_ended = True
            if check_facing_flower(self.render.filtered_objects_inview_polar): reward += 1
            else: reward -= 1
        if self.step_count >= self.max_step: self._episode_ended = True
        if self.step_count>self.start_penalty_step: reward -= self.time_penalty
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.)

    def _reset(self, **kwargs):
        self._episode_ended=False
        self.step_count = 0
        init_pose = init_bat_pose()
        init_objects = init_objects()
        self.motion.reset(pose=init_pose)
        self.objects = init_objects
        self._state = np.zeros(OBSERVATION_SIZE)
        return ts.restart(self._state)
