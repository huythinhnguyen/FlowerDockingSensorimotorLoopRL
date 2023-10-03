from typing import Any
from numpy.typing import ArrayLike
import numpy as np
import sys

from . import Setting 

DEFAULT_BOT = 'Create2'

def wrap_to_pi(a):
    if a > np.pi:
        a -= 2*np.pi
    elif a < -np.pi:
        a += 2*np.pi
    return a


class State:
    def __init__(self, pose: ArrayLike, kinematic: ArrayLike, dt: float,
                 max_linear_velocity: float,
                 max_angular_velocity: float,
                 max_linear_acceleration: float,
                 max_angular_acceleration: float,
                 max_linear_deceleration: float,
                 ):
        """ pose: [x, y, yaw] (meter, rad), kinematic: [v (m/s), w(rad/s)]
            dt = 1/echo_call_rate (s)
        """
        self.pose = np.asarray(pose).astype(np.float32).reshape(3,)
        self.kinematic = np.asarray(kinematic).astype(np.float32).reshape(2,)
        self.previous_kinematics = np.copy(self.kinematic)
        self._init_state = np.concatenate((self.pose, self.kinematic))
        self.dt = dt
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity= max_angular_velocity
        self.max_linear_acceleration = max_linear_acceleration
        self.max_angular_acceleration= max_angular_acceleration
        self.max_linear_deceleration = max_linear_deceleration


    def run(self, *args, **kwargs) -> np.ndarray:
        return self._run(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwds: Any) -> np.ndarray:
        return self.run(*args, **kwds)

    def reset(self, pose: ArrayLike=[], kinematic: ArrayLike=[]):
        """ pose: [x, y, yaw] (meter, rad), kinematic: [v (m/s), w(rad/s)]
        """
        if len(pose)>0: self.pose = np.asarray(pose).astype(np.float32).reshape(3,)
        else: pose = self._init_state[:3]
        if len(kinematic)>0: self.kinematic = np.asarray(kinematic).astype(np.float32).reshape(2,)
        else: kinematic = self._init_state[3:]
        self._init_state = np.concatenate((self.pose, self.kinematic))

    def update_kinematic(self, kinematic: ArrayLike = [], **kwargs):
        """ kinematic: [v (m/s), w(rad/s)]
        """
        if len(kinematic)==0: kinematic = self.kinematic
        velocity_kwd =  set(kwargs.keys()) & set(['v', 'new_v', 'velocity', 'velo'])
        angular_kwd = set(kwargs.keys()) & set(['w', 'new_w', 'rotational_rate', 'rot_rate', 'angular_velocity', 'angular_velo'])
        if velocity_kwd: kinematic[0] = kwargs[velocity_kwd.pop()]
        if angular_kwd: kinematic[1] = kwargs[angular_kwd.pop()]
        kinematic = self._limit_acceleartion(kinematic)
        kinematic = self._limit_velocity(kinematic)
        self.kinematic = np.asarray(kinematic).astype(np.float32).reshape(2,)
        return self.kinematic
    
    def _limit_acceleartion(self, kinematic: ArrayLike):
        v, w = kinematic
        dv = v - self.previous_kinematics[0]
        dw = w - self.previous_kinematics[1]
        dv = np.clip(dv, self.max_linear_deceleration*self.dt, self.max_linear_acceleration*self.dt)
        dw = np.clip(dw, -self.max_angular_acceleration*self.dt, self.max_angular_acceleration*self.dt)
        return np.asarray([self.previous_kinematics[0]+dv, self.previous_kinematics[1]+dw])
    
    def _limit_velocity(self, kinematic: ArrayLike):
        v, w = kinematic
        v = np.clip(v, 0.0, self.max_linear_velocity)
        w = np.clip(w, -self.max_angular_velocity, self.max_angular_velocity)
        return np.asarray([v, w])

    def turning_radius(self):
        if np.abs(self.kinematic[1]) > 1e-6: return self.kinematic[0]/self.kinematic[1]
        else: return 'inf'


    def ICC(self):
        x, y, yaw = self.pose
        R = self.turning_radius()
        if R != 'inf':
            Cx = x - R*np.sin(yaw)
            Cy = y + R*np.cos(yaw)
            return [Cx, Cy]
        else: 
            return None


    def update_pose(self):
        x, y, yaw = self.pose
        v,w = self.kinematic
        ICC = self.ICC()
        if ICC != None:
            turn = w*self.dt
            Rotation = np.array( [[ np.cos(turn), -np.sin(turn), 0.0 ],
                                  [ np.sin(turn),  np.cos(turn), 0.0 ],
                                  [ 0.0         ,   0.0        , 0.0 ]] )
            translation = -1*np.array([*ICC,0]).reshape(3,1)
            inverse_translation = np.array([*ICC, turn]).reshape(3,1)
            new_pose = np.matmul( Rotation,self.pose.reshape(3,1) + translation) + inverse_translation
            new_pose[2,0] = self.pose[2] + turn
        else :
            move = v*self.dt
            translation = move*np.array([np.cos(yaw), np.sin(yaw), 0.0])
            new_pose = self.pose + translation
        self.pose = new_pose.reshape(3,)
        self.pose[2] = wrap_to_pi(self.pose[2])
    
    def _run(self, *args, **kwargs):
        self.update_kinematic(*args, **kwargs)
        self.update_pose()
        return self.pose

    @property
    def max_linear_velocity(self):
        return self._max_linear_velocity
    
    @max_linear_velocity.setter
    def max_linear_velocity(self, value):
        self._max_linear_velocity = value

    @property
    def max_angular_velocity(self):
        return self._max_angular_velocity
    
    @max_angular_velocity.setter
    def max_angular_velocity(self, value):
        self._max_angular_velocity = value

    @property
    def max_linear_acceleration(self):
        return self._max_linear_acceleration
    
    @max_linear_acceleration.setter
    def max_linear_acceleration(self, value):
        self._max_linear_acceleration = value

    @property
    def max_angular_acceleration(self):
        return self._max_angular_acceleration
    
    @max_angular_acceleration.setter
    def max_angular_acceleration(self, value):
        self._max_angular_acceleration = value

    @property
    def max_linear_deceleration(self):
        return self._max_linear_deceleration
    
    @max_linear_deceleration.setter
    def max_linear_deceleration(self, value):
        self._max_linear_deceleration = value


class GPS:
    def __init__(self, var=Setting.steam_gps_var, pose=None, offset=True, noise=True, **kwarg):
        self.var = np.array(var)
        if type(pose)==np.ndarray:
            self.pose = pose.reshape(3,)
        else:
            self.pose = np.array([0.0, 0.0, 0.0]) if pose==None else np.array(pose)
        self.pose_hat = self.pose + np.sqrt(self.var)*np.random.randn(3)
        self.pose_hat[2] = wrap_to_pi(self.pose_hat[2])
        self.init_pose = np.copy(self.pose)
        if offset:
            if len(kwarg)>0:
                self.bot = Setting.Create2(mode=kwarg['mode'], custom_spec = kwarg['custom_spec'], noise=noise)
            else:
                self.bot = Setting.Create2(mode=DEFAULT_BOT, noise=noise)
        self.offset = offset
        

    def reset(self,pose = None):
        self.pose = np.copy(self.init_pose) if pose==None else np.array(pose)
        self.pose_hat = self.pose + np.sqrt(self.var)*np.random.randn(3)
        self.pose_hat[2] = wrap_to_pi(self.pose_hat[2])
        self.init_pose = np.copy(self.pose)
        if self.offset:
            self.bot.reset()
        

    def update(self,pose):
        self.pose = pose
        self.pose_hat = pose + np.sqrt(self.var)*np.random.randn(3)
        offset = self.bot.heading_offset if self.offset else 0.0
        self.pose_hat[2] = wrap_to_pi(self.pose_hat[2] + offset)

